"""追补（增量同步）：按原始记录 ID 比较两个版本的证据表，交付净变更。

语义
----
- 比较的是两端的**证据表**（原始记录身份 + 批次 + 原始方向测点位置 +
  原始偏差/误差），不是折算后的矩阵读数，也不是逐轮操作流水：跨轮撤回再
  用同一 ID，即便折算值没变，只要原始证据不同就交付；中途出现又消失的
  ID、最终恢复为起点同样内容的 ID 都不交付。
- 每个快照支持从它往前最多 ``CATCHUP_WINDOW``（16）次成功订正的追补
  （含恰好 16 次）；成功但未改动证据的订正也占一轮。
- 追补起点（anchor）：新建数据集为其初始版本；已有数据目录第一次用本版
  打开时为当时的当前版本，持久化在 ``meta.json``，以后重开不重置。

存储
----
追补历史与版本目录分离，放在 ``<root>/changelog/``：

- ``meta.json``：``{"format": 1, "anchor": N}``。
- ``c_XXXXXXXX.json``：一轮成功订正的证据增撤：
  ``{"seq": N, "adds": {rid: [batch, src, dst, raw_d, raw_e], ...},
  "removes": {rid: [...], ...}}``。removes 记录被撤证据撤掉前的原始内容，
  使从当前版本向任意历史端重放都可行。文件一旦发布不再修改。

一致性
------
每轮订正在切换 ``CURRENT`` 之前把本轮 changelog 原子落盘（tmp + fsync +
replace + 目录 fsync），因此已发布的订正重开后仍可追补；被拒绝或在切换
``CURRENT`` 前中断的订正至多留下 ``seq > CURRENT`` 的残留文件，打开时
清理，不会出现在追补里。

保留
----
不为追补延期保留完整的历史矩阵目录：版本目录仍按原有规则回收。changelog
轮次文件跟随读事务钉版本回收——保留当前版本与所有活动快照各自往前
``CATCHUP_WINDOW`` 轮所需的范围，其余删除。

成本
----
提交侧只写本轮增撤（来自事务本身，不做整仓比对）；一次追补只重放请求
跨度（以及目标版本到当前版本）内发生过变化的记录与相关索引，不遍历未变
的证据。
"""

from __future__ import annotations

import json
import os

from .errors import DatasetError
from .model import CatchupChange, CatchupResult, EvidenceRef

CATCHUP_WINDOW = 16
_FORMAT = 1


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _changelog_dir(path):
    return os.path.join(path, "changelog")


def _round_name(seq):
    return f"c_{seq:08d}.json"


def _parse_round_name(name):
    if not name.startswith("c_") or not name.endswith(".json"):
        return None
    try:
        return int(name[2 : -len(".json")])
    except ValueError:
        return None


def ensure_catchup_state(path, current_seq):
    """打开/建立追补状态，返回追补起点（anchor）版本号。

    已有状态直接读取（重开不重置起点）；没有则以当前版本为起点建立
    （已有数据目录第一次用本版打开的情形）。同时清理 seq 大于当前版本
    的残留轮次文件（切换 CURRENT 前中断的订正）。
    """
    cdir = _changelog_dir(path)
    os.makedirs(cdir, exist_ok=True)
    meta_path = os.path.join(cdir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") != _FORMAT or not isinstance(meta.get("anchor"), int):
            raise DatasetError("追补状态文件损坏: changelog/meta.json")
        anchor = meta["anchor"]
        if anchor > current_seq:
            raise DatasetError("追补起点晚于当前版本，追补状态损坏")
    else:
        anchor = current_seq
        tmp = meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"format": _FORMAT, "anchor": anchor}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, meta_path)
        _fsync_dir(cdir)
    for name in os.listdir(cdir):
        seq = _parse_round_name(name)
        if seq is not None and seq > current_seq:
            # 切换 CURRENT 前中断的订正留下的轮次文件
            os.remove(os.path.join(cdir, name))
        elif seq is None and name.endswith(".tmp"):
            # 原子写入被中断留下的临时文件
            os.remove(os.path.join(cdir, name))
    return anchor


def write_round(path, seq, adds, removes):
    """把一轮成功订正的证据增撤原子落盘（必须在切换 CURRENT 之前调用）。

    adds/removes：record_id -> (batch, src, dst, raw_d, raw_e)，
    removes 为被撤证据撤掉前的原始内容。成功但未改动证据的订正也写一条
    空记录，同样占一轮。
    """
    cdir = _changelog_dir(path)
    payload = {
        "seq": seq,
        "adds": {rid: list(content) for rid, content in adds.items()},
        "removes": {rid: list(content) for rid, content in removes.items()},
    }
    tmp = os.path.join(cdir, _round_name(seq) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(cdir, _round_name(seq)))
    _fsync_dir(cdir)


def load_round(path, seq):
    """读取一轮 changelog，返回 (adds, removes)。

    两者均为 dict：record_id -> (batch, src, dst, raw_d, raw_e)。
    """
    fpath = os.path.join(_changelog_dir(path), _round_name(seq))
    try:
        with open(fpath, encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        raise DatasetError(f"追补历史缺失：版本 {seq} 的变更记录不存在") from None
    if payload.get("seq") != seq:
        raise DatasetError(f"追补历史损坏：版本 {seq} 的变更记录序号不符")
    adds = {rid: tuple(c) for rid, c in payload.get("adds", {}).items()}
    removes = {rid: tuple(c) for rid, c in payload.get("removes", {}).items()}
    return adds, removes


def gc_rounds(path, keep_seqs):
    """回收不再被当前版本或任何活动快照需要的轮次文件。

    keep_seqs 为当前版本与所有被读事务钉住的版本号；保留其中最小者往前
    CATCHUP_WINDOW 轮起的范围。
    """
    cdir = _changelog_dir(path)
    if not os.path.isdir(cdir):
        return
    min_keep = min(keep_seqs) - CATCHUP_WINDOW
    for name in os.listdir(cdir):
        seq = _parse_round_name(name)
        if seq is not None and seq < min_keep:
            os.remove(os.path.join(cdir, name))


def _record_content(version, rid):
    """record_id 在给定版本中的原始证据内容；不存在返回 None。"""
    t = version.rec_index.get(rid)
    if t is None:
        return None
    k = version.fact_of_entry(t)
    for ev in version.facts[(version.I[k], version.J[k])][2]:
        if ev[0] == rid:
            return ev[1:]
    raise DatasetError("记录索引与事实表不一致，数据集损坏")


def compute_catchup(path, cursor, target_seq, current_version):
    """计算 (cursor, target_seq] 跨度内的净变更，返回 CatchupResult。

    current_version 为调用方持有的当前 Version（不可变）；目标端内容从
    当前版本的记录索引取出后向目标版本倒放，起点端内容再向游标倒放。
    只处理跨度内发生过变化的记录，不遍历未变的证据。
    """
    # 跨度内各轮增撤（正序），并收集发生过变化的记录集合
    rounds = []
    touched = set()
    for seq in range(cursor + 1, target_seq + 1):
        adds, removes = load_round(path, seq)
        rounds.append((adds, removes))
        touched.update(adds)
        touched.update(removes)
    stats = {"records_touched": len(touched), "rounds": target_seq - cursor}
    if not touched:
        # 自身追补或跨度内订正都未改动证据：空差异，无需任何倒放
        return CatchupResult(cursor, target_seq, (), stats)
    # 目标端内容：先取这些记录在当前版本的内容
    new_state = {}
    for rid in touched:
        content = _record_content(current_version, rid)
        if content is not None:
            new_state[rid] = content
    # 目标版本更旧时，逐轮倒放 (target, current]：撤掉新增、恢复被撤
    for seq in range(current_version.seq, target_seq, -1):
        adds, removes = load_round(path, seq)
        for rid in adds:
            if rid in touched:
                new_state.pop(rid, None)
        for rid, content in removes.items():
            if rid in touched:
                new_state[rid] = content
    # 起点端内容：从目标端倒放 (cursor, target]
    old_state = dict(new_state)
    for adds, removes in reversed(rounds):
        for rid in adds:
            old_state.pop(rid, None)
        for rid, content in removes.items():
            old_state[rid] = content
    # 净变更：两端内容不同的记录各一条，按记录 ID 字符串序交付
    changes = []
    for rid in sorted(touched):
        old = old_state.get(rid)
        new = new_state.get(rid)
        if old != new:
            changes.append(
                CatchupChange(
                    rid,
                    EvidenceRef(rid, *old) if old is not None else None,
                    EvidenceRef(rid, *new) if new is not None else None,
                )
            )
    return CatchupResult(cursor, target_seq, tuple(changes), stats)
