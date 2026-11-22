"""可订正的落盘成对比测数据集。

- 测点：创建时给定稳定且互不相同的代码，排序位置为 0..N-1 稠密下标，N<=1e8。
- 事实：有向记录 (a,b,d,e) 折算到规范方向（位置小->大）只存一份；
  反向读取折算 (-d, e)，盘上不存正反两份。
- 首装与订正共用同一条事务路径：同对无论方向，只在折算值相同条件下
  合并证据；提交时校验剩余支持相互一致，冲突拒绝发布（不会用后面的行盖掉）。
- MVCC：每次提交生成新的不可变版本目录并原子切换 CURRENT；
  读事务钉住开始时的版本，与写者并行也不会读出新旧混搭；
  只保留当前版本与被读事务钉住的版本，其余由 GC 清理。
"""

from __future__ import annotations

import json
import os
import shutil
import threading

from .errors import (
    ConflictError,
    DatasetError,
    DuplicateRecordError,
    StaleTransactionError,
    UnknownRecordError,
    ValidationError,
)
from .model import FORWARD, REVERSE, SELF, Cell, WindowResult
from .version import Version

MAX_STATIONS = 100_000_000  # 测点不超过一亿个
_INT64_MIN = -(2**63)


def _version_name(seq):
    return f"v_{seq:08d}"


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_stations(stations):
    if not stations:
        raise ValidationError("测点列表不能为空")
    if len(stations) > MAX_STATIONS:
        raise ValidationError("测点数量超过一亿上限")
    seen = set()
    for code in stations:
        if not isinstance(code, str) or not code:
            raise ValidationError("测点代码必须为非空字符串")
        if "\n" in code or "\r" in code:
            raise ValidationError("测点代码不能包含换行符")
        if code in seen:
            raise ValidationError(f"测点代码重复: {code!r}")
        seen.add(code)


def _validate_record_id(record_id):
    if not isinstance(record_id, str) or not record_id:
        raise ValidationError("记录身份必须为非空字符串")


def _validate_batch(batch):
    if not isinstance(batch, str) or not batch:
        raise ValidationError("有效批次必须为非空字符串")


def _validate_range(lo, hi, n, what):
    for v in (lo, hi):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValidationError(f"{what}区间端点必须为整数")
    if not (0 <= lo <= hi <= n):
        raise ValidationError(f"{what}区间 [{lo},{hi}) 超出 [0,{n})")


class Dataset:
    """数据集入口。同一目录同一时刻只应打开一个 Dataset 实例。"""

    # ------------------------------------------------------------------ 打开/创建

    @classmethod
    def create(cls, path, stations, records=None):
        """创建数据集。stations 为按排序位置排列的互不相同的代码。

        records 可选，为首装记录 (record_id, batch, a, b, d, e)，
        走与订正相同的事务路径：折算值相同才合并证据，冲突报错而非覆盖。
        """
        stations = list(stations)
        _validate_stations(stations)
        if os.path.exists(path) and os.listdir(path):
            raise DatasetError(f"目标目录已存在且非空: {path}")
        os.makedirs(os.path.join(path, "versions"), exist_ok=True)
        with open(os.path.join(path, "stations.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(stations))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        v = Version(1, len(stations), {})
        v.save(os.path.join(path, "versions", _version_name(1)))
        _write_current(path, 1)
        ds = cls(path)
        if records:
            tx = ds.correction()
            for rec in records:
                tx.add(*rec)
            tx.commit()
        return ds

    @classmethod
    def open(cls, path):
        if not os.path.isdir(path):
            raise DatasetError(f"数据集目录不存在: {path}")
        return cls(path)

    def __init__(self, path):
        self._path = path
        stations_file = os.path.join(path, "stations.txt")
        if not os.path.isfile(stations_file):
            raise DatasetError(f"缺少测点文件: {stations_file}")
        with open(stations_file, encoding="utf-8") as f:
            self._stations = f.read().splitlines()
        _validate_stations(self._stations)
        self._pos = {c: i for i, c in enumerate(self._stations)}
        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pins = {}
        self._closed = False
        self._current = self._load_current()
        # 清理非当前版本的残留目录（如崩溃留下的半成品）
        self._gc_locked()

    def _load_current(self):
        current_file = os.path.join(self._path, "CURRENT")
        if not os.path.isfile(current_file):
            raise DatasetError("缺少 CURRENT 文件，数据集可能未完整创建")
        with open(current_file, encoding="utf-8") as f:
            name = f.read().strip()
        vdir = os.path.join(self._path, "versions", name)
        if not os.path.isdir(vdir):
            raise DatasetError(f"CURRENT 指向的版本目录不存在: {name}")
        return Version.load(vdir, len(self._stations))

    # ------------------------------------------------------------------ 属性

    @property
    def stations(self):
        return tuple(self._stations)

    @property
    def n_stations(self):
        return len(self._stations)

    @property
    def version_seq(self):
        return self._current.seq

    @property
    def fact_count(self):
        return len(self._current.I)

    # ------------------------------------------------------------------ 读

    def read(self):
        """开启读事务：钉住当前版本，返回 Snapshot（建议 with 使用）。"""
        with self._state_lock:
            self._ensure_open()
            v = self._current
            self._pins[v.seq] = self._pins.get(v.seq, 0) + 1
        return Snapshot(self, v)

    # ------------------------------------------------------------------ 写

    def correction(self):
        """开启一个订正事务（单写者；基于开启时的当前版本）。"""
        with self._state_lock:
            self._ensure_open()
            base = self._current
        return Correction(self, base)

    def _commit(self, tx):
        with self._write_lock:
            with self._state_lock:
                self._ensure_open()
                base = self._current
            if tx._base is not base:
                raise StaleTransactionError("订正基于过期版本，请重新开启订正事务")
            # 汇总受影响的测点对
            touched = set()
            for rid in tx._revokes:
                t = base.rec_index[rid]  # revoke 时已校验存在
                k = base.fact_of_entry(t)
                touched.add((base.I[k], base.J[k]))
            adds_by_pair = {}
            for rid, (batch, a, b, d, e) in tx._adds.items():
                pa = self._pos[a]
                pb = self._pos[b]
                i, j = (pa, pb) if pa <= pb else (pb, pa)
                adds_by_pair.setdefault((i, j), []).append((rid, batch, pa, pb, d, e))
                touched.add((i, j))
            new_facts = dict(base.facts)
            for pair in touched:
                old = base.facts.get(pair)
                evs = list(old[2]) if old is not None else []
                if tx._revokes:
                    evs = [ev for ev in evs if ev[0] not in tx._revokes]
                evs.extend(adds_by_pair.get(pair, ()))
                if not evs:
                    # 证据撤空：该对不再生成单元
                    new_facts.pop(pair, None)
                    continue
                vals = set()
                for _rid, _batch, pa, pb, rd, re_ in evs:
                    vals.add((rd if pa <= pb else -rd, re_))
                if len(vals) != 1:
                    ci, cj = self._stations[pair[0]], self._stations[pair[1]]
                    raise ConflictError(
                        f"测点对 ({ci!r},{cj!r}) 剩余证据折算值不一致: "
                        f"{sorted(vals)!r}，拒绝发布"
                    )
                d, e = vals.pop()
                evs.sort(key=lambda ev: ev[0])
                new_facts[pair] = (d, e, tuple(evs))
            new_version = Version(base.seq + 1, len(self._stations), new_facts)
            new_version.save(os.path.join(self._path, "versions", _version_name(new_version.seq)))
            self._publish(new_version)
            self._gc_locked()
            return new_version.seq

    def _publish(self, version):
        _write_current(self._path, version.seq)
        with self._state_lock:
            self._current = version

    # ------------------------------------------------------------------ GC / 关闭

    def gc(self):
        """清理非当前且未被读事务钉住的版本目录。"""
        with self._write_lock:
            self._gc_locked()

    def _gc_locked(self):
        with self._state_lock:
            keep = {self._current.seq}
            keep.update(seq for seq, cnt in self._pins.items() if cnt > 0)
        vroot = os.path.join(self._path, "versions")
        if not os.path.isdir(vroot):
            return
        for name in os.listdir(vroot):
            try:
                seq = int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                seq = None
            if seq not in keep:
                shutil.rmtree(os.path.join(vroot, name), ignore_errors=True)

    def close(self):
        with self._state_lock:
            self._closed = True

    def _ensure_open(self):
        if self._closed:
            raise DatasetError("数据集已关闭")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _write_current(path, seq):
    tmp = os.path.join(path, "CURRENT.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_version_name(seq))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(path, "CURRENT"))
    _fsync_dir(path)


class Correction:
    """订正事务：同时增、撤多条有身份的证据，提交时整体校验并原子发布。"""

    def __init__(self, ds, base):
        self._ds = ds
        self._base = base
        self._adds = {}  # record_id -> (batch, a, b, d, e)
        self._revokes = set()
        self._finished = False

    def _check_active(self):
        if self._finished:
            raise DatasetError("订正事务已结束")

    def add(self, record_id, batch, a, b, d, e):
        """新增一条有向证据：从测点 a 到测点 b，整数偏差 d，非负误差 e。"""
        self._check_active()
        _validate_record_id(record_id)
        _validate_batch(batch)
        if not isinstance(d, int) or isinstance(d, bool):
            raise ValidationError("偏差 d 必须为整数")
        if not (_INT64_MIN < d < 2**63):
            raise ValidationError("偏差 d 超出可安全取反的 int64 范围")
        if isinstance(e, bool) or not isinstance(e, (int, float)):
            raise ValidationError("误差 e 必须为非负数值")
        e = float(e)
        if not e >= 0.0:  # NaN 也会在此被拒绝
            raise ValidationError("误差 e 必须为非负")
        pos = self._ds._pos
        if a not in pos or b not in pos:
            raise ValidationError(f"未知测点代码: {a!r} 或 {b!r}")
        if a == b and d != 0:
            raise ValidationError("自对只允许 d=0")
        if (
            record_id in self._base.rec_index
            or record_id in self._adds
            or record_id in self._revokes
        ):
            raise DuplicateRecordError(f"记录身份已存在: {record_id!r}")
        # 注意：不在此处校验折算值一致性 —— 同事务允许先增后撤的临时不一致，
        # 只在提交时校验最终剩余支持是否相互一致。
        self._adds[record_id] = (batch, a, b, d, e)

    def revoke(self, record_id):
        """撤回一条已存在的记录（必须引用已提交版本中存在的记录）。"""
        self._check_active()
        _validate_record_id(record_id)
        if record_id in self._revokes:
            raise ValidationError(f"同一记录被重复撤回: {record_id!r}")
        if record_id not in self._base.rec_index:
            raise UnknownRecordError(f"撤回引用了不存在的记录: {record_id!r}")
        self._revokes.add(record_id)

    def commit(self):
        """校验并原子发布；失败则数据集保持原版本。"""
        self._check_active()
        self._finished = True
        return self._ds._commit(self)

    def rollback(self):
        self._finished = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._finished:
            self.rollback()


class Snapshot:
    """读事务：固定在某一版本上，与写者并行也不会读出新旧混搭。"""

    def __init__(self, ds, version):
        self._ds = ds
        self._v = version
        self._closed = False

    @property
    def version_seq(self):
        return self._v.seq

    def window(self, r0, r1, c0, c1):
        """按排序位置给两个半开区间 [r0,r1) x [c0,c1)，交付其中有向单元。

        正向命中：规范点 i∈[r0,r1), j∈[c0,c1)（含对角，计 self）；
        反向命中：规范点 i∈[c0,c1), j∈[r0,r1) 且 i<j，折算为 (j,i,-d,e)。
        结果按 (row, col) 排序；反向单元指向同一批证据并标记方向。
        """
        v = self._v
        n = v.n_stations
        _validate_range(r0, r1, n, "行")
        _validate_range(c0, c1, n, "列")
        stats = {"nodes_visited": 0, "facts_total": len(v.I)}
        fwd = []
        rev = []
        if r0 < r1 and c0 < c1:
            v.rect_query(r0, r1, c0, c1, fwd, stats)
            tmp = []
            v.rect_query(c0, c1, r0, r1, tmp, stats)
            rev = [k for k in tmp if v.I[k] < v.J[k]]  # 对角只在正向计一次
        stations = self._ds._stations
        cells = []
        for k in fwd:
            i = v.I[k]
            j = v.J[k]
            orient = SELF if i == j else FORWARD
            cells.append(
                Cell(i, j, v.D[k], v.E[k], orient, v.evidence_of(k), stations[i], stations[j])
            )
        for k in rev:
            i = v.I[k]
            j = v.J[k]
            cells.append(
                Cell(
                    j, i, -v.D[k], v.E[k], REVERSE, v.evidence_of(k), stations[j], stations[i]
                )
            )
        cells.sort(key=lambda c: (c.row, c.col))
        evidence_index = {}
        for c in cells:
            for ev in c.evidence:
                if ev.record_id not in evidence_index:
                    evidence_index[ev.record_id] = ev
        return WindowResult(r0, r1, c0, c1, v.seq, cells, evidence_index, stats)

    def pair(self, row, col):
        """按任一测点的行/列取同一对的结果；无证据返回 None。"""
        v = self._v
        n = v.n_stations
        for p in (row, col):
            if not isinstance(p, int) or isinstance(p, bool) or not (0 <= p < n):
                raise ValidationError(f"测点位置越界: {p!r}")
        i, j = (row, col) if row <= col else (col, row)
        k = v.pair_index.get((i, j))
        if k is None:
            return None
        stations = self._ds._stations
        ev = v.evidence_of(k)
        if row == col:
            return Cell(row, col, v.D[k], v.E[k], SELF, ev, stations[row], stations[col])
        if row < col:
            return Cell(row, col, v.D[k], v.E[k], FORWARD, ev, stations[row], stations[col])
        return Cell(row, col, -v.D[k], v.E[k], REVERSE, ev, stations[row], stations[col])

    def close(self):
        if self._closed:
            return
        self._closed = True
        ds = self._ds
        with ds._state_lock:
            seq = self._v.seq
            cnt = ds._pins.get(seq, 0)
            if cnt <= 1:
                ds._pins.pop(seq, None)
            else:
                ds._pins[seq] = cnt - 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
