"""追补（增量同步）接口的测试。

核对方法：测试侧独立维护"台账"（record_id -> (batch, src, dst, raw_d, raw_e)，
src/dst 为测点排序位置），每个版本留一份；追补结果与两端台账的直接比较
对齐，并把变更应用到起点台账上核对与目标台账一致。
"""

import math
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pairwise_dataset.dataset as dataset_mod  # noqa: E402
from pairwise_dataset import (  # noqa: E402
    CATCHUP_WINDOW,
    ConflictError,
    Dataset,
    DatasetError,
    ValidationError,
)


# ---------------------------------------------------------------------- 工具


def make_codes(n):
    return [f"P{i:07d}" for i in range(n)]


def ev_tuple(ev):
    """EvidenceRef -> 可比较元组；None 原样返回。"""
    if ev is None:
        return None
    return (ev.batch, ev.src, ev.dst, ev.raw_d, ev.raw_e)


def changes_as_tuples(changes):
    """追补变更 -> [(record_id, old_tuple, new_tuple)]，便于与台账直接比较。"""
    return [(ch.record_id, ev_tuple(ch.old), ev_tuple(ch.new)) for ch in changes]


def direct_diff(old_records, new_records):
    """小台账两端直接比较：按记录 ID 字符串序，每个有差异的 ID 一条。"""
    out = []
    for rid in sorted(set(old_records) | set(new_records)):
        old = old_records.get(rid)
        new = new_records.get(rid)
        if old != new:
            out.append((rid, old, new))
    return out


def apply_changes(records, changes):
    """把追补变更应用到台账（dict）上。"""
    for ch in changes:
        if ch.new is None:
            records.pop(ch.record_id, None)
        else:
            records[ch.record_id] = ev_tuple(ch.new)


def add_consistent(rng, tx, live, codes, n, rid):
    """增一条与 live 中现有证据折算值一致的随机记录（位置形式）。"""
    a, b = rng.randrange(n), rng.randrange(n)
    i, j = min(a, b), max(a, b)
    cur = None
    for r in live.values():
        pa, pb = r[1], r[2]  # (batch, a, b, d, e)
        if (min(pa, pb), max(pa, pb)) == (i, j):
            cur = (r[3] if pa <= pb else -r[3], r[4])
            break
    if cur is None:
        d = 0 if i == j else rng.randint(-100, 100)
        e = rng.choice([0.0, 0.5, 1.0])
    else:
        d, e = cur
    if i == j or rng.random() < 0.5:
        ra, rb, rd = i, j, d
    else:
        ra, rb, rd = j, i, -d
    batch = rng.choice(["ba", "bb", "bc"])
    tx.add(rid, batch, codes[ra], codes[rb], rd, e)
    live[rid] = (batch, ra, rb, rd, e)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="pwd_catchup_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def sub(self, name="ds"):
        return os.path.join(self.dir, name)


# ---------------------------------------------------------------------- 基本语义


class CatchupBasicTest(TempDirCase):
    def test_catchup_matches_direct_compare_and_apply(self):
        codes = make_codes(8)
        ds = Dataset.create(self.sub(), codes)
        live = {}
        ledgers = {ds.version_seq: dict(live)}  # v1：空

        def commit(tx_ops):
            tx = ds.correction()
            for op, args in tx_ops:
                if op == "add":
                    rid, batch, a, b, d, e = args
                    tx.add(rid, batch, codes[a], codes[b], d, e)
                    live[rid] = (batch, a, b, d, float(e))
                else:
                    tx.revoke(args)
                    live.pop(args)
            tx.commit()
            ledgers[ds.version_seq] = dict(live)

        # v2 首装；v3 撤 r2 增 r4；v4 撤 r1；v5 跨轮重用 r1（全新内容）
        commit([("add", ("r1", "b1", 0, 1, 5, 0.5)),
                ("add", ("r2", "b1", 2, 3, -7, 1.0)),
                ("add", ("r3", "b2", 4, 5, 9, 0.25))])
        commit([("revoke", "r2"), ("add", ("r4", "b3", 6, 7, 2, 0.5))])
        commit([("revoke", "r1")])
        commit([("add", ("r1", "b9", 1, 0, -8, 0.5))])
        target = ds.version_seq
        self.assertEqual(target, 5)

        with ds.read() as snap:
            for cursor in (2, 3, 4, 5):
                res = snap.catchup(cursor)
                self.assertEqual(res.from_seq, cursor)
                self.assertEqual(res.to_seq, target)
                # 与两端台账的直接比较对齐
                self.assertEqual(
                    changes_as_tuples(res.changes),
                    direct_diff(ledgers[cursor], ledgers[target]),
                )
                # 每个有差异的 ID 只出现一次，按记录 ID 字符串序
                rids = [c.record_id for c in res.changes]
                self.assertEqual(rids, sorted(set(rids)))
                # 起点台账应用变更后必须与目标台账一致
                applied = dict(ledgers[cursor])
                apply_changes(applied, res.changes)
                self.assertEqual(applied, ledgers[target])
            # 从快照自身追补 -> 空差异
            self.assertEqual(snap.catchup(target).changes, ())
        ds.close()

    def test_revoke_reuse_id_and_cancelled_rounds(self):
        """撤回后重用同一 ID：按原始证据判断，不看折算后的矩阵读数。"""
        codes = make_codes(6)
        ds = Dataset.create(self.sub(), codes)
        with ds.correction() as tx:  # v2
            tx.add("r1", "b1", codes[0], codes[1], 5, 0.5)
            tx.add("r2", "b1", codes[2], codes[3], 7, 1.0)
            tx.commit()
        with ds.read() as snap:
            self.assertEqual(snap.pair(0, 1).d, 5)
        with ds.correction() as tx:  # v3：撤 r1
            tx.revoke("r1")
            tx.commit()
        with ds.correction() as tx:  # v4：重用 r1，换批次换方向，折算值相同
            tx.add("r1", "b2", codes[1], codes[0], -5, 0.5)
            tx.commit()
        with ds.read() as snap:
            # 矩阵读数与 v2 完全相同（折算后都是 (0,1)=5）
            self.assertEqual(snap.pair(0, 1).d, 5)
            res = snap.catchup(2)
            # 但原始证据变了（批次、方向），必须交付
            self.assertEqual(
                changes_as_tuples(res.changes),
                [("r1", ("b1", 0, 1, 5, 0.5), ("b2", 1, 0, -5, 0.5))],
            )
        with ds.correction() as tx:  # v5：再撤
            tx.revoke("r1")
            tx.commit()
        with ds.correction() as tx:  # v6：恢复为与 v2 完全相同的内容
            tx.add("r1", "b1", codes[0], codes[1], 5, 0.5)
            tx.commit()
        with ds.correction() as tx:  # v7：出现
            tx.add("tmp", "b3", codes[4], codes[5], 1, 0.0)
            tx.commit()
        snap7 = ds.read()  # 钉住 tmp 存在的 v7
        with ds.correction() as tx:  # v8：又消失
            tx.revoke("tmp")
            tx.commit()
        with ds.read() as snap:
            # 最终恢复为起点同样内容的 ID 不交付
            self.assertEqual(snap.catchup(2).changes, ())
            # 中途出现又消失的 ID 不交付
            self.assertEqual(snap.catchup(6).changes, ())
            # 单轮跨度：tmp 被撤，交付（起点端存在、目标端为空）
            self.assertEqual(
                changes_as_tuples(snap.catchup(7).changes),
                [("tmp", ("b3", 4, 5, 1, 0.0), None)],
            )
        # 在 tmp 存在的旧快照上（新提交不影响其追补结果）：tmp 新增交付
        self.assertEqual(
            changes_as_tuples(snap7.catchup(6).changes),
            [("tmp", None, ("b3", 4, 5, 1, 0.0))],
        )
        snap7.close()
        ds.close()

    def test_initial_load_via_create_records(self):
        """create(records=...) 的首装同样进入追补：从初始版本 v1 起可追。"""
        codes = make_codes(4)
        ds = Dataset.create(
            self.sub(),
            codes,
            records=[
                ("r1", "b1", codes[0], codes[1], 5, 0.5),
                ("r2", "b1", codes[2], codes[3], -7, 1.0),
            ],
        )
        self.assertEqual(ds.version_seq, 2)
        with ds.read() as snap:
            res = snap.catchup(1)  # 初始版本 -> 首装版本
            self.assertEqual(
                changes_as_tuples(res.changes),
                [("r1", None, ("b1", 0, 1, 5, 0.5)),
                 ("r2", None, ("b1", 2, 3, -7, 1.0))],
            )
            self.assertEqual(res.to_seq, 2)
        ds.close()

    def test_error_value_compared_numerically(self):
        """误差按数值比较：-0.0 与 0.0 视为相同，不交付。"""
        codes = make_codes(4)
        ds = Dataset.create(self.sub(), codes)
        with ds.correction() as tx:  # v2：误差 -0.0（位模式与 0.0 不同）
            tx.add("r1", "b1", codes[0], codes[1], 5, -0.0)
            tx.commit()
        with ds.read() as snap:
            raw_e = snap.pair(0, 1).evidence[0].raw_e
            self.assertLess(math.copysign(1.0, raw_e), 0)  # 确认存的是 -0.0
        with ds.correction() as tx:  # v3
            tx.revoke("r1")
            tx.commit()
        with ds.correction() as tx:  # v4：同内容但误差 0.0
            tx.add("r1", "b1", codes[0], codes[1], 5, 0.0)
            tx.commit()
        with ds.read() as snap:
            # 数值相等 -> 不交付
            self.assertEqual(snap.catchup(2).changes, ())
        with ds.correction() as tx:  # v5
            tx.revoke("r1")
            tx.commit()
        with ds.correction() as tx:  # v6：误差真变了 -> 交付
            tx.add("r1", "b1", codes[0], codes[1], 5, 1.0)
            tx.commit()
        with ds.read() as snap:
            self.assertEqual(
                changes_as_tuples(snap.catchup(4).changes),
                [("r1", ("b1", 0, 1, 5, 0.0), ("b1", 0, 1, 5, 1.0))],
            )
        ds.close()

    def test_old_snapshot_replays_around_later_commits(self):
        """旧快照追补：同一 ID 在跨度之后又被改动，目标端内容须正确倒放。"""
        codes = make_codes(4)
        ds = Dataset.create(self.sub(), codes)
        with ds.correction() as tx:  # v2：r1 = A
            tx.add("r1", "bA", codes[0], codes[1], 1, 0.0)
            tx.commit()
        with ds.correction() as tx:  # v3
            tx.revoke("r1")
            tx.commit()
        with ds.correction() as tx:  # v4：r1 = B
            tx.add("r1", "bB", codes[0], codes[1], 2, 0.0)
            tx.commit()
        snap4 = ds.read()  # 钉住 v4
        with ds.correction() as tx:  # v5
            tx.revoke("r1")
            tx.commit()
        with ds.correction() as tx:  # v6：r1 = C（当前版本）
            tx.add("r1", "bC", codes[0], codes[1], 3, 0.0)
            tx.commit()
        self.assertEqual(ds.version_seq, 6)
        # 旧快照上的追补不受 v5/v6 影响：目标端须倒放回 v4 的内容 B
        res = snap4.catchup(2)
        self.assertEqual(res.to_seq, 4)
        self.assertEqual(
            changes_as_tuples(res.changes),
            [("r1", ("bA", 0, 1, 1, 0.0), ("bB", 0, 1, 2, 0.0))],
        )
        # 当前快照上同一游标：看到 C
        with ds.read() as snap:
            self.assertEqual(
                changes_as_tuples(snap.catchup(2).changes),
                [("r1", ("bA", 0, 1, 1, 0.0), ("bC", 0, 1, 3, 0.0))],
            )
        snap4.close()
        ds.close()


# ---------------------------------------------------------------------- 保留窗口与游标


class CatchupWindowTest(TempDirCase):
    def _build(self, rounds_after_snapshot=0):
        """v2 首装 r0；之后每轮增一条 rk，返回 (ds, 各版本号)。"""
        codes = make_codes(40)
        ds = Dataset.create(self.sub(), codes)
        with ds.correction() as tx:
            tx.add("r0", "b0", codes[0], codes[1], 0, 0.0)
            tx.commit()
        for k in range(1, 17):
            with ds.correction() as tx:
                tx.add(f"r{k}", f"b{k}", codes[2 * k], codes[2 * k + 1], k, 0.0)
                tx.commit()
        # 当前 v18
        return ds, codes

    def test_retention_window_and_stale_cursor(self):
        ds, codes = self._build()
        self.assertEqual(ds.version_seq, 2 + 16)
        snap_old = ds.read()  # 钉住 v18
        res_a = snap_old.catchup(2)  # 恰好 16 轮
        self.assertEqual(res_a.to_seq, 18)
        # 按记录 ID 的字符串顺序交付（"r10" < "r2"）
        expected = [
            (f"r{k}", None, (f"b{k}", 2 * k, 2 * k + 1, k, 0.0)) for k in range(1, 17)
        ]
        expected.sort(key=lambda t: t[0])
        self.assertEqual(changes_as_tuples(res_a.changes), expected)
        # 旧快照开着时再提交两轮：用它追补的范围和结果不因新提交而变
        for k in (17, 18):
            with ds.correction() as tx:
                tx.add(f"r{k}", f"b{k}", codes[2 * k], codes[2 * k + 1], k, 0.0)
                tx.commit()
        self.assertEqual(ds.version_seq, 20)
        res_b = snap_old.catchup(2)  # 同一游标，在旧快照上仍恰好有效
        self.assertEqual(changes_as_tuples(res_b.changes), changes_as_tuples(res_a.changes))
        self.assertEqual(res_b.to_seq, 18)
        # 同一游标对更新的快照已过期
        snap_new = ds.read()  # v20
        with self.assertRaises(ValidationError):
            snap_new.catchup(2)  # 18 轮 > 16
        res_c = snap_new.catchup(4)  # 恰好 16 轮，有效
        self.assertEqual(len(res_c.changes), 16)
        self.assertEqual(res_c.to_seq, 20)
        # 起点之前 / 超过保留范围 / 比目标更新 / 非法游标
        with self.assertRaises(ValidationError):
            snap_old.catchup(0)   # 早于追补起点（anchor=1）
        with self.assertRaises(ValidationError):
            snap_old.catchup(1)   # 17 轮，超过保留范围
        with self.assertRaises(ValidationError):
            snap_old.catchup(19)  # 比目标快照更新
        for bad in ("3", 2.5, True, None):
            with self.assertRaises(ValidationError):
                snap_old.catchup(bad)
        # 从快照自身追补 -> 空差异
        self.assertEqual(snap_old.catchup(18).changes, ())
        self.assertEqual(snap_new.catchup(20).changes, ())
        # 关闭的快照报错
        snap_old.close()
        with self.assertRaises(DatasetError):
            snap_old.catchup(2)
        snap_new.close()
        ds.close()

    def test_empty_commit_consumes_round(self):
        """成功但未改动证据的订正也占一轮。"""
        codes = make_codes(4)
        ds = Dataset.create(self.sub(), codes)
        with ds.correction() as tx:  # v2
            tx.add("r0", "b0", codes[0], codes[1], 1, 0.0)
            tx.commit()
        for _ in range(15):  # v3..v17：15 轮空订正
            with ds.correction() as tx:
                tx.commit()
        with ds.correction() as tx:  # v18：一轮真实订正
            tx.add("r1", "b1", codes[2], codes[3], 2, 0.0)
            tx.commit()
        self.assertEqual(ds.version_seq, 18)
        with ds.read() as snap:
            res = snap.catchup(2)  # 恰好 16 轮（15 空 + 1 实）
            self.assertEqual(
                changes_as_tuples(res.changes),
                [("r1", None, ("b1", 2, 3, 2, 0.0))],
            )
        with ds.correction() as tx:  # v19：再一轮空订正
            tx.commit()
        with ds.read() as snap:
            with self.assertRaises(ValidationError):
                snap.catchup(2)  # 17 轮，超窗
            self.assertEqual(len(snap.catchup(3).changes), 1)  # 16 轮，有效
        ds.close()


# ---------------------------------------------------------------------- 崩溃一致性


class CatchupCrashTest(TempDirCase):
    def test_unpublished_round_disappears_after_reopen(self):
        codes = make_codes(4)
        ds = Dataset.create(self.sub(), codes)
        with ds.correction() as tx:  # v2
            tx.add("r1", "b1", codes[0], codes[1], 5, 0.5)
            tx.commit()
        with ds.correction() as tx:  # v3
            tx.add("r2", "b1", codes[2], codes[3], 7, 1.0)
            tx.commit()
        # 被拒绝的订正：折算值冲突，不发布
        tx = ds.correction()
        tx.add("r3", "b9", codes[0], codes[1], 6, 0.5)
        with self.assertRaises(ConflictError):
            tx.commit()
        self.assertEqual(ds.version_seq, 3)
        # 在切换 CURRENT 前中断的订正（模拟崩溃）
        orig_write_current = dataset_mod._write_current

        def boom(path, seq):
            raise RuntimeError("模拟切换 CURRENT 前崩溃")

        dataset_mod._write_current = boom
        try:
            tx = ds.correction()
            tx.add("r4", "b2", codes[1], codes[2], 4, 0.5)
            with self.assertRaises(RuntimeError):
                tx.commit()
        finally:
            dataset_mod._write_current = orig_write_current
        ds.close()

        # 重开：成功的差异仍在，未发布的差异不在
        ds = Dataset.open(self.sub())
        self.assertEqual(ds.version_seq, 3)
        cdir = os.path.join(self.sub(), "changelog")
        self.assertEqual(
            sorted(os.listdir(cdir)),
            ["c_00000002.json", "c_00000003.json", "meta.json"],
        )
        with ds.read() as snap:
            self.assertEqual(
                changes_as_tuples(snap.catchup(1).changes),
                [("r1", None, ("b1", 0, 1, 5, 0.5)),
                 ("r2", None, ("b1", 2, 3, 7, 1.0))],
            )
            self.assertEqual(
                changes_as_tuples(snap.catchup(2).changes),
                [("r2", None, ("b1", 2, 3, 7, 1.0))],
            )
            self.assertIsNone(snap.pair(1, 2))  # r4 未发布
        # 已发布的订正重开后仍须可追补；中断后同一版本号可被后续订正复用
        with ds.correction() as tx:  # v4
            tx.add("r5", "b3", codes[0], codes[2], 3, 0.25)
            tx.commit()
        ds.close()
        ds = Dataset.open(self.sub())
        with ds.read() as snap:
            self.assertEqual(
                changes_as_tuples(snap.catchup(2).changes),
                [("r2", None, ("b1", 2, 3, 7, 1.0)),
                 ("r5", None, ("b3", 0, 2, 3, 0.25))],
            )
            self.assertEqual(
                changes_as_tuples(snap.catchup(3).changes),
                [("r5", None, ("b3", 0, 2, 3, 0.25))],
            )
        ds.close()


# ---------------------------------------------------------------------- 起点（anchor）


class CatchupAnchorTest(TempDirCase):
    def test_anchor_established_on_first_open_and_stable(self):
        codes = make_codes(4)
        ds = Dataset.create(self.sub(), codes)
        with ds.correction() as tx:  # v2
            tx.add("a1", "b1", codes[0], codes[1], 1, 0.0)
            tx.commit()
        with ds.correction() as tx:  # v3
            tx.add("a2", "b1", codes[2], codes[3], 2, 0.0)
            tx.commit()
        ds.close()
        # 模拟"已有数据目录第一次用新版打开"：没有追补状态目录
        shutil.rmtree(os.path.join(self.sub(), "changelog"))
        ds = Dataset.open(self.sub())
        self.assertEqual(ds.version_seq, 3)
        with ds.read() as snap:
            # 起点 = 当时的当前版本；自身追补为空
            self.assertEqual(snap.catchup(3).changes, ())
            # 起点之前的游标报错，不给出部分差异
            for bad in (1, 2):
                with self.assertRaises(ValidationError):
                    snap.catchup(bad)
        with ds.correction() as tx:  # v4
            tx.add("a3", "b2", codes[1], codes[2], 3, 0.0)
            tx.commit()
        with ds.correction() as tx:  # v5
            tx.revoke("a1")
            tx.commit()
        with ds.read() as snap:
            res = snap.catchup(3)
            self.assertEqual(
                changes_as_tuples(res.changes),
                [("a1", ("b1", 0, 1, 1, 0.0), None),
                 ("a3", None, ("b2", 1, 2, 3, 0.0))],
            )
        ds.close()
        # 以后重开不能重置这个起点
        ds = Dataset.open(self.sub())
        with ds.read() as snap:
            for bad in (1, 2):
                with self.assertRaises(ValidationError):
                    snap.catchup(bad)
            res = snap.catchup(3)  # 起点仍然有效
            self.assertEqual(len(res.changes), 2)
        ds.close()


# ---------------------------------------------------------------------- 回收


class CatchupGcTest(TempDirCase):
    def test_changelog_gc_follows_pins(self):
        codes = make_codes(60)
        ds = Dataset.create(self.sub(), codes)
        with ds.correction() as tx:  # v2
            tx.add("r0", "b0", codes[0], codes[1], 0, 0.0)
            tx.commit()
        for k in range(1, 9):  # v3..v10
            with ds.correction() as tx:
                tx.add(f"r{k}", "b", codes[2 * k], codes[2 * k + 1], k, 0.0)
                tx.commit()
        snap_old = ds.read()  # 钉住 v10
        for k in range(9, 21):  # v11..v22
            with ds.correction() as tx:
                tx.add(f"r{k}", "b", codes[2 * k], codes[2 * k + 1], k, 0.0)
                tx.commit()
        self.assertEqual(ds.version_seq, 22)
        cdir = os.path.join(self.sub(), "changelog")
        rounds = sorted(f for f in os.listdir(cdir) if f.startswith("c_"))
        # 旧快照钉住 v10：v10 往前 16 轮起全部保留（实际覆盖所有已写轮次）
        self.assertEqual(len(rounds), 21)  # c_00000002 .. c_00000022
        # 旧快照仍可用完整窗口追补
        self.assertEqual(len(snap_old.catchup(2).changes), 8)
        snap_old.close()
        ds.gc()
        rounds = sorted(f for f in os.listdir(cdir) if f.startswith("c_"))
        # 不再被需要的历史被回收：只剩当前版本往前 16 轮
        self.assertEqual(rounds, [f"c_{s:08d}.json" for s in range(6, 23)])
        # 完整的历史矩阵目录不被延期保留：只剩当前版本
        vroot = os.path.join(self.sub(), "versions")
        self.assertEqual(os.listdir(vroot), ["v_00000022"])
        with ds.read() as snap:
            self.assertEqual(len(snap.catchup(6).changes), 16)  # 恰好 16 轮
            with self.assertRaises(ValidationError):
                snap.catchup(5)  # 17 轮，超窗
        ds.close()


# ---------------------------------------------------------------------- 不做整仓扫描


class CatchupScaleTest(TempDirCase):
    def test_short_catchup_does_not_scan_whole_dataset(self):
        """运行时生成大量未变证据：短跨度追补只处理跨度内变化的记录。"""
        n = 60_000
        codes = make_codes(n)
        ds = Dataset.create(self.sub(), codes)
        n_bulk = 20_000
        with ds.correction() as tx:  # v2：两万个互不相交的对
            for t in range(n_bulk):
                tx.add(f"b{t}", "bulk", codes[3 * t], codes[3 * t + 1],
                       t % 1000 - 500, 0.5)
            tx.commit()
        with ds.correction() as tx:  # v3：只动 3 条记录
            tx.revoke("b0")
            tx.revoke("b1")
            tx.add("x1", "fix", codes[5], codes[6], 3, 0.5)
            tx.commit()
        with ds.read() as snap:
            res = snap.catchup(2)
            # 只为跨度内发生过变化的记录付出工作，不遍历未变的证据
            self.assertEqual(res.stats["records_touched"], 3)
            self.assertEqual(res.stats["rounds"], 1)
            self.assertEqual(
                changes_as_tuples(res.changes),
                [("b0", ("bulk", 0, 1, -500, 0.5), None),
                 ("b1", ("bulk", 3, 4, -499, 0.5), None),
                 ("x1", None, ("fix", 5, 6, 3, 0.5))],
            )
            # 自身追补零变化
            self.assertEqual(snap.catchup(3).stats["records_touched"], 0)
        ds.close()


# ---------------------------------------------------------------------- 随机交叉核对


class CatchupRandomizedTest(TempDirCase):
    def test_randomized_catchup_crosscheck(self):
        rng = random.Random(43)
        n = 30
        codes = make_codes(n)
        ds = Dataset.create(self.sub(), codes)
        live = {}  # rid -> (batch, a, b, d, e)
        retired = []  # 之前事务撤掉、可跨轮重用的 rid
        ledgers = {ds.version_seq: {}}
        rid_gen = [0]
        for _round in range(24):
            tx = ds.correction()
            revoked_now = []
            for rid in rng.sample(list(live), min(len(live), rng.randint(0, 3))):
                tx.revoke(rid)
                live.pop(rid)
                revoked_now.append(rid)
            for _ in range(rng.randint(1, 4)):
                if retired and rng.random() < 0.3:
                    rid = retired.pop(rng.randrange(len(retired)))
                else:
                    rid = f"n{rid_gen[0]}"
                    rid_gen[0] += 1
                add_consistent(rng, tx, live, codes, n, rid)
            tx.commit()
            retired.extend(revoked_now)
            ledgers[ds.version_seq] = dict(live)
            target = ds.version_seq
            lo = max(1, target - CATCHUP_WINDOW)
            for _ in range(2):
                cursor = rng.randint(lo, target)
                with ds.read() as snap:
                    res = snap.catchup(cursor)
                self.assertEqual(res.from_seq, cursor)
                self.assertEqual(res.to_seq, target)
                self.assertEqual(
                    changes_as_tuples(res.changes),
                    direct_diff(ledgers[cursor], ledgers[target]),
                )
                applied = dict(ledgers[cursor])
                apply_changes(applied, res.changes)
                self.assertEqual(applied, ledgers[target])
                rids = [c.record_id for c in res.changes]
                self.assertEqual(rids, sorted(set(rids)))
        ds.close()


if __name__ == "__main__":
    unittest.main()
