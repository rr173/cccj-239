"""换版（测点重排，Dataset.reorder）的测试。

核对方法：
- 小测点表 + 正/反向证据：窗口、单对、补采工单在换版前后分别与"按代码
  独立展开的全矩阵"和"独立枚举的缺口"比对；
- 台账：测试侧按代码保存原始证据（rid -> (batch, code_a, code_b, d, e)），
  用两端版本各自的测点顺序独立展开成台账（位置按端点版本解释），与追补
  结果逐条对齐，并把变更应用到起点台账核对与目标台账一致；
- 已开对象（旧快照、它的工单、未提交订正）与发布前中断的行为。
"""

import os
import random
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pairwise_dataset.dataset as dataset_mod  # noqa: E402
from pairwise_dataset import (  # noqa: E402
    CATCHUP_WINDOW,
    Dataset,
    StaleTransactionError,
    ValidationError,
)
from tests.test_catchup import (  # noqa: E402
    apply_changes,
    changes_as_tuples,
    direct_diff,
    ev_tuple,
)
from tests.test_pairwise_dataset import (  # noqa: E402
    TempDirCase,
    cells_of,
    expand,
    expected_window,
    make_codes,
)
from tests.test_workorder import all_pages, brute_gaps  # noqa: E402


# ---------------------------------------------------------------------- 工具


def pos_recs(recs, order):
    """把按代码的记录 (rid, batch, code_a, code_b, d, e) 折算成某版顺序下的
    位置形式，供 expand / brute_gaps 独立展开。"""
    pos = {c: i for i, c in enumerate(order)}
    return [(rid, bat, pos[a], pos[b], d, e) for rid, bat, a, b, d, e in recs]


def ledger_of_records(recs, order):
    """按代码保存的原始证据 + 该版顺序 -> 台账（位置按该版坐标解释）。"""
    pos = {c: i for i, c in enumerate(order)}
    return {
        rid: (bat, pos[a], pos[b], d, float(e)) for rid, bat, a, b, d, e in recs
    }


# ---------------------------------------------------------------------- 窗口 / 单对 / 补采


class ReorderWindowTest(TempDirCase):
    """有正反向证据的小测点表：核对换版前后的窗口、单对与补采。"""

    CODES = ["A", "B", "C", "D", "E"]
    RECS = [  # 按代码的原始证据
        ("r1", "b1", "A", "B", 5, 0.5),
        ("r2", "b1", "B", "A", -5, 0.5),   # 反向证据，同对合并
        ("r3", "b2", "D", "A", -7, 1.0),   # 反向记录
        ("r4", "b2", "C", "E", 0, 2.0),    # 显式零（不是缺测）
        ("r5", "b3", "D", "D", 0, 0.0),    # 自对
        ("r6", "b4", "E", "A", 9, 0.25),
    ]
    NEW_ORDER = ["D", "A", "E", "B", "C"]

    def _dataset(self):
        return Dataset.create(self.sub(), self.CODES, records=list(self.RECS))

    def test_window_pair_backfill_before_and_after_reorder(self):
        ds = self._dataset()
        n = len(self.CODES)
        reqs = [(0, 2, 2, 4, 2), (1, 4, 0, 5, 1)]
        seq0 = ds.version_seq

        snap_old = ds.read()
        # 换版前：窗口与独立展开的全矩阵一致（创建顺序坐标）
        full_old = expand(pos_recs(self.RECS, self.CODES))
        self.assertEqual(
            cells_of(snap_old.window(0, n, 0, n)), expected_window(full_old, 0, n, 0, n)
        )
        self.assertEqual(snap_old.stations, tuple(self.CODES))
        self.assertEqual(ds.stations, tuple(self.CODES))
        # 换版前的补采与独立枚举一致
        order_old = snap_old.backfill(reqs)
        gaps_old = brute_gaps(pos_recs(self.RECS, self.CODES), reqs, n)
        self.assertGreater(len(gaps_old), 0)
        self.assertEqual(all_pages(order_old, 3), gaps_old)

        # 换版前开启的未提交订正
        tx_pending = ds.correction()
        tx_pending.add("r7", "b5", "B", "D", 4, 1.0)

        # 换版：发布新坐标版本
        seq1 = ds.reorder(self.NEW_ORDER)
        self.assertEqual(seq1, seq0 + 1)
        self.assertEqual(ds.version_seq, seq1)
        self.assertEqual(ds.stations, tuple(self.NEW_ORDER))

        # 换版后，之前开启的未提交订正按过期拒绝
        with self.assertRaises(StaleTransactionError):
            tx_pending.commit()

        # 旧快照与它的工单仍用旧坐标（页边界不移动）
        self.assertEqual(snap_old.version_seq, seq0)
        self.assertEqual(snap_old.stations, tuple(self.CODES))
        self.assertEqual(
            cells_of(snap_old.window(0, n, 0, n)), expected_window(full_old, 0, n, 0, n)
        )
        self.assertEqual(snap_old.pair(0, 1).d, 5)
        self.assertEqual(all_pages(order_old, 3), gaps_old)
        self.assertEqual(order_old.total_gaps, len(gaps_old))

        # 新快照：该版测点顺序可读；矩阵正向按新版位置解释
        full_new = expand(pos_recs(self.RECS, self.NEW_ORDER))
        with ds.read() as snap:
            self.assertEqual(snap.version_seq, seq1)
            self.assertEqual(snap.stations, tuple(self.NEW_ORDER))
            self.assertEqual(
                cells_of(snap.window(0, n, 0, n)), expected_window(full_new, 0, n, 0, n)
            )
            # 按代码读有向值与换版前一致（位置由该版顺序查出）
            pos = {c: i for i, c in enumerate(self.NEW_ORDER)}
            self.assertEqual(snap.pair(pos["A"], pos["B"]).d, 5)    # A->B 仍 5
            self.assertEqual(snap.pair(pos["B"], pos["A"]).d, -5)
            self.assertEqual(snap.pair(pos["A"], pos["D"]).d, 7)    # A->D 仍 7
            self.assertEqual(snap.pair(pos["D"], pos["A"]).d, -7)
            self.assertEqual(snap.pair(pos["C"], pos["E"]).d, 0)    # 显式零仍在
            self.assertEqual(snap.pair(pos["D"], pos["D"]).d, 0)    # 自对仍在
            self.assertIsNone(snap.pair(pos["A"], pos["C"]))        # 缺测仍缺测
            # 矩阵正向随新版位置翻转：{D,A} 在新版 D=0 < A=1，正向为 D->A
            cell = snap.pair(0, 1)
            self.assertEqual((cell.row_code, cell.col_code), ("D", "A"))
            self.assertEqual(cell.d, -7)
            self.assertEqual(cell.orientation, "forward")
            # 换版后的补采用该版坐标，与独立枚举一致
            order_new = snap.backfill(reqs)
            self.assertEqual(order_new.version_seq, seq1)
            self.assertEqual(
                all_pages(order_new, 3),
                brute_gaps(pos_recs(self.RECS, self.NEW_ORDER), reqs, n),
            )

        # 换版后的新订正仍按代码提交
        with ds.correction() as tx:
            tx.add("r7", "b5", "B", "D", 4, 1.0)
            tx.commit()
        live = list(self.RECS) + [("r7", "b5", "B", "D", 4, 1.0)]
        full_after = expand(pos_recs(live, self.NEW_ORDER))
        with ds.read() as snap:
            self.assertEqual(
                cells_of(snap.window(0, n, 0, n)),
                expected_window(full_after, 0, n, 0, n),
            )
            pos = {c: i for i, c in enumerate(self.NEW_ORDER)}
            self.assertEqual(snap.pair(pos["B"], pos["D"]).d, 4)
            # 新记录的证据位置按新版坐标交付
            res = snap.catchup(seq1)
            got = {ch.record_id: ev_tuple(ch.new) for ch in res.changes}
            self.assertEqual(got, {"r7": ("b5", pos["B"], pos["D"], 4, 1.0)})
        snap_old.close()
        ds.close()


# ---------------------------------------------------------------------- 校验与相同顺序


class ReorderValidationTest(TempDirCase):
    def test_invalid_orders_rejected_and_version_untouched(self):
        codes = ["A", "B", "C", "D"]
        ds = Dataset.create(
            self.sub(), codes, records=[("r1", "b1", "A", "B", 5, 0.5)]
        )
        seq0 = ds.version_seq
        for bad in (
            ["A", "B", "C"],                # 缺测点
            ["A", "B", "C", "D", "E"],      # 多出未知代码
            ["A", "B", "C", "C"],           # 重复（且缺 D）
            ["A", "B", "C", "X"],           # 未知代码
            ["A", "B", "C", ""],            # 空代码
            ["A", "B", "C", 1],             # 非字符串
            ["A", "B", "C", None],          # 非字符串
            [],                             # 空顺序
            "ABCD",                         # 字符串不是序列
            None,                           # 不可迭代
        ):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                ds.reorder(bad)
            self.assertEqual(ds.version_seq, seq0)
            self.assertEqual(ds.stations, tuple(codes))
        with ds.read() as snap:
            self.assertEqual(snap.pair(0, 1).d, 5)  # 坐标与证据都未变
        ds.close()

    def test_same_order_returns_current_version_without_round(self):
        codes = ["A", "B", "C", "D"]
        ds = Dataset.create(
            self.sub(), codes, records=[("r1", "b1", "A", "B", 5, 0.5)]
        )
        seq0 = ds.version_seq
        vroot = os.path.join(self.sub(), "versions")
        cdir = os.path.join(self.sub(), "changelog")
        # 相同顺序：直接返回当前版，不产生新版本、不占追补轮次
        self.assertEqual(ds.reorder(list(codes)), seq0)
        self.assertEqual(ds.reorder(tuple(codes)), seq0)  # 任意可迭代序列
        self.assertEqual(ds.version_seq, seq0)
        self.assertEqual(os.listdir(vroot), [f"v_{seq0:08d}"])
        self.assertEqual(
            sorted(f for f in os.listdir(cdir) if f.startswith("c_")),
            [f"c_{s:08d}.json" for s in range(2, seq0 + 1)],
        )
        # 换到别的顺序再换回创建顺序：回到创建顺序的版本不带 stations.txt
        seq1 = ds.reorder(["D", "C", "B", "A"])
        self.assertEqual(seq1, seq0 + 1)
        self.assertTrue(
            os.path.isfile(os.path.join(vroot, f"v_{seq1:08d}", "stations.txt"))
        )
        with ds.read() as snap:
            self.assertEqual(snap.stations, ("D", "C", "B", "A"))
        seq2 = ds.reorder(list(codes))
        self.assertEqual(seq2, seq1 + 1)
        self.assertEqual(ds.stations, tuple(codes))
        self.assertFalse(
            os.path.exists(os.path.join(vroot, f"v_{seq2:08d}", "stations.txt"))
        )
        with ds.read() as snap:
            self.assertEqual(snap.stations, tuple(codes))
            self.assertEqual(snap.pair(0, 1).d, 5)
        ds.close()

    def test_existing_directory_without_reorder_works_directly(self):
        """已有目录（版本目录没有 stations.txt 的旧格式）可直接打开并换版。"""
        codes = ["A", "B", "C"]
        ds = Dataset.create(
            self.sub(), codes, records=[("r1", "b1", "A", "C", 4, 0.5)]
        )
        ds.close()
        vroot = os.path.join(self.sub(), "versions")
        # 预检：未换版的版本目录不写 stations.txt，与旧格式一致
        for name in os.listdir(vroot):
            self.assertFalse(
                os.path.exists(os.path.join(vroot, name, "stations.txt"))
            )
        ds = Dataset.open(self.sub())
        with ds.read() as snap:
            self.assertEqual(snap.stations, tuple(codes))
            self.assertEqual(snap.pair(0, 2).d, 4)
        ds.reorder(["C", "A", "B"])
        with ds.read() as snap:
            self.assertEqual(snap.stations, ("C", "A", "B"))
            # 按代码读有向值不变：A->C 仍是 4；新坐标 C=0, A=1
            self.assertEqual(snap.pair(1, 0).d, 4)
            self.assertEqual(snap.pair(0, 1).d, -4)
        ds.close()
        # 重开后仍是新坐标
        ds = Dataset.open(self.sub())
        with ds.read() as snap:
            self.assertEqual(snap.stations, ("C", "A", "B"))
            self.assertEqual(snap.pair(1, 0).d, 4)
        ds.close()


# ---------------------------------------------------------------------- 混合追补


class MixedHistoryCatchupTest(TempDirCase):
    """换版与普通订正混合：台账按代码保留证据、用两端顺序独立构造。"""

    CODES = ["A", "B", "C", "D", "E", "F"]
    PI1 = ["A", "F", "C", "D", "B", "E"]   # A/C/D 不动，B/E/F 移动
    PI2 = ["E", "F", "D", "C", "B", "A"]

    def test_mixed_reorders_and_corrections(self):
        codes = self.CODES
        ds = Dataset.create(self.sub(), codes)
        live = {}  # rid -> (batch, code_a, code_b, d, e)，按代码保存
        orders = {ds.version_seq: list(codes)}      # 每个版本的测点顺序
        snaps = {ds.version_seq: ds.read()}         # 钉住每个版本
        ledgers = {ds.version_seq: {}}              # 每个版本的独立台账

        def ledger_of(order):
            pos = {c: i for i, c in enumerate(order)}
            return {
                rid: (bat, pos[a], pos[b], d, float(e))
                for rid, (bat, a, b, d, e) in live.items()
            }

        def commit(ops):
            tx = ds.correction()
            for op in ops:
                if op[0] == "add":
                    _, rid, bat, a, b, d, e = op
                    tx.add(rid, bat, a, b, d, e)
                    live[rid] = (bat, a, b, d, float(e))
                else:
                    tx.revoke(op[1])
                    live.pop(op[1])
            seq = tx.commit()
            orders[seq] = list(orders[seq - 1])
            ledgers[seq] = ledger_of(orders[seq])
            snaps[seq] = ds.read()

        def reorder(order):
            seq = ds.reorder(order)
            orders[seq] = list(order)
            ledgers[seq] = ledger_of(order)
            snaps[seq] = ds.read()

        commit([("add", "r1", "b1", "A", "B", 5, 0.5),
                ("add", "r2", "b1", "B", "A", -5, 0.5),
                ("add", "r3", "b2", "C", "D", 7, 1.0),
                ("add", "r4", "b3", "E", "E", 0, 0.0),
                ("add", "r5", "b2", "F", "C", -9, 0.25)])            # v2
        commit([("add", "r6", "b4", "D", "A", 3, 0.5),
                ("revoke", "r3"),
                ("add", "r7", "b5", "C", "D", 7, 1.0)])              # v3
        reorder(self.PI1)                                             # v4 纯换版
        commit([("add", "r8", "b6", "F", "B", 4, 2.0),
                ("revoke", "r6")])                                    # v5
        reorder(self.PI2)                                             # v6
        commit([("revoke", "r5"),
                ("add", "r9", "b7", "C", "F", 9, 0.25)])             # v7
        reorder(list(codes))                                          # v8 换回创建顺序
        commit([])                                                    # v9 空订正也占一轮
        reorder(self.PI1)                                             # v10
        reorder(self.PI2)                                             # v11
        reorder(self.PI1)                                             # v12 回到 v10 的顺序
        target = ds.version_seq
        self.assertEqual(target, 12)

        # 预检：换版确实只改坐标不改记录集合（测试不是空转的）
        self.assertEqual(ledgers[3].keys(), ledgers[4].keys())
        self.assertNotEqual(ledgers[3], ledgers[4])
        # v10 与 v12 顺序相同、证据未变 -> 两端台账一致
        self.assertEqual(ledgers[10], ledgers[12])

        # 纯换版一轮：只换位置的记录按 ID 交付两端内容，分别按端点版本解释
        res = snaps[4].catchup(3)
        got = {ch.record_id: (ev_tuple(ch.old), ev_tuple(ch.new)) for ch in res.changes}
        # PI1 中 A/C/D 位置不变：r6(D->A)、r7(C->D) 不交付；B/E/F 移动
        self.assertEqual(set(got), {"r1", "r2", "r4", "r5"})
        self.assertEqual(got["r1"], (("b1", 0, 1, 5, 0.5), ("b1", 0, 4, 5, 0.5)))
        self.assertEqual(got["r2"], (("b1", 1, 0, -5, 0.5), ("b1", 4, 0, -5, 0.5)))
        self.assertEqual(got["r4"], (("b3", 4, 4, 0, 0.0), ("b3", 5, 5, 0, 0.0)))
        self.assertEqual(got["r5"], (("b2", 5, 2, -9, 0.25), ("b2", 1, 2, -9, 0.25)))
        self.assertEqual(res.stats["records_touched"], 4)  # 只处理移动了的记录

        # 路线改回原顺序且证据没变（v10 -> v12 经 v11 兜一圈）：净变更为空
        self.assertEqual(snaps[12].catchup(10).changes, ())
        # 但单看其中一段换版仍交付位置变化
        self.assertNotEqual(snaps[12].catchup(11).changes, ())

        # 全量交叉核对：保留窗口内任取 (cursor, target)，与两端独立台账的
        # 直接比较对齐；起点台账应用差异后与目标台账一致
        for cursor in range(max(1, target - CATCHUP_WINDOW), target + 1):
            for tgt in range(cursor, target + 1):
                res = snaps[tgt].catchup(cursor)
                self.assertEqual(res.from_seq, cursor)
                self.assertEqual(res.to_seq, tgt)
                self.assertEqual(
                    changes_as_tuples(res.changes),
                    direct_diff(ledgers[cursor], ledgers[tgt]),
                    msg=f"{cursor}->{tgt}",
                )
                applied = dict(ledgers[cursor])
                apply_changes(applied, res.changes)
                self.assertEqual(applied, ledgers[tgt], msg=f"{cursor}->{tgt}")

        # 重开后已发布的换版轮仍可追补（跨连续换版只交付净变更）
        for snap in snaps.values():
            snap.close()
        ds.close()
        ds = Dataset.open(self.sub())
        self.assertEqual(ds.stations, tuple(self.PI1))
        with ds.read() as snap:
            res = snap.catchup(9)   # v9(创建顺序) -> v12(PI1)，跨三轮换版
            self.assertEqual(
                changes_as_tuples(res.changes),
                direct_diff(ledgers[9], ledgers[12]),
            )
            applied = dict(ledgers[9])
            apply_changes(applied, res.changes)
            self.assertEqual(applied, ledgers[12])
        ds.close()

    def test_reorder_counts_in_retention_window(self):
        """一轮真正重排计入现有 16 轮保留范围（含恰好 16 轮）。"""
        codes = ["A", "B", "C", "D"]
        ds = Dataset.create(self.sub(), codes)
        with ds.correction() as tx:  # v2
            tx.add("r0", "b0", "A", "B", 1, 0.0)
            tx.commit()
        ds.reorder(["D", "C", "B", "A"])  # v3：一轮真正重排
        for _ in range(15):               # v4..v18：15 轮空订正
            with ds.correction() as tx:
                tx.commit()
        self.assertEqual(ds.version_seq, 18)
        with ds.read() as snap:
            res = snap.catchup(2)  # 恰好 16 轮（含一轮换版），有效
            # r0 只换了位置：A 0->3，B 1->2，按两端版本坐标交付
            self.assertEqual(
                changes_as_tuples(res.changes),
                [("r0", ("b0", 0, 1, 1, 0.0), ("b0", 3, 2, 1, 0.0))],
            )
        with ds.correction() as tx:  # v19：换版后的新订正仍按代码提交
            tx.add("r1", "b1", "C", "D", 2, 0.0)
            tx.commit()
        with ds.read() as snap:
            with self.assertRaises(ValidationError):
                snap.catchup(2)  # 17 轮，超窗
            # 16 轮，有效；新记录的位置按换版后的坐标交付（C=1, D=0）
            self.assertEqual(
                changes_as_tuples(snap.catchup(3).changes),
                [("r1", None, ("b1", 1, 0, 2, 0.0))],
            )
        ds.close()


# ---------------------------------------------------------------------- 已开对象


class OpenObjectsTest(TempDirCase):
    """先前打开的快照、它建的工单与未提交订正：换版后仍用旧坐标。"""

    def test_open_snapshot_workorder_and_pending_tx_survive_reorder(self):
        codes = ["A", "B", "C", "D", "E", "F"]
        recs = [
            ("r1", "b1", "A", "B", 5, 0.5),
            ("r2", "b2", "C", "D", 7, 1.0),
            ("r3", "b2", "E", "F", -3, 0.25),
            ("r4", "b3", "F", "A", 2, 0.5),
        ]
        ds = Dataset.create(self.sub(), codes, records=list(recs))  # v2
        with ds.correction() as tx:  # v3
            tx.add("r5", "b4", "B", "C", 1, 0.0)
            tx.commit()
        live = list(recs) + [("r5", "b4", "B", "C", 1, 0.0)]
        seq0 = ds.version_seq
        n = len(codes)

        snap_old = ds.read()
        reqs = [(0, 3, 3, 6, 1)]
        order_old = snap_old.backfill(reqs)
        gaps_old = all_pages(order_old, 2)
        self.assertEqual(gaps_old, brute_gaps(pos_recs(live, codes), reqs, n))
        catch_old = changes_as_tuples(snap_old.catchup(2))
        tx_pending = ds.correction()  # 换版前开启的订正
        tx_pending.add("r6", "b5", "A", "C", 8, 1.0)

        new_order = ["F", "E", "D", "C", "B", "A"]
        seq1 = ds.reorder(new_order)
        self.assertEqual(seq1, seq0 + 1)

        # 换版前开启的未提交订正按过期拒绝；新订正仍按代码提交
        with self.assertRaises(StaleTransactionError):
            tx_pending.commit()

        # 旧快照与它建的工单仍用旧坐标，追补结果也不因换版而变
        self.assertEqual(snap_old.stations, tuple(codes))
        full_old = expand(pos_recs(live, codes))
        self.assertEqual(
            cells_of(snap_old.window(0, n, 0, n)), expected_window(full_old, 0, n, 0, n)
        )
        self.assertEqual(snap_old.pair(0, 1).d, 5)
        self.assertEqual(all_pages(order_old, 2), gaps_old)
        self.assertEqual(order_old.total_gaps, len(gaps_old))
        self.assertEqual(changes_as_tuples(snap_old.catchup(2)), catch_old)

        # 新快照用新坐标：窗口与补采都按新版位置解释
        full_new = expand(pos_recs(live, new_order))
        with ds.read() as snap_new:
            self.assertEqual(snap_new.stations, tuple(new_order))
            self.assertEqual(
                cells_of(snap_new.window(0, n, 0, n)),
                expected_window(full_new, 0, n, 0, n),
            )
            self.assertEqual(
                all_pages(snap_new.backfill(reqs), 2),
                brute_gaps(pos_recs(live, new_order), reqs, n),
            )
        # 新订正按代码提交，证据位置按新版坐标进入追补
        with ds.correction() as tx:
            tx.add("r6", "b5", "A", "C", 8, 1.0)
            tx.commit()
        pos = {c: i for i, c in enumerate(new_order)}
        with ds.read() as snap:
            self.assertEqual(snap.pair(pos["A"], pos["C"]).d, 8)
            got = {ch.record_id: ev_tuple(ch.new) for ch in snap.catchup(seq1).changes}
            self.assertEqual(got, {"r6": ("b5", pos["A"], pos["C"], 8, 1.0)})
        snap_old.close()
        ds.close()


# ---------------------------------------------------------------------- 发布前中断


class ReorderCrashTest(TempDirCase):
    def test_interrupted_reorder_leaves_no_trace(self):
        codes = ["A", "B", "C", "D"]
        recs = [("r1", "b1", "A", "B", 5, 0.5), ("r2", "b2", "C", "A", -3, 1.0)]
        ds = Dataset.create(self.sub(), codes, records=list(recs))  # v2
        with ds.correction() as tx:  # v3
            tx.add("r3", "b3", "D", "B", 2, 0.0)
            tx.commit()
        live = list(recs) + [("r3", "b3", "D", "B", 2, 0.0)]
        order1 = ["D", "C", "B", "A"]
        order2 = ["B", "A", "D", "C"]
        seq1 = ds.reorder(order1)  # v4：成功换版
        self.assertEqual(ds.stations, tuple(order1))

        # 在切换 CURRENT 前中断的换版（模拟崩溃）
        orig_write_current = dataset_mod._write_current

        def boom(path, seq):
            raise RuntimeError("模拟切换 CURRENT 前崩溃")

        dataset_mod._write_current = boom
        try:
            with self.assertRaises(RuntimeError):
                ds.reorder(order2)
        finally:
            dataset_mod._write_current = orig_write_current
        # 中断不改变当前坐标
        self.assertEqual(ds.version_seq, seq1)
        self.assertEqual(ds.stations, tuple(order1))
        ds.close()

        # 重开：只能读到完整的旧版；中断不改变坐标、不占用追补版本
        ds = Dataset.open(self.sub())
        self.assertEqual(ds.version_seq, seq1)
        self.assertEqual(ds.stations, tuple(order1))
        cdir = os.path.join(self.sub(), "changelog")
        self.assertEqual(
            sorted(f for f in os.listdir(cdir) if f.startswith("c_")),
            [f"c_{s:08d}.json" for s in range(2, seq1 + 1)],
        )
        vroot = os.path.join(self.sub(), "versions")
        self.assertEqual(os.listdir(vroot), [f"v_{seq1:08d}"])
        with ds.read() as snap:
            self.assertEqual(snap.stations, tuple(order1))
            self.assertEqual(
                changes_as_tuples(snap.catchup(3)),
                direct_diff(
                    ledger_of_records(live, codes), ledger_of_records(live, order1)
                ),
            )
        # 同一版本号可被后续换版复用
        seq2 = ds.reorder(order2)
        self.assertEqual(seq2, seq1 + 1)
        self.assertEqual(ds.stations, tuple(order2))
        with ds.read() as snap:
            self.assertEqual(snap.stations, tuple(order2))
            # 跨成功换版的追补按两端版本坐标解释
            self.assertEqual(
                changes_as_tuples(snap.catchup(seq1)),
                direct_diff(
                    ledger_of_records(live, order1), ledger_of_records(live, order2)
                ),
            )
            self.assertEqual(
                changes_as_tuples(snap.catchup(3)),
                direct_diff(
                    ledger_of_records(live, codes), ledger_of_records(live, order2)
                ),
            )
        ds.close()
        # 重开读到完整的新版
        ds = Dataset.open(self.sub())
        self.assertEqual(ds.stations, tuple(order2))
        with ds.read() as snap:
            pos = {c: i for i, c in enumerate(order2)}
            self.assertEqual(snap.pair(pos["A"], pos["B"]).d, 5)
        ds.close()


# ---------------------------------------------------------------------- 换版后的选择性


class ReorderSelectivityTest(TempDirCase):
    def test_narrow_window_and_catchup_selectivity_after_reorder(self):
        """换版重建索引后：窄窗仍利用两轴选择性，短追补仍只处理变化记录。"""
        rng = random.Random(9)
        n = 60_000
        codes = make_codes(n)
        n_facts = 20_000
        pairs = set()
        while len(pairs) < n_facts:
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        recs = [
            (f"r{k}", "hot", codes[i], codes[j], rng.randint(-10**6, 10**6), 0.5)
            for k, (i, j) in enumerate(sorted(pairs))
        ]
        ds = Dataset.create(self.sub(), codes, records=recs)
        new_order = list(reversed(codes))  # 完全逆序换版
        ds.reorder(new_order)
        full = expand(pos_recs(recs, new_order))
        with ds.read() as snap:
            self.assertEqual(snap.stations, tuple(new_order))
            for _ in range(60):
                a = rng.randrange(n - 8)
                b = rng.randrange(n - 8)
                w = (a, a + 8, b, b + 8)
                res = snap.window(*w)
                self.assertEqual(cells_of(res), expected_window(full, *w))
                self.assertLess(res.stats["nodes_visited"], 10_000)
                self.assertLess(res.stats["nodes_visited"], n_facts // 5)
        # 换版后的普通订正与短跨度追补仍只处理本轮变化的记录
        r0 = recs[0]
        with ds.correction() as tx:
            tx.revoke("r0")
            tx.revoke("r1")
            tx.add("x1", "fix", r0[2], r0[3], r0[4] + 1, 0.5)
            tx.commit()
        with ds.read() as snap:
            res = snap.catchup(ds.version_seq - 1)
            self.assertEqual(res.stats["records_touched"], 3)
            self.assertEqual(
                {ch.record_id for ch in res.changes}, {"r0", "r1", "x1"}
            )
        ds.close()


# ---------------------------------------------------------------------- 随机交叉核对


class ReorderRandomizedTest(TempDirCase):
    def test_randomized_reorder_and_correction_crosscheck(self):
        rng = random.Random(77)
        n = 12
        codes = [f"S{k}" for k in range(n)]
        ds = Dataset.create(self.sub(), codes)
        live = {}  # rid -> (batch, code_a, code_b, d, e)
        ledgers = {ds.version_seq: {}}
        rid_gen = [0]

        def live_recs():
            return [(rid, *v) for rid, v in live.items()]

        def ledger_of(order):
            return ledger_of_records(live_recs(), order)

        def check_windows(snap, order):
            full = expand(pos_recs(live_recs(), order))
            self.assertEqual(
                cells_of(snap.window(0, n, 0, n)), expected_window(full, 0, n, 0, n)
            )
            for _ in range(15):
                r0, r1 = sorted((rng.randrange(n + 1), rng.randrange(n + 1)))
                c0, c1 = sorted((rng.randrange(n + 1), rng.randrange(n + 1)))
                self.assertEqual(
                    cells_of(snap.window(r0, r1, c0, c1)),
                    expected_window(full, r0, r1, c0, c1),
                )

        def pair_value(a, b):
            """live 中无序对 {a,b} 的折算值（a->b 方向），无证据返回 None。"""
            for bat, ra, rb, rd, re_ in live.values():
                if {ra, rb} == {a, b}:
                    return (rd if (ra, rb) == (a, b) else -rd, re_)
            return None

        snap_frozen = ds.read()  # 钉住初始空版本，之后一直应读到空矩阵
        frozen_order = list(codes)
        for _round in range(10):
            if rng.random() < 0.5:
                # 随机全量重排
                order = list(codes)
                rng.shuffle(order)
                seq = ds.reorder(order)
                self.assertEqual(ds.stations, tuple(order))
            else:
                # 普通订正：撤一批 + 按代码增一批（与剩余证据折算值一致）
                tx = ds.correction()
                for rid in rng.sample(list(live), min(len(live), rng.randint(0, 3))):
                    tx.revoke(rid)
                    live.pop(rid)
                for _ in range(rng.randint(1, 4)):
                    a, b = rng.choice(codes), rng.choice(codes)
                    cur = pair_value(a, b)
                    if cur is None:
                        d = 0 if a == b else rng.randint(-100, 100)
                        e = rng.choice([0.0, 0.5, 1.0])
                    else:
                        d, e = cur
                    if a == b or rng.random() < 0.5:
                        ra, rb, rd = a, b, d
                    else:
                        ra, rb, rd = b, a, -d
                    rid = f"n{rid_gen[0]}"
                    rid_gen[0] += 1
                    tx.add(rid, "bx", ra, rb, rd, e)
                    live[rid] = ("bx", ra, rb, rd, float(e))
                seq = tx.commit()
                order = list(ds.stations)
            ledgers[seq] = ledger_of(order)
            with ds.read() as snap:
                self.assertEqual(snap.stations, tuple(order))
                check_windows(snap, order)
            # 钉住的初始快照仍用旧坐标、读到空矩阵
            self.assertEqual(snap_frozen.stations, tuple(frozen_order))
            self.assertEqual(cells_of(snap_frozen.window(0, n, 0, n)), [])
            # 保留窗口内随机取一个游标核对追补
            target = ds.version_seq
            cursor = rng.randint(max(1, target - CATCHUP_WINDOW), target)
            if cursor in ledgers:
                with ds.read() as snap:
                    res = snap.catchup(cursor)
                self.assertEqual(
                    changes_as_tuples(res.changes),
                    direct_diff(ledgers[cursor], ledgers[target]),
                )
                applied = dict(ledgers[cursor])
                apply_changes(applied, res.changes)
                self.assertEqual(applied, ledgers[target])
        snap_frozen.close()
        ds.close()


if __name__ == "__main__":
    unittest.main()
