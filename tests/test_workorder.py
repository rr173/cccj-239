"""补采工单（Snapshot.backfill）测试。

核对方法：
- 小数据：独立枚举两段间所有无序对，按“取最大门槛、不叠加、镜像不重复”
  算出缺口序列，与工单逐页比对（含跨页、末页、越界空页）；
- 订正只改有效批次不改读数：旧工单页边界不动，新快照建单按新证据判断；
- 十万测点：生成大范围隐藏对，核对精确总数、随机页与直接跳到末页。
"""

import os
import random
import sys
import unittest
from bisect import bisect_right

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairwise_dataset import (  # noqa: E402
    MAX_PAGE_SIZE,
    MAX_REQUIREMENTS,
    Dataset,
    DatasetError,
    ValidationError,
)
from tests.test_pairwise_dataset import TempDirCase, make_codes, to_codes  # noqa: E402


# ---------------------------------------------------------------------- 独立枚举


def brute_gaps(recs, requirements, n):
    """独立展开：返回 [(row, col, required, existing), ...]，按 (row, col) 排序。"""
    # 批次集合：无序对 -> {batch}（不区分原始方向；同批多条只算一次）
    fresh = {}
    for _rid, batch, a, b, _d, _e in recs:
        i, j = (a, b) if a <= b else (b, a)
        fresh.setdefault((i, j), set()).add(batch)
    required = {}
    for a0, a1, b0, b1, k in requirements:
        for i in range(a0, a1):
            for j in range(b0, b1):
                if i == j:
                    continue
                pair = (i, j) if i < j else (j, i)
                if required.get(pair, 0) < k:
                    required[pair] = k
    out = []
    for (i, j), need in required.items():
        have = len(fresh.get((i, j), ()))
        if have < need:
            out.append((i, j, need, have))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def page_tuples(page):
    return [(e.row, e.col, e.required_batches, e.existing_batches) for e in page]


def all_pages(order, size):
    out = []
    idx = 0
    while True:
        page = order.page(idx, size)
        if not page.entries:
            break
        out.extend(page_tuples(page))
        idx += size
    return out


# ---------------------------------------------------------------------- 小数据核对


class WorkOrderCase(TempDirCase):
    def _dataset(self, name, codes, recs):
        return Dataset.create(self.sub(name), codes, records=to_codes(recs, codes))


class SmallDataTest(WorkOrderCase):

    def test_overlapping_requirements_against_enumeration(self):
        n = 24
        codes = make_codes(n)
        rng = random.Random(101)
        recs = []
        rid = 0
        # 每个对被覆盖的测点随机给 0..4 个批次，同批可有多条证据、可反向。
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.55:
                    for b in range(rng.randint(0, 4)):
                        batch = f"B{rng.randrange(3)}"
                        for _copy in range(rng.randint(1, 2)):
                            if rng.random() < 0.5:
                                recs.append((f"r{rid}", batch, i, j, 7, 0.5))
                            else:
                                recs.append((f"r{rid}", batch, j, i, -7, 0.5))
                            rid += 1
        ds = self._dataset("small", codes, recs)
        requirements = [
            (0, 8, 8, 16, 2),       # 左半 x 右半
            (4, 12, 14, 20, 3),     # 与上一条重叠，门槛更高
            (8, 16, 0, 8, 2),       # 镜像写法：不能重复派单
            (16, 24, 2, 6, 1),      # 边缘段
            (0, 7, 9, 11, 4),       # 高门槛，多数不达标
        ]
        expected = brute_gaps(recs, requirements, n)
        with ds.read() as snap:
            order = snap.backfill(requirements)
            self.assertEqual(order.version_seq, snap.version_seq)
            self.assertEqual(order.total_gaps, len(expected))
            self.assertGreater(len(expected), 0)
            for size in (1, 2, 5, 7, 1000):
                self.assertEqual(all_pages(order, size), expected, msg=f"size={size}")
            # 页级细节：末页不足、next_index、越界空页、每条只交付一次
            size = 6
            seen = set()
            idx = 0
            while idx < order.total_gaps:
                page = order.page(idx, size)
                self.assertEqual(page.version_seq, snap.version_seq)
                self.assertEqual(page.total_gaps, len(expected))
                self.assertLessEqual(len(page), size)
                self.assertEqual(page.next_index, idx + len(page))
                for tup in page_tuples(page):
                    self.assertNotIn((tup[0], tup[1]), seen)
                    seen.add((tup[0], tup[1]))
                idx += size
            self.assertEqual(len(seen), len(expected))
            last = order.page(order.total_gaps - 1, 10)
            self.assertEqual(len(last), 1)
            self.assertEqual(page_tuples(last), [expected[-1]])
            for bad in (order.total_gaps, order.total_gaps + 1, order.total_gaps + 5000):
                page = order.page(bad, 10)
                self.assertEqual(page.entries, ())
                self.assertEqual(page.total_gaps, len(expected))
                self.assertEqual(page.next_index, bad)

            # 单条要求、且只剩一个缺口
            one = snap.backfill([(0, 1, 1, 2, 1)])
            gaps = brute_gaps(recs, [(0, 1, 1, 2, 1)], n)
            self.assertEqual(one.total_gaps, len(gaps))
            self.assertEqual(all_pages(one, 3), gaps)

            # 空要求集 -> 空工单
            empty = snap.backfill([])
            self.assertEqual(empty.total_gaps, 0)
            self.assertEqual(empty.page(0, 10).entries, ())
            self.assertEqual(empty.page(0, 10).total_gaps, 0)

            # 不受要求约束的对不进入工单：只覆盖 [0,2)x[2,4)
            only = snap.backfill([(0, 2, 2, 4, 1)])
            self.assertTrue(all(0 <= e.row < 2 and 2 <= e.col < 4 for e in only.page(0, 50)))
        ds.close()

    def test_required_is_max_not_sum_and_self_excluded(self):
        n = 6
        codes = make_codes(n)
        # 每个对都恰好 2 个批次
        recs = []
        for i in range(n):
            for j in range(i + 1, n):
                recs.append((f"a{i}_{j}", "b1", i, j, 1, 0))
                recs.append((f"b{i}_{j}", "b2", j, i, -1, 0))
        ds = self._dataset("maxsum", codes, recs)
        with ds.read() as snap:
            # 同一批对命中两条要求：门槛 2 与 3，取 3 而不是 5
            order = snap.backfill([(0, 2, 2, 4, 2), (0, 2, 2, 4, 3)])
            got = page_tuples(order.page(0, 100))
            self.assertEqual(len(got), 4)  # 2x2 个对
            self.assertTrue(all(t[2] == 3 and t[3] == 2 for t in got))
            # 镜像要求不重复派单
            order = snap.backfill([(0, 2, 2, 4, 3), (2, 4, 0, 2, 3)])
            self.assertEqual(order.total_gaps, 4)
            self.assertEqual(all_pages(order, 100), page_tuples(snap.backfill(
                [(0, 2, 2, 4, 3)]).page(0, 100)))
            # 两段有重叠：段内自对被排除，段内其他对仍按时覆盖（两段都含的
            # 区域里 i<j 全部命中）；镜像写法覆盖同一批对，不重复派单。
            overlapping = [(0, 4, 2, 6, 3)]
            expected = brute_gaps(recs, overlapping, n)
            order = snap.backfill(overlapping)
            self.assertEqual(order.total_gaps, len(expected))
            self.assertEqual(order.total_gaps, 4 * 4 - 3)  # 16 对去掉 (2,2),(3,3) 两个自对
            self.assertEqual(all_pages(order, 100), expected)
            mirrored = snap.backfill([(0, 4, 2, 6, 3), (2, 6, 0, 4, 3)])
            self.assertEqual(mirrored.total_gaps, len(expected))
            self.assertEqual(all_pages(mirrored, 100), expected)
            # 两段相同：覆盖段内所有 i<j；本测试每个对都有 2 个批次，
            # 门槛 2 全达标，门槛 3 全缺口。
            same = snap.backfill([(0, 4, 0, 4, 2)])
            self.assertEqual(same.total_gaps, 0)
            self.assertEqual(page_tuples(same.page(0, 10)), [])
            same = snap.backfill([(0, 4, 0, 4, 3)])  # 门槛高于所有对 -> 全缺口
            expected = [(i, j, 3, 2) for i in range(4) for j in range(i + 1, 4)]
            self.assertEqual(page_tuples(same.page(0, 100)), expected)
            self.assertEqual(same.total_gaps, 6)
        ds.close()

    def test_validation(self):
        codes = make_codes(6)
        ds = self._dataset("val", codes, [])
        with ds.read() as snap:
            for bad in [
                [(0, 0, 2, 4, 1)],                          # 空区间
                [(0, 3, 2, 7, 1)],                          # 越界
                [(-1, 3, 2, 4, 1)],                         # 负端点
                [(3, 2, 0, 1, 1)],                          # 反序区间
                [(0, 3, 3, 4, 0)],                          # 门槛非正
                [(0, 3, 3, 4, 1.5)],                        # 门槛非整数
                [(0, 3, 3, 4, True)],                       # 门槛为 bool
                [(0, 3, 3, 4)],                             # 元组长度不对
                [("0", 3, 3, 4, 1)],                        # 端点非整数
                [None],                                     # 非元组
                "not-reqs",                                 # 字符串
            ]:
                with self.assertRaises(ValidationError, msg=repr(bad)):
                    snap.backfill(bad)
            # 两段重叠是合法的；只有空区间、越界等才报错
            snap.backfill([(0, 4, 2, 6, 1)])
            with self.assertRaises(ValidationError):
                snap.backfill([(2, 6, 2, 2, 1)])
            # 超过 128 条
            too_many = [(0, 1, 1, 2, 1)] * (MAX_REQUIREMENTS + 1)
            with self.assertRaises(ValidationError):
                snap.backfill(too_many)
            # 恰好 128 条是允许的（全部同一规则，只是重复）
            self.assertEqual(snap.backfill([(0, 1, 1, 2, 1)] * MAX_REQUIREMENTS).total_gaps, 1)

            order = snap.backfill([(0, 1, 1, 2, 1)])
            for bad_size in (0, -1, MAX_PAGE_SIZE + 1, 2.0, True):
                with self.assertRaises(ValidationError, msg=repr(bad_size)):
                    order.page(0, bad_size)
            for bad_index in (-1, 1.0, True):
                with self.assertRaises(ValidationError, msg=repr(bad_index)):
                    order.page(bad_index, 10)
            order.page(0, 1)
            order.page(0, MAX_PAGE_SIZE)
        # 快照关闭后不能再取页
        order = None
        with ds.read() as snap:
            order = snap.backfill([(0, 1, 1, 2, 1)])
            order.page(0, 10)
        with self.assertRaises(DatasetError):
            order.page(0, 10)
        ds.close()


# ---------------------------------------------------------------------- 订正只改批次


class BatchOnlyCorrectionTest(WorkOrderCase):

    def test_correction_changes_batches_not_readings(self):
        n = 12
        codes = make_codes(n)
        # (0,1) 一个批次，(2,3) 两个批次；读数都一样，方便只换批次。
        recs = [
            ("r1", "b1", 0, 1, 5, 1),
            ("r2", "b2", 2, 3, 5, 1),
            ("r3", "b3", 3, 2, -5, 1),
        ]
        ds = self._dataset("rev", codes, recs)
        requirements = [(0, 2, 2, 4, 2)]
        expected_before = brute_gaps(recs, requirements, n)
        self.assertEqual(
            expected_before,
            [(0, 2, 2, 0), (0, 3, 2, 0), (1, 2, 2, 0), (1, 3, 2, 0)],
        )

        with ds.read() as snap_old:
            old = snap_old.backfill(requirements)
            old_page = old.page(0, 10)
            old_total = old.total_gaps
            self.assertEqual(old_total, 4)
            self.assertEqual(page_tuples(old_page), expected_before)
            boundary = page_tuples(old.page(1, 2))
            self.assertEqual(boundary, expected_before[1:3])

            # 订正：给 (0,1) 补一个批次（读数不变），并给 (1,3) 加两个批次；
            # 同时撤回 (2,3) 的一个批次（读数仍一致）。
            with ds.correction() as tx:
                tx.add("x1", "b9", codes[0], codes[1], 5, 1)
                tx.add("x2", "b7", codes[1], codes[3], 5, 1)
                tx.add("x3", "b8", codes[3], codes[1], -5, 1)
                tx.revoke("r3")
                tx.commit()

            live = [("r1", "b1", 0, 1, 5, 1), ("r2", "b2", 2, 3, 5, 1),
                    ("x1", "b9", 0, 1, 5, 1), ("x2", "b7", 1, 3, 5, 1),
                    ("x3", "b8", 3, 1, -5, 1)]
            expected_after = brute_gaps(live, requirements, n)
            self.assertEqual(
                expected_after,
                [(0, 2, 2, 0), (0, 3, 2, 0), (1, 2, 2, 0)],
            )

            # 读数（d/e）没变，只有有效批次变了
            self.assertEqual(snap_old.pair(0, 1).d, 5)

            with ds.read() as snap_new:
                new = snap_new.backfill(requirements)
                self.assertEqual(new.total_gaps, 3)
                self.assertEqual(page_tuples(new.page(0, 10)), expected_after)
                self.assertGreater(new.version_seq, old.version_seq)

            # 旧工单的页边界与内容保持冻结
            self.assertEqual(old.total_gaps, old_total)
            self.assertEqual(page_tuples(old.page(0, 10)), expected_before)
            self.assertEqual(page_tuples(old.page(1, 2)), expected_before[1:3])
            self.assertEqual(page_tuples(old.page(3, 2)), expected_before[3:])

        ds.close()

    def test_revoke_to_zero_and_add_back_batches(self):
        n = 8
        codes = make_codes(n)
        recs = [("a", "b1", 0, 1, 3, 0), ("b", "b2", 0, 1, 3, 0), ("c", "b1", 0, 1, 3, 0)]
        ds = self._dataset("zero", codes, recs)
        # 第一段 [0,2) 与第二段 [2,4)，覆盖 (0,1) 与另外三条无证据的对
        req = [(0, 2, 2, 4, 2)]
        with ds.read() as snap:
            order = snap.backfill(req)
            # (0,1) 有 2 个批次（b1 两条只算一次）-> 达标
            got = {tuple(t[:2]): t[2:] for t in page_tuples(order.page(0, 100))}
            self.assertNotIn((0, 1), got)
            self.assertEqual(got[(1, 2)], (2, 0))
        with ds.correction() as tx:
            tx.revoke("b")
            tx.commit()
        # 撤掉一个批次后，(0,1) 也变缺口；读数仍一致，不变
        with ds.read() as snap:
            self.assertEqual(snap.pair(0, 1).d, 3)
            order = snap.backfill(req)
            got = {tuple(t[:2]): t[2:] for t in page_tuples(order.page(0, 100))}
            self.assertEqual(got[(1, 2)], (2, 0))
            order_pair = snap.backfill([(0, 1, 1, 2, 2)])
            self.assertEqual(page_tuples(order_pair.page(0, 10)), [(0, 1, 2, 1)])
        ds.close()


# ---------------------------------------------------------------------- 十万测点


class LargeScaleTest(TempDirCase):
    def test_hundred_thousand_stations_wide_range(self):
        n = 100_000
        codes = make_codes(n)
        rng = random.Random(2024)
        # 少量事实：一部分在宽段内、一部分在段外
        recs = []
        wide = [(5, 40_000, 40_000, 95_000, 2)]     # 隐藏对约 35k * 55k = 1.9e9
        for t in range(300):
            i = rng.randrange(5, 40_000)
            j = rng.randrange(40_000, 95_000)
            batches = {f"W{t % 3}", f"W{(t + 1) % 3}"}
            if t % 10 == 0:
                batches = {"W0"}
            for bidx, batch in enumerate(sorted(batches)):
                recs.append((f"w{t}_{bidx}", batch, i, j, 1, 0))
        # 段外噪声：不应影响工单
        for t in range(200):
            recs.append((f"o{t}", "OUT", 0, 99_999, 1, 0))
        # 另一个大段，用来验证镜像不重复
        requirements = wide + [(40_000, 95_000, 5, 40_000, 2)]
        ds = Dataset.create(self.sub("big"), codes, records=to_codes(recs, codes))

        # 精确总数：用不展开测点的公式独立算一遍
        # 段划分： [0,5) [5,40000) [40000,95000) [95000,100000)
        # 被覆盖对 = [5,40000) x [40000,95000)
        total_pairs = 39_995 * 55_000
        all_pairs = set()  # 实际有证据的（行,列），用于核对交付条目
        fresh = {}
        for _rid, batch, a, b, _d, _e in recs:
            i, j = (a, b) if a <= b else (b, a)
            if 5 <= i < 40_000 and 40_000 <= j < 95_000:
                all_pairs.add((i, j))
                fresh.setdefault((i, j), set()).add(batch)
        satisfied_pairs = {p for p, s in fresh.items() if len(s) >= 2}
        # 300 条随机记录可能有重复对；只断言规模合理
        n_satisfied = len(satisfied_pairs)
        self.assertLessEqual(n_satisfied, 300)
        expected_total = total_pairs - n_satisfied
        unfilled = {p: len(s) for p, s in fresh.items() if len(s) < 2}

        with ds.read() as snap:
            order = snap.backfill(requirements)
            self.assertEqual(order.version_seq, snap.version_seq)
            self.assertEqual(order.total_gaps, expected_total)
            self.assertGreater(order.total_gaps, 1_900_000_000)

            # 直接跳到末页：不遍历前面的缺口
            last = order.page(order.total_gaps - 1, 10)
            self.assertEqual(len(last), 1)
            entry = last.entries[0]
            self.assertEqual((entry.row, entry.col), (39_999, 94_999))
            self.assertEqual(entry.required_batches, 2)
            self.assertEqual(entry.existing_batches, 0)

            # 倒数第二页：从任意靠后序号取一页，结果连续且有序
            idx = order.total_gaps - 5
            tail = order.page(idx, 10)
            self.assertEqual(len(tail), 5)
            keys = [(e.row, e.col) for e in tail]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(keys[-1], (39_999, 94_999))

            # 随机页与顺序统计独立核对：不枚举全体对，只按段内公式算期望。
            for _ in range(30):
                pos = rng.randrange(order.total_gaps)
                page = order.page(pos, 3)
                self.assertEqual(page.index, pos)
                for off, e in enumerate(page):
                    self.assertEqual((e.row, e.col), self._nth_pair(pos + off, satisfied_pairs))
                    self.assertGreaterEqual(e.row, 5)
                    self.assertLess(e.row, 40_000)
                    self.assertGreaterEqual(e.col, 40_000)
                    self.assertLess(e.col, 95_000)
                    self.assertLess(e.row, e.col)
                    self.assertEqual(e.required_batches, 2)
                    self.assertEqual(e.existing_batches, unfilled.get((e.row, e.col), 0))
                    self.assertNotIn((e.row, e.col), satisfied_pairs)

            # 页边界精确：某行的末端与下一行的开头
            first = order.page(0, 3)
            self.assertEqual([(e.row, e.col) for e in first], [(5, 40_000), (5, 40_001), (5, 40_002)])
            row_end = order.page(54_999, 2)  # 第 0 行的最后一个被覆盖对是 (5,94999)
            self.assertEqual([(e.row, e.col) for e in row_end][0], (5, 94_999))
        ds.close()
        # 工单不随快照关闭而丢总数，但不能取页
        self.assertEqual(order.total_gaps, expected_total)
        with self.assertRaises(DatasetError):
            order.page(0, 10)

    @staticmethod
    def _nth_pair(pos, satisfied):
        """第 pos 个缺口（0 起）：行主序遍历 [5,40000) x [40000,95000)，
        扣除已达标事实。用小范围移动定位（这里满意事实只有几百个）。"""
        # 先按矩形序号粗定位，再逐个跨过已达标事实。
        sat = sorted(satisfied)
        # 二分：找最小矩形序号 r 使 r+1 - #sat<=pair(r) >= pos+1
        lo, hi = 0, 39_995 * 55_000 - 1
        while lo < hi:
            mid = (lo + hi) // 2
            q, r = divmod(mid, 55_000)
            pair = (5 + q, 40_000 + r)
            if (mid + 1) - bisect_right(sat, pair) >= pos + 1:
                hi = mid
            else:
                lo = mid + 1
        q, r = divmod(lo, 55_000)
        return (5 + q, 40_000 + r)


class WorkOrderCostTest(WorkOrderCase):
    def test_cost_independent_of_hidden_pairs(self):
        """隐藏对从 1e6 级涨到 1e9 级，建单与跳页开销不应随之线性增长。"""
        import time

        n = 100_000
        codes = make_codes(n)
        # 少量事实（都在段内、大多达标或只有 1 批），用于证明只处理实际事��。
        recs = []
        for t in range(200):
            recs.append((f"s{t}", "b1", t, 60_000 + t, 1, 0))
            recs.append((f"u{t}", "b2", 60_000 + t, t, -1, 0))
        ds = Dataset.create(self.sub("cost"), codes, records=to_codes(recs, codes))

        def measure(requirements):
            start = time.perf_counter()
            with ds.read() as snap:
                order = snap.backfill(requirements)
                build = time.perf_counter() - start
                start = time.perf_counter()
                for idx in (0, order.total_gaps // 2, order.total_gaps - 1):
                    order.page(idx, 1000)
                page = time.perf_counter() - start
            return order.total_gaps, build, page

        small_gaps, small_build, small_page = measure([(0, 1_000, 1_000, 2_000, 1)])
        big_gaps, big_build, big_page = measure([(0, 50_000, 50_000, 100_000, 1)])
        self.assertGreater(big_gaps, 1000 * small_gaps)  # 隐藏对规模差三个量级
        # 建单与取页都不是“按缺口数”工作：阈值宽裕，只排除线性实现。
        self.assertLess(big_build, small_build * 10 + 0.5)
        self.assertLess(big_page, small_page * 10 + 0.5)
        # 一次取满整页直接命中，不必遍历前面几十亿个缺口
        self.assertLess(big_page, 1.0)
        with ds.read() as snap:
            order = snap.backfill([(0, 50_000, 50_000, 100_000, 1)])
            self.assertEqual(len(order.page(0, 1000)), 1000)
            self.assertEqual(len(order.page(order.total_gaps - 1, 1000)), 1)
            # 末页只有一个缺口且是最大位置对
            e = order.page(order.total_gaps - 1, 1000).entries[0]
            self.assertEqual((e.row, e.col), (49_999, 99_999))
            self.assertEqual(e.existing_batches, 0)
        ds.close()


class WorkOrderReuseTest(TempDirCase):
    def test_prepare_reused_and_no_original_rescan(self):
        n = 60
        codes = make_codes(n)
        recs = []
        for i in range(0, 20):
            for j in range(20, 40):
                recs.append((f"r{i}_{j}", "b1", i, j, 1, 0))
        ds = Dataset.create(self.sub("reuse"), codes, records=to_codes(recs, codes))
        with ds.read() as snap:
            order = snap.backfill([(0, 20, 20, 40, 1)])
            self.assertEqual(order.total_gaps, 0)
            # 建单后改动原始快照数据不应影响已冻结的准备结果：
            # 伪造一个新增事实到版本数组，若取页重新扫描原始数据就会露馅。
            v = snap._v
            v.I.append(0)
            v.J.append(30)
            v.D.append(1)
            v.E.append(0.0)
            v.ev_off.append(len(v.ev_off) and v.ev_off[-1])
            self.assertEqual(order.total_gaps, 0)
            self.assertEqual(order.page(0, 10).entries, ())
        ds.close()


if __name__ == "__main__":
    unittest.main()
