"""pairwise_dataset 的测试。

核对方法：把记录独立展开成全矩阵 dict[(row,col)] -> (d,e,orientation,evs)，
再与数据集的窗口读取逐单元比对（窗形、负值、零、反向多证据、对角、并行读写）。
"""

import os
import random
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairwise_dataset import (  # noqa: E402
    FORWARD,
    REVERSE,
    SELF,
    ConflictError,
    Dataset,
    DuplicateRecordError,
    StaleTransactionError,
    UnknownRecordError,
    ValidationError,
)


# ---------------------------------------------------------------------- 工具


def make_codes(n):
    return [f"P{i:07d}" for i in range(n)]


def norm_evs(evs):
    """把证据规范成可比较的有序元组。"""
    return tuple(
        sorted((ev.record_id, ev.batch, ev.src, ev.dst, ev.raw_d, ev.raw_e) for ev in evs)
    )


def cells_of(res):
    return [
        (c.row, c.col, c.d, c.e, c.orientation, norm_evs(c.evidence)) for c in res.cells
    ]


def expand(recs):
    """独立展开成全矩阵。recs: (rid, batch, a, b, d, e)，a/b 为排序位置。

    返回 dict[(row,col)] -> (d, e, orientation, evs)，evs 按 rid 排序。
    """
    pairs = {}
    for rid, batch, a, b, d, e in recs:
        i, j = (a, b) if a <= b else (b, a)
        pairs.setdefault((i, j), []).append((rid, batch, a, b, int(d), float(e)))
    full = {}
    for (i, j), evs in pairs.items():
        vals = set()
        for rid, batch, a, b, rd, re_ in evs:
            vals.add((rd if a <= b else -rd, re_))
        assert len(vals) == 1, f"测试数据本身不一致: {(i, j)}"
        (dv, ev_), = vals
        nev = tuple(sorted(evs))
        full[(i, j)] = (dv, ev_, SELF if i == j else FORWARD, nev)
        if i != j:
            full[(j, i)] = (-dv, ev_, REVERSE, nev)
    return full


def expected_window(full, r0, r1, c0, c1):
    out = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            cell = full.get((r, c))
            if cell is not None:
                d, e, o, evs = cell
                out.append((r, c, d, e, o, evs))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def gen_records(rng, n_stations, n_pairs, max_ev=3, self_ok=True, zero_frac=0.15):
    """生成折算值一致的随机记录（同对多证据、两方向、自对、显式零）。"""
    pairs = set()
    while len(pairs) < n_pairs:
        a = rng.randrange(n_stations)
        b = rng.randrange(n_stations)
        if a == b and not self_ok:
            continue
        pairs.add((min(a, b), max(a, b)))
    recs = []
    for k, (i, j) in enumerate(sorted(pairs)):
        if i == j:
            d = 0
        elif rng.random() < zero_frac:
            d = 0
        else:
            d = rng.randint(-10**6, 10**6)
        e = rng.choice([0, 1, 2, 0.5, 3.25])
        for t in range(rng.randint(1, max_ev)):
            if i == j or rng.random() < 0.5:
                a, b, rd = i, j, d
            else:
                a, b, rd = j, i, -d
            recs.append((f"rec{k}_{t}", f"batch{k % 5}", a, b, rd, e))
    rng.shuffle(recs)
    return recs


def to_codes(recs, codes):
    return [(rid, bat, codes[a], codes[b], d, e) for rid, bat, a, b, d, e in recs]


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="pwd_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def sub(self, name="ds"):
        return os.path.join(self.dir, name)


# ---------------------------------------------------------------------- 基础语义


class BasicSemanticsTest(TempDirCase):
    def test_forward_reverse_zero_negative_self(self):
        codes = make_codes(6)
        recs = [
            ("r1", "b1", 0, 1, 5, 1),       # 正向
            ("r2", "b1", 3, 1, -7, 0.5),    # 反向记录（从 3 到 1），折算 (1,3)=7
            ("r3", "b2", 2, 4, 0, 2),       # 显式零（不是缺测）
            ("r4", "b2", 5, 5, 0, 0),       # 自对
            ("r5", "b3", 4, 0, 9, 1),       # 反向记录，折算 (0,4)=-9
        ]
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        full = expand(recs)
        with ds.read() as snap:
            res = snap.window(0, 6, 0, 6)
            self.assertEqual(cells_of(res), expected_window(full, 0, 6, 0, 6))
            # 按任一测点的行列取同一份结果
            c1 = snap.pair(1, 3)
            c2 = snap.pair(3, 1)
            self.assertEqual(c1.d, 7)
            self.assertEqual(c2.d, -7)
            self.assertEqual(norm_evs(c1.evidence), norm_evs(c2.evidence))
            self.assertEqual(c1.orientation, FORWARD)
            self.assertEqual(c2.orientation, REVERSE)
            # 显式零存在，缺测对不存在
            self.assertEqual(snap.pair(2, 4).d, 0)
            self.assertIsNone(snap.pair(0, 2))
            self.assertEqual(snap.pair(5, 5).orientation, SELF)
            # 负值折算
            self.assertEqual(snap.pair(0, 4).d, -9)
            self.assertEqual(snap.pair(4, 0).d, 9)
        ds.close()

    def test_initial_load_merge_and_conflict(self):
        codes = make_codes(4)
        # 同对两方向、折算值相同 -> 合并证据
        ds = Dataset.create(self.sub("a"), codes)
        with ds.correction() as tx:
            tx.add("r1", "b1", codes[0], codes[1], 5, 1)
            tx.add("r2", "b1", codes[1], codes[0], -5, 1)
            tx.commit()
        with ds.read() as snap:
            cell = snap.pair(0, 1)
            self.assertEqual(cell.d, 5)
            self.assertEqual([ev.record_id for ev in cell.evidence], ["r1", "r2"])
            rev = snap.pair(1, 0)
            self.assertEqual(rev.d, -5)
            self.assertEqual([ev.record_id for ev in rev.evidence], ["r1", "r2"])
        # 冲突：折算值不同，拒绝发布而不是用最后一行盖掉
        tx = ds.correction()
        tx.add("r3", "b2", codes[0], codes[1], 6, 1)
        with self.assertRaises(ConflictError):
            tx.commit()
        with ds.read() as snap:
            self.assertEqual(snap.pair(0, 1).d, 5)
            self.assertEqual([ev.record_id for ev in snap.pair(0, 1).evidence], ["r1", "r2"])
        ds.close()
        # 首装冲突同样报错
        with self.assertRaises(ConflictError):
            Dataset.create(
                self.sub("b"),
                codes,
                records=[
                    ( "x1", "b1", codes[0], codes[1], 5, 1),
                    ("x2", "b1", codes[1], codes[0], -4, 1),  # 折算 4 != 5
                ],
            )

    def test_diagonal_sweep_shapes(self):
        codes = make_codes(10)
        recs = [
            ("r1", "b1", 5, 2, -11, 1),   # 折算 (2,5)=11
            ("r2", "b1", 3, 3, 0, 0),     # 自对
            ("r3", "b2", 7, 8, -4, 2),    # 正向负值
        ]
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        full = expand(recs)
        with ds.read() as snap:
            # 完全扫过对角线：两个真实朝向各一条、对角单元一条
            res = snap.window(0, 10, 0, 10)
            got = cells_of(res)
            self.assertEqual(got, expected_window(full, 0, 10, 0, 10))
            orient = {(c.row, c.col): c.orientation for c in res.cells}
            self.assertEqual(orient[(2, 5)], FORWARD)
            self.assertEqual(orient[(5, 2)], REVERSE)
            self.assertEqual(orient[(3, 3)], SELF)
            self.assertEqual(len(res.cells), 5)
            # 方形窗口同时包含 (2,5) 与 (5,2)
            self.assertEqual(len(snap.window(2, 6, 2, 6).cells), 3)
            # 只含一个朝向的窄窗
            res = snap.window(5, 6, 2, 3)
            self.assertEqual(len(res.cells), 1)
            c = res.cells[0]
            self.assertEqual((c.row, c.col, c.d, c.orientation), (5, 2, -11, REVERSE))
            self.assertEqual([ev.record_id for ev in c.evidence], ["r1"])
            res = snap.window(2, 3, 5, 6)
            self.assertEqual(len(res.cells), 1)
            self.assertEqual((res.cells[0].d, res.cells[0].orientation), (11, FORWARD))
            # 只含对角的窗口
            res = snap.window(0, 4, 0, 4)
            self.assertEqual([(c.row, c.col, c.orientation) for c in res.cells], [(3, 3, SELF)])
        ds.close()

    def test_window_sorted_and_evidence_index(self):
        codes = make_codes(12)
        rng = random.Random(5)
        recs = gen_records(rng, 12, 40, max_ev=3)
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        with ds.read() as snap:
            res = snap.window(0, 12, 0, 12)
            keys = [(c.row, c.col) for c in res.cells]
            self.assertEqual(keys, sorted(keys))
            # 窗内证据索引 = 各单元证据的并集，反向单元指向同一批证据
            idx = {}
            for c in res.cells:
                for ev in c.evidence:
                    idx.setdefault(ev.record_id, ev)
            self.assertEqual(res.evidence_index, idx)
            for c in res.cells:
                if c.orientation == REVERSE:
                    fwd = snap.pair(c.col, c.row)
                    self.assertEqual(norm_evs(fwd.evidence), norm_evs(c.evidence))
        ds.close()

    def test_validation(self):
        codes = make_codes(4)
        ds = Dataset.create(self.sub(), codes)
        tx = ds.correction()
        with self.assertRaises(ValidationError):
            tx.add("s1", "b", codes[0], codes[0], 1, 0)      # 自对 d!=0
        with self.assertRaises(ValidationError):
            tx.add("s2", "b", codes[0], codes[1], 1, -0.5)   # 负误差
        with self.assertRaises(ValidationError):
            tx.add("s3", "b", codes[0], "NOPE", 1, 0)        # 未知测点
        with self.assertRaises(ValidationError):
            tx.add("s4", "b", codes[0], codes[1], 1.5, 0)    # d 非整数
        tx.rollback()
        tx = ds.correction()
        tx.add("ok", "b", codes[0], codes[1], 3, 0)
        with self.assertRaises(DuplicateRecordError):
            tx.add("ok", "b", codes[0], codes[1], 3, 0)
        tx.commit()
        tx2 = ds.correction()
        with self.assertRaises(DuplicateRecordError):
            tx2.add("ok", "b", codes[0], codes[1], 3, 0)     # 与已提交记录重复
        with self.assertRaises(UnknownRecordError):
            tx2.revoke("missing")                            # 撤回不存在的记录
        tx2.rollback()
        with ds.read() as snap:
            for bad in [(-1, 2, 0, 2), (0, 5, 0, 2), (2, 1, 0, 2), (0, 2, 0, 9)]:
                with self.assertRaises(ValidationError):
                    snap.window(*bad)
            with self.assertRaises(ValidationError):
                snap.pair(0, 4)
        ds.close()


# ---------------------------------------------------------------------- 订正事务


class CorrectionTest(TempDirCase):
    def test_revoke_and_add_same_tx(self):
        codes = make_codes(6)
        recs = [("r1", "b1", 0, 1, 5, 1), ("r2", "b1", 2, 3, 9, 0), ("r3", "b1", 4, 4, 0, 0)]
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        live = {r[0]: r for r in recs}
        # 同事务撤旧增新：先加不同值（临时不一致）再撤旧，不应被拒绝
        tx = ds.correction()
        tx.add("r10", "b2", codes[0], codes[1], 7, 1)   # 与 r1 的 5 临时冲突
        tx.revoke("r1")
        tx.commit()
        live.pop("r1")
        live["r10"] = ("r10", "b2", 0, 1, 7, 1)
        full = expand(live.values())
        with ds.read() as snap:
            self.assertEqual(snap.pair(0, 1).d, 7)
            self.assertEqual(cells_of(snap.window(0, 6, 0, 6)), expected_window(full, 0, 6, 0, 6))
        # 最终不一致 -> 拒绝发布，数据集保持原版本
        tx = ds.correction()
        tx.add("r11", "b3", codes[0], codes[1], 8, 1)
        seq_before = ds.version_seq
        with self.assertRaises(ConflictError):
            tx.commit()
        self.assertEqual(ds.version_seq, seq_before)
        with ds.read() as snap:
            self.assertEqual(snap.pair(0, 1).d, 7)
        # 撤空证据 -> 该对不再生成单元
        tx = ds.correction()
        tx.revoke("r2")
        tx.commit()
        live.pop("r2")
        full = expand(live.values())
        with ds.read() as snap:
            self.assertIsNone(snap.pair(2, 3))
            self.assertEqual(cells_of(snap.window(0, 6, 0, 6)), expected_window(full, 0, 6, 0, 6))
        # 撤回后又允许同身份重新加入（下一事务）
        tx = ds.correction()
        tx.add("r2", "b9", codes[2], codes[3], 9, 0)
        tx.commit()
        with ds.read() as snap:
            self.assertEqual(snap.pair(2, 3).d, 9)
        ds.close()

    def test_stale_transaction(self):
        codes = make_codes(4)
        ds = Dataset.create(self.sub(), codes)
        tx1 = ds.correction()
        tx2 = ds.correction()
        tx1.add("a", "b", codes[0], codes[1], 1, 0)
        tx1.commit()
        tx2.add("b", "b", codes[0], codes[1], 1, 0)
        with self.assertRaises(StaleTransactionError):
            tx2.commit()
        ds.close()

    def test_correction_randomized_crosscheck(self):
        rng = random.Random(21)
        n = 40
        codes = make_codes(n)
        recs = gen_records(rng, n, 120, max_ev=2)
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        live = {r[0]: r for r in recs}
        rid_gen = [0]

        def pair_value(pair):
            for r in live.values():
                a, b = r[2], r[3]
                p = (a, b) if a <= b else (b, a)
                if p == pair:
                    return (r[4] if a <= b else -r[4], float(r[5]))
            return None

        for _round in range(8):
            full_before = expand(live.values())
            snap_before = ds.read()
            tx = ds.correction()
            # 撤一批
            live_rids = list(live)
            for rid in rng.sample(live_rids, min(15, len(live_rids))):
                tx.revoke(rid)
                live.pop(rid)
            # 增一批：新对或与剩余证据折算值一致
            for _ in range(15):
                a, b = rng.randrange(n), rng.randrange(n)
                i, j = min(a, b), max(a, b)
                rid = f"n{rid_gen[0]}"
                rid_gen[0] += 1
                cur = pair_value((i, j))
                if cur is None:
                    d = 0 if i == j else rng.randint(-500, 500)
                    e = rng.choice([0, 1, 0.5])
                else:
                    d, e = cur
                if rng.random() < 0.5 or i == j:
                    rec = (rid, "bx", i, j, d, e)
                else:
                    rec = (rid, "bx", j, i, -d, e)
                tx.add(rid, "bx", codes[rec[2]], codes[rec[3]], rec[4], rec[5])
                live[rid] = rec
            tx.commit()
            full_after = expand(live.values())
            # 旧快照仍读到旧版本（无新旧混搭），新快照读到新版本
            for _ in range(30):
                r0, r1 = sorted((rng.randrange(n + 1), rng.randrange(n + 1)))
                c0, c1 = sorted((rng.randrange(n + 1), rng.randrange(n + 1)))
                self.assertEqual(
                    cells_of(snap_before.window(r0, r1, c0, c1)),
                    expected_window(full_before, r0, r1, c0, c1),
                )
                with ds.read() as snap_new:
                    self.assertEqual(
                        cells_of(snap_new.window(r0, r1, c0, c1)),
                        expected_window(full_after, r0, r1, c0, c1),
                    )
            snap_before.close()
        ds.close()


# ---------------------------------------------------------------------- MVCC / 并发


class MvccTest(TempDirCase):
    def test_snapshot_isolation_and_gc(self):
        codes = make_codes(8)
        recs = [("r1", "b1", 1, 2, 5, 1), ("r2", "b1", 3, 4, 8, 0)]
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        v0 = ds.version_seq
        full0 = expand(recs)
        snap0 = ds.read()  # 钉住订正前版本
        tx = ds.correction()
        tx.revoke("r1")
        tx.add("r9", "b2", codes[1], codes[2], 50, 1)
        tx.commit()
        live = {"r2": recs[1], "r9": ("r9", "b2", 1, 2, 50, 1)}
        full1 = expand(live.values())
        tx = ds.correction()
        tx.add("r10", "b3", codes[5], codes[6], 1, 0)
        tx.commit()
        # 钉住的旧版本读到的仍是旧事实，即便窗口覆盖被修改的单元
        self.assertEqual(
            cells_of(snap0.window(0, 8, 0, 8)), expected_window(full0, 0, 8, 0, 8)
        )
        with ds.read() as snap_new:
            full_now = expand(list(live.values()) + [("r10", "b3", 5, 6, 1, 0)])
            self.assertEqual(
                cells_of(snap_new.window(0, 8, 0, 8)), expected_window(full_now, 0, 8, 0, 8)
            )
        # 盘上只保留当前版本与被钉住的版本
        ds.gc()
        kept = sorted(os.listdir(os.path.join(self.sub(), "versions")))
        self.assertEqual(kept, sorted([f"v_{v0:08d}", f"v_{ds.version_seq:08d}"]))
        snap0.close()
        ds.gc()
        kept = sorted(os.listdir(os.path.join(self.sub(), "versions")))
        self.assertEqual(kept, [f"v_{ds.version_seq:08d}"])
        ds.close()

    def test_concurrent_readers_writer(self):
        rng = random.Random(7)
        n = 300
        codes = make_codes(n)
        recs = gen_records(rng, n, 1500, max_ev=2)
        ds = Dataset.create(self.sub(), codes)
        live = {}
        tx = ds.correction()
        for r in recs:
            tx.add(r[0], r[1], codes[r[2]], codes[r[3]], r[4], r[5])
            live[r[0]] = r
        tx.commit()
        matrices = {ds.version_seq: expand(live.values())}
        mlock = threading.Lock()
        errors = []

        def reader(seed):
            try:
                r = random.Random(seed)
                for _ in range(80):
                    snap = ds.read()
                    seq = snap.version_seq
                    r0 = r.randrange(n)
                    r1 = min(n, r0 + r.randrange(0, 25))
                    c0 = r.randrange(n)
                    c1 = min(n, c0 + r.randrange(0, 25))
                    got = cells_of(snap.window(r0, r1, c0, c1))
                    with mlock:
                        m = matrices[seq]
                    if got != expected_window(m, r0, r1, c0, c1):
                        errors.append(f"seq={seq} 窗口读出混搭")
                    snap.close()
            except Exception as ex:  # noqa: BLE001
                errors.append(repr(ex))

        def writer():
            try:
                r = random.Random(99)
                rid_gen = [10**6]
                for _ in range(15):
                    tx = ds.correction()
                    live_rids = list(live)
                    for rid in r.sample(live_rids, min(30, len(live_rids))):
                        tx.revoke(rid)
                        live.pop(rid)
                    present = {}
                    for rec in live.values():
                        a, b = rec[2], rec[3]
                        present.setdefault((min(a, b), max(a, b)), rec)
                    for _ in range(20):
                        i, j = sorted((r.randrange(n), r.randrange(n)))
                        rid = f"x{rid_gen[0]}"
                        rid_gen[0] += 1
                        cur = present.get((i, j))
                        if cur is None:
                            d = 0 if i == j else r.randint(-1000, 1000)
                            e = r.choice([0, 1, 0.5])
                        else:
                            d = cur[4] if cur[2] <= cur[3] else -cur[4]
                            e = float(cur[5])
                        rec = (rid, "bw", i, j, d, e)
                        tx.add(rid, "bw", codes[i], codes[j], d, e)
                        live[rid] = rec
                        present[(i, j)] = rec
                    new_matrix = expand(live.values())
                    with mlock:
                        matrices[ds.version_seq + 1] = new_matrix
                    tx.commit()
            except Exception as ex:  # noqa: BLE001
                errors.append(repr(ex))

        threads = [threading.Thread(target=reader, args=(s,)) for s in (1, 2, 3)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        ds.close()


# ---------------------------------------------------------------------- 随机全矩阵核对


class RandomizedCrosscheckTest(TempDirCase):
    def test_random_windows_against_full_matrix(self):
        rng = random.Random(11)
        n = 60
        codes = make_codes(n)
        recs = gen_records(rng, n, 300, max_ev=3)
        full = expand(recs)
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        with ds.read() as snap:
            for _ in range(400):
                r0, r1 = sorted((rng.randrange(n + 1), rng.randrange(n + 1)))
                c0, c1 = sorted((rng.randrange(n + 1), rng.randrange(n + 1)))
                res = snap.window(r0, r1, c0, c1)
                self.assertEqual(
                    cells_of(res), expected_window(full, r0, r1, c0, c1),
                    msg=f"window [{r0},{r1})x[{c0},{c1})",
                )
                idx = {}
                for c in res.cells:
                    for ev in c.evidence:
                        idx.setdefault(ev.record_id, ev)
                self.assertEqual(res.evidence_index, idx)
            # 单点双向核对
            for _ in range(300):
                a, b = rng.randrange(n), rng.randrange(n)
                cell = snap.pair(a, b)
                expc = full.get((a, b))
                if expc is None:
                    self.assertIsNone(cell)
                else:
                    d, e, o, evs = expc
                    self.assertEqual((cell.d, cell.e, cell.orientation, norm_evs(cell.evidence)),
                                     (d, e, o, evs))
        ds.close()


# ---------------------------------------------------------------------- 持久化


class PersistenceTest(TempDirCase):
    def test_reopen_and_continue(self):
        rng = random.Random(31)
        n = 50
        codes = make_codes(n)
        recs = gen_records(rng, n, 200, max_ev=2)
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        live = {r[0]: r for r in recs}
        tx = ds.correction()
        tx.revoke(recs[0][0])
        live.pop(recs[0][0])
        tx.add("z1", "b9", codes[10], codes[20], -33, 0.5)
        live["z1"] = ("z1", "b9", 10, 20, -33, 0.5)
        tx.commit()
        seq = ds.version_seq
        ds.close()
        full = expand(live.values())
        ds2 = Dataset.open(self.sub())
        self.assertEqual(ds2.version_seq, seq)
        with ds2.read() as snap:
            for _ in range(60):
                r0, r1 = sorted((rng.randrange(n + 1), rng.randrange(n + 1)))
                c0, c1 = sorted((rng.randrange(n + 1), rng.randrange(n + 1)))
                self.assertEqual(
                    cells_of(snap.window(r0, r1, c0, c1)),
                    expected_window(full, r0, r1, c0, c1),
                )
        # 重开后续订正
        tx = ds2.correction()
        tx.add("z2", "b9", codes[1], codes[1], 0, 0)
        tx.commit()
        live["z2"] = ("z2", "b9", 1, 1, 0, 0)
        ds2.close()
        ds3 = Dataset.open(self.sub())
        full = expand(live.values())
        with ds3.read() as snap:
            self.assertEqual(cells_of(snap.window(0, n, 0, n)), expected_window(full, 0, n, 0, n))
        ds3.close()


# ---------------------------------------------------------------------- 两轴选择性证明


class SelectivityTest(TempDirCase):
    def test_narrow_window_uses_both_axes(self):
        """大量单轴命中却都不在目标矩形：k-d 树必须同时利用两轴剪枝。"""
        n = 400_000
        r0, r1 = 150_000, 200_000
        codes = make_codes(n)
        recs = []
        # A 组：行在窗口行区间、列远在窗外（仅行轴命中），5 万条
        for t in range(50_000):
            recs.append((f"a{t}", "bA", r0 + t, 250_000 + t, t % 1000 - 500, 1))
        # B 组：列在窗口列区间、行远在窗外（仅列轴命中），5 万条
        for t in range(50_000):
            recs.append((f"b{t}", "bB", 50_000 + t, r0 + t, 7, 0.5))
        # C 组：远处对角噪声，5 万条
        for t in range(50_000):
            recs.append((f"c{t}", "bC", t, t, 0, 2))
        # D 组：远处普通对噪声，5 万条
        for t in range(50_000):
            recs.append((f"d{t}", "bD", 300_000 + t, 300_001 + t, 3, 1))
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        self.assertEqual(ds.fact_count, 200_000)
        with ds.read() as snap:
            res = snap.window(r0, r1, r0, r1)
            self.assertEqual(len(res.cells), 0)  # 矩形内确实没有事实
            visited = res.stats["nodes_visited"]
            single_axis_hits = 100_000  # A 组（行命中）+ B 组（列命中）
            # 若只利用单轴，至少要扫过 5 万条单轴命中；两轴剪枝应低几个数量级
            self.assertLess(visited, 5_000)
            self.assertLess(visited, single_axis_hits // 20)
            # 抽查有内容的窄窗仍然正确
            res = snap.window(r0, r0 + 10, 250_000, 250_010)
            self.assertEqual(len(res.cells), 10)
            self.assertEqual(
                [(c.row, c.col) for c in res.cells],
                [(r0 + t, 250_000 + t) for t in range(10)],
            )
            res = snap.window(250_000, 250_005, r0, r0 + 5)  # 反向读同一批事实
            self.assertEqual(len(res.cells), 5)
            for c in res.cells:
                t = c.col - r0  # A 组第 t 条：正向记录 d = t%1000-500
                self.assertEqual(c.orientation, REVERSE)
                self.assertEqual(c.d, -(t % 1000 - 500))
                self.assertEqual([ev.record_id for ev in c.evidence], [f"a{t}"])
        ds.close()


# ---------------------------------------------------------------------- 热点窄窗


class HotNarrowWindowTest(TempDirCase):
    def test_many_disjoint_records_hot_window(self):
        rng = random.Random(3)
        n = 300_000
        codes = make_codes(n)
        n_facts = 150_000
        pairs = set()
        while len(pairs) < n_facts:
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        recs = []
        for k, (i, j) in enumerate(sorted(pairs)):
            recs.append((f"r{k}", "hot", i, j, rng.randint(-10**9, 10**9), rng.choice([0, 1, 0.25])))
        full = expand(recs)
        ds = Dataset.create(self.sub(), codes, records=to_codes(recs, codes))
        with ds.read() as snap:
            # 热点窄窗反复读：结果稳定且访问节点数远小于全仓事实数
            hot = (123_456, 123_464, 234_567, 234_575)
            exp = expected_window(full, *hot)
            for _ in range(200):
                res = snap.window(*hot)
                self.assertEqual(cells_of(res), exp)
                self.assertLess(res.stats["nodes_visited"], 10_000)
            # 随机窄窗与全矩阵核对
            for _ in range(150):
                a = rng.randrange(n - 8)
                b = rng.randrange(n - 8)
                w = (a, a + 8, b, b + 8)
                res = snap.window(*w)
                self.assertEqual(cells_of(res), expected_window(full, *w))
                self.assertLess(res.stats["nodes_visited"], 10_000)
                self.assertLess(res.stats["nodes_visited"], n_facts // 10)
        ds.close()


if __name__ == "__main__":
    unittest.main()
