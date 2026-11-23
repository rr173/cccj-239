"""单个不可变版本：存储 + 二维范围索引。

盘上布局（每个版本一个目录）：
    meta.json      版本号、事实数、证据数
    I.bin/J.bin    规范点坐标（按 (i,j) 字典序，i<=j），uint32
    D.bin/E.bin    折算后的偏差 int64 / 误差 float64
    ev_off.bin     每个事实的证据区间偏移（n+1 个 uint64）
    ev_rec/ev_batch/ev_dir/ev_rd/ev_re  证据条目（记录表索引、批次表索引、
                   原始方向标志、原始 d、原始 e）
    records.json   记录身份表（索引 = ev_rec 值）
    batches.json   批次表
    stations.txt   本版本的测点代码顺序（行号 = 排序位置）；仅在与创建
                   顺序不同的版本目录中存在，缺失时回退到根 stations.txt

每个无序测点对只存一份规范方向的事实；反向读取在查询时折算，
盘上不存在正反两份事实。坐标（排序位置）按本版本自己的测点顺序解释：
换版（重排）后的版本与旧版本各自使用各自的坐标。
"""

from __future__ import annotations

import array
import json
import os
import sys
from bisect import bisect_right

from .errors import DatasetError
from .model import EvidenceRef

# 数组名 -> array typecode
_ARRAY_DEFS = {
    "I": "I",
    "J": "I",
    "D": "q",
    "E": "d",
    "ev_off": "Q",
    "ev_rec": "I",
    "ev_batch": "I",
    "ev_dir": "B",
    "ev_rd": "q",
    "ev_re": "d",
}

_MIN_ITEMSIZE = {"I": 4, "Q": 8, "q": 8, "d": 8, "B": 1}

FORMAT_VERSION = 1


def _check_platform():
    for code, need in _MIN_ITEMSIZE.items():
        if array.array(code).itemsize < need:
            raise RuntimeError(f"平台 array typecode {code!r} 位宽不足")


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Version:
    """不可变版本：测点顺序 + 规范事实表 + 证据表 + k-d 树二维索引。"""

    def __init__(self, seq, stations, facts):
        """stations: 本版本的测点代码顺序（排序位置 0..N-1）。
        facts: dict[(i,j)] -> (d, e, evs)，i<=j，坐标按本版本顺序；
        evs 为 (record_id, batch, src, dst, raw_d, raw_e) 元组。"""
        _check_platform()
        self.seq = seq
        self.stations = list(stations)
        self.n_stations = len(self.stations)
        self.facts = facts
        self._build_arrays()
        self._build_tree()

    # ------------------------------------------------------------------ 构建

    def _build_arrays(self):
        pairs = sorted(self.facts)
        I = array.array("I")
        J = array.array("I")
        D = array.array("q")
        E = array.array("d")
        ev_off = array.array("Q", [0])
        ev_rec = array.array("I")
        ev_batch = array.array("I")
        ev_dir = array.array("B")
        ev_rd = array.array("q")
        ev_re = array.array("d")
        records = []
        batches = []
        rec_pos = {}
        bat_pos = {}
        rec_index = {}
        for (i, j) in pairs:
            d, e, evs = self.facts[(i, j)]
            I.append(i)
            J.append(j)
            D.append(d)
            E.append(e)
            for ev in evs:
                rid, batch, pa, pb, rd, re_ = ev
                ri = rec_pos.get(rid)
                if ri is None:
                    ri = len(records)
                    rec_pos[rid] = ri
                    records.append(rid)
                bi = bat_pos.get(batch)
                if bi is None:
                    bi = len(batches)
                    bat_pos[batch] = bi
                    batches.append(batch)
                rec_index[rid] = len(ev_rec)
                ev_rec.append(ri)
                ev_batch.append(bi)
                ev_dir.append(0 if pa <= pb else 1)
                ev_rd.append(rd)
                ev_re.append(re_)
            ev_off.append(len(ev_rec))
        self.I = I
        self.J = J
        self.D = D
        self.E = E
        self.ev_off = ev_off
        self.ev_rec = ev_rec
        self.ev_batch = ev_batch
        self.ev_dir = ev_dir
        self.ev_rd = ev_rd
        self.ev_re = ev_re
        self.records = records
        self.batches = batches
        self.rec_index = rec_index
        self.pair_index = {p: k for k, p in enumerate(pairs)}

    def _build_tree(self):
        """对规范点建平衡 k-d 树（交替按 i/j 切分），数组化存储。"""
        n = len(self.I)
        self._t_fact = array.array("I")
        self._t_left = array.array("i")
        self._t_right = array.array("i")
        self._root = -1
        if n == 0:
            return
        I, J = self.I, self.J
        # 两个轴的全序排名（(i,j) 唯一，无并列）
        rank_i = array.array("I", [0]) * n
        rank_j = array.array("I", [0]) * n
        for pos, k in enumerate(sorted(range(n), key=lambda k: (I[k], J[k]))):
            rank_i[k] = pos
        for pos, k in enumerate(sorted(range(n), key=lambda k: (J[k], I[k]))):
            rank_j[k] = pos
        key_funcs = (rank_i.__getitem__, rank_j.__getitem__)
        tf = self._t_fact
        tl = self._t_left
        tr = self._t_right

        def build(ids, depth):
            m = len(ids)
            if m == 0:
                return -1
            ids.sort(key=key_funcs[depth & 1])
            mid = m >> 1
            node = len(tf)
            tf.append(ids[mid])
            tl.append(-1)
            tr.append(-1)
            if mid:
                tl[node] = build(ids[:mid], depth + 1)
            if mid + 1 < m:
                tr[node] = build(ids[mid + 1 :], depth + 1)
            return node

        self._root = build(list(range(n)), 0)

    # ------------------------------------------------------------------ 查询

    def rect_query(self, qi0, qi1, qj0, qj1, out, stats):
        """半开矩形 [qi0,qi1) x [qj0,qj1) 内的规范点事实下标追加到 out。"""
        if self._root < 0:
            return
        tf = self._t_fact
        tl = self._t_left
        tr = self._t_right
        I = self.I
        J = self.J
        N = self.n_stations
        stack = [(self._root, 0, N, 0, N, 0)]
        push = stack.append
        pop = stack.pop
        while stack:
            node, i0, i1, j0, j1, depth = pop()
            if qi1 <= i0 or i1 <= qi0 or qj1 <= j0 or j1 <= qj0:
                continue
            stats["nodes_visited"] += 1
            k = tf[node]
            pi = I[k]
            pj = J[k]
            if qi0 <= pi < qi1 and qj0 <= pj < qj1:
                out.append(k)
            ln = tl[node]
            rn = tr[node]
            if depth & 1:
                if ln != -1:
                    push((ln, i0, i1, j0, min(j1, pj + 1), depth + 1))
                if rn != -1:
                    push((rn, i0, i1, max(j0, pj), j1, depth + 1))
            else:
                if ln != -1:
                    push((ln, i0, min(i1, pi + 1), j0, j1, depth + 1))
                if rn != -1:
                    push((rn, max(i0, pi), i1, j0, j1, depth + 1))

    def evidence_of(self, k):
        """事实 k 的支持证据（按 record_id 排序）。"""
        out = []
        i = self.I[k]
        j = self.J[k]
        for t in range(self.ev_off[k], self.ev_off[k + 1]):
            rid = self.records[self.ev_rec[t]]
            bat = self.batches[self.ev_batch[t]]
            if self.ev_dir[t] == 0:
                pa, pb = i, j
            else:
                pa, pb = j, i
            out.append(EvidenceRef(rid, bat, pa, pb, self.ev_rd[t], self.ev_re[t]))
        out.sort(key=lambda ev: ev.record_id)
        return tuple(out)

    def fact_of_entry(self, t):
        """证据条目 t 所属事实下标。"""
        return bisect_right(self.ev_off, t) - 1

    # ------------------------------------------------------------------ 落盘

    def save(self, dirpath, write_stations=True):
        os.makedirs(dirpath, exist_ok=False)
        arrays = {
            "I": self.I,
            "J": self.J,
            "D": self.D,
            "E": self.E,
            "ev_off": self.ev_off,
            "ev_rec": self.ev_rec,
            "ev_batch": self.ev_batch,
            "ev_dir": self.ev_dir,
            "ev_rd": self.ev_rd,
            "ev_re": self.ev_re,
        }
        big_endian = sys.byteorder == "big"
        for name, arr in arrays.items():
            a = arr
            if big_endian:
                a = array.array(arr.typecode, arr)
                a.byteswap()
            with open(os.path.join(dirpath, name + ".bin"), "wb") as f:
                a.tofile(f)
                f.flush()
                os.fsync(f.fileno())
        for name, table in (("records", self.records), ("batches", self.batches)):
            with open(os.path.join(dirpath, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(table, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
        if write_stations:
            # 本版本顺序与创建顺序不同：把该版测点顺序随版本目录一起落盘
            with open(os.path.join(dirpath, "stations.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(self.stations))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
        meta = {
            "format": FORMAT_VERSION,
            "seq": self.seq,
            "facts": len(self.I),
            "evidence": len(self.ev_rec),
        }
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(dirpath)

    @classmethod
    def load(cls, dirpath, root_stations):
        """加载版本目录。root_stations 为根 stations.txt 的创建顺序：
        版本目录没有自己的 stations.txt 时（旧格式或与创建顺序相同）
        回退到创建顺序。"""
        with open(os.path.join(dirpath, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") != FORMAT_VERSION:
            raise RuntimeError(f"不支持的格式版本: {meta.get('format')}")
        stations_file = os.path.join(dirpath, "stations.txt")
        if os.path.isfile(stations_file):
            with open(stations_file, encoding="utf-8") as f:
                stations = f.read().splitlines()
            if len(stations) != len(root_stations) or set(stations) != set(root_stations):
                raise DatasetError(f"版本目录的测点顺序文件损坏: {stations_file}")
        else:
            stations = list(root_stations)
        _check_platform()
        big_endian = sys.byteorder == "big"
        arrays = {}
        for name, code in _ARRAY_DEFS.items():
            arr = array.array(code)
            path = os.path.join(dirpath, name + ".bin")
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                arr.fromfile(f, size // arr.itemsize)
            if big_endian:
                arr.byteswap()
            arrays[name] = arr
        with open(os.path.join(dirpath, "records.json"), encoding="utf-8") as f:
            records = json.load(f)
        with open(os.path.join(dirpath, "batches.json"), encoding="utf-8") as f:
            batches = json.load(f)

        v = cls.__new__(cls)
        v.seq = meta["seq"]
        v.stations = stations
        v.n_stations = len(stations)
        v.I = arrays["I"]
        v.J = arrays["J"]
        v.D = arrays["D"]
        v.E = arrays["E"]
        v.ev_off = arrays["ev_off"]
        v.ev_rec = arrays["ev_rec"]
        v.ev_batch = arrays["ev_batch"]
        v.ev_dir = arrays["ev_dir"]
        v.ev_rd = arrays["ev_rd"]
        v.ev_re = arrays["ev_re"]
        v.records = records
        v.batches = batches
        # 重建索引与事实字典
        rec_index = {}
        facts = {}
        I, J, D, E = v.I, v.J, v.D, v.E
        ev_off = v.ev_off
        for k in range(len(I)):
            i = I[k]
            j = J[k]
            evs = []
            for t in range(ev_off[k], ev_off[k + 1]):
                rid = records[v.ev_rec[t]]
                bat = batches[v.ev_batch[t]]
                if v.ev_dir[t] == 0:
                    pa, pb = i, j
                else:
                    pa, pb = j, i
                rec_index[rid] = t
                evs.append((rid, bat, pa, pb, v.ev_rd[t], v.ev_re[t]))
            facts[(i, j)] = (D[k], E[k], tuple(evs))
        v.rec_index = rec_index
        v.facts = facts
        v.pair_index = {p: k for k, p in enumerate(sorted(facts))}
        v._build_tree()
        return v
