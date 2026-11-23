"""补采工单：在只读快照上按覆盖要求查出仍缺独立有效批次的测点对。

语义
----
- 每条要求 ``(a0, a1, b0, b1, k)``：从半开区间 ``[a0,a1)`` 与 ``[b0,b1)``
  各取一个测点组成的无序对 ``{i,j}``，至少要有 ``k`` 个不同的有效批次。
  两段允许重叠；自对 ``{i,i}`` 不是待检对。
- 两段没有先后：交换两段覆盖同一批对，不重复派单。
- 一对命中多条要求时取其中最大的 ``k``，不相加；只有被至少一条要求覆盖的
  对才可能进入工单。
- 批次来自快照中的现有证据：同一测点对同一批次的多条证据只算一次，不区分
  原始方向；没有证据即零批。已有读数 ``d/e`` 的大小不影响是否达标。
- 缺口按 ``(row, col)`` 字典序排列（row < col）；``index`` 从 0 起，
  ``index >= total_gaps`` 返回空页。

实现
----
宽范围要求隐含的应测对可以达到数十亿，因此建单时**不展开任何隐含测点对**：

1. 把所有要求端点压缩成“段”。段内各点对所有要求属于哪一侧完全相同，
   于是任意一对被覆盖的测点必落在同一对“行段/列段”里，且门槛恒定。
2. 被覆盖对的总数由段对的 ``len_s * len_t`` 之和算出（O(段数²)，与隐藏
   对数无关）；再减去“证据批次已达标”的**实际事实**（O(事实数)），得到
   精确缺口总数。未达标的实际事实单独留作候选。
3. 被覆盖对按行主序看成矩形并集（另有行段内部的三角块）。分页用前缀和
   顺序统计在 O(log 段数 + log 页长) 内直接定位缺口，只为命中页交付条目；
   不生成全体应测对或全体缺口，也不为跳页逐个遍历前面的缺口。

建单后工单冻结当时的规则、快照版本与上述准备结果；后续订正不会移动旧工单
的页边界。快照关闭后不能再取页。
"""

from __future__ import annotations

import bisect

from .errors import DatasetError, ValidationError
from .model import GapEntry

MAX_REQUIREMENTS = 128
MIN_PAGE_SIZE = 1
MAX_PAGE_SIZE = 1000


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _normalize_requirements(requirements, n):
    """校验并返回 [(a0,a1,b0,b1,k), ...]。"""
    if isinstance(requirements, (str, bytes)):
        raise ValidationError("要求集必须是要求的序列，而不是字符串")
    try:
        items = list(requirements)
    except TypeError:
        raise ValidationError("要求集必须可迭代") from None
    if len(items) > MAX_REQUIREMENTS:
        raise ValidationError(
            f"每份工单至多 {MAX_REQUIREMENTS} 条要求，收到 {len(items)} 条"
        )
    out = []
    for pos, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 5:
            raise ValidationError(f"第 {pos} 条要求必须形如 (a0, a1, b0, b1, k)")
        a0, a1, b0, b1, k = item
        for value, tag in ((a0, "a0"), (a1, "a1"), (b0, "b0"), (b1, "b1")):
            if not _is_int(value):
                raise ValidationError(f"第 {pos} 条要求的区间端点 {tag} 必须为整数")
        for lo, hi, tag in ((a0, a1, "第一段"), (b0, b1, "第二段")):
            if not (0 <= lo < hi <= n):
                raise ValidationError(
                    f"第 {pos} 条要求的{tag}必须为 [0,{n}) 内的非空半开区间，"
                    f"收到 [{lo},{hi})"
                )
        if not _is_int(k):
            raise ValidationError(f"第 {pos} 条要求的批次门槛必须为整数")
        if k < 1:
            raise ValidationError(f"第 {pos} 条要求的批次门槛必须为正整数")
        out.append((a0, a1, b0, b1, k))
    return out


def _max_threshold(mask, thresholds):
    best = 0
    while mask:
        low = mask & -mask
        k = thresholds[low.bit_length() - 1]
        if k > best:
            best = k
        mask ^= low
    return best


class _RowSegment:
    """一个行段内被覆盖列的布局。

    ``runs`` 为行段之外的列段 ``(col_start, col_len)``，按列升序；
    ``tri`` 表示本行段内部还有三角块：第 q 行（0 起）的列为
    ``start+q+1 .. start+len-1``。
    """

    __slots__ = ("start", "length", "base", "run_starts", "run_lens",
                 "run_pref", "tri", "total")

    def __init__(self, start, length, runs, tri):
        self.start = start
        self.length = length
        self.run_starts = [r[0] for r in runs]
        self.run_lens = [r[1] for r in runs]
        pref = [0]
        for ln in self.run_lens:
            pref.append(pref[-1] + ln)
        self.run_pref = pref
        self.base = pref[-1]
        self.tri = tri
        m = length
        self.total = m * self.base + (m * (m - 1) // 2 if tri else 0)

    def row_prefix(self, q):
        """本段前 q 行的被覆盖对数。"""
        total = q * self.base
        if self.tri:
            total += q * (self.length - 1) - q * (q - 1) // 2
        return total


class GapPage:
    """一页缺口；由 0..1000 个 GapEntry 组成。"""

    __slots__ = ("index", "size", "version_seq", "total_gaps", "entries")

    def __init__(self, index, size, version_seq, total_gaps, entries):
        self.index = index
        self.size = size
        self.version_seq = version_seq
        self.total_gaps = total_gaps
        self.entries = tuple(entries)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def next_index(self):
        return self.index + len(self.entries)

    def __repr__(self):
        return (
            f"GapPage(index={self.index}, size={self.size}, "
            f"entries={len(self.entries)}, total={self.total_gaps}, "
            f"version={self.version_seq})"
        )


class WorkOrder:
    """一份冻结的补采工单：给出快照版本、精确缺口总数，并可逐页取缺口。"""

    def __init__(self, snapshot, requirements):
        reqs = _normalize_requirements(requirements, snapshot.n_stations)
        self._snap = snapshot
        self._version_seq = snapshot.version_seq
        self._requirements = tuple(reqs)
        self._existing = {}
        self._satisfied = []
        self._blocks = {}
        self._self_k = {}
        self._satisfied = []
        self._active = []
        self._prefix = [0]
        self._total = 0
        self._rect_total = 0
        if reqs:
            self._build(reqs, snapshot)

    # ------------------------------------------------------------------ 建单

    def _build(self, reqs, snapshot):
        endpoints = sorted({x for r in reqs for x in r[:4]})
        starts = endpoints[:-1]
        lengths = [endpoints[i + 1] - endpoints[i] for i in range(len(starts))]
        n_seg = len(starts)

        # 每段在每条要求的哪一侧（段被所有要求端点切开，故至多落在一侧）。
        masks_a = [0] * n_seg
        masks_b = [0] * n_seg
        thresholds = [r[4] for r in reqs]
        for bit, (a0, a1, b0, b1, _k) in enumerate(reqs):
            flag = 1 << bit
            for s in range(bisect.bisect_left(starts, a0),
                           bisect.bisect_left(starts, a1)):
                masks_a[s] |= flag
            for s in range(bisect.bisect_left(starts, b0),
                           bisect.bisect_left(starts, b1)):
                masks_b[s] |= flag

        # 段对 -> 命中要求中的最大门槛；两段交换是同一批对，只存 s<t。
        blocks = {}
        block_ts = [[] for _ in range(n_seg)]
        for s in range(n_seg):
            ma = masks_a[s]
            mb = masks_b[s]
            for t in range(s + 1, n_seg):
                m = (ma & masks_b[t]) | (mb & masks_a[t])
                if m:
                    blocks[(s, t)] = _max_threshold(m, thresholds)
                    block_ts[s].append(t)
        # 同一段内部：要求的两段都覆盖本段时，段内所有 i<j 都被覆盖。
        self_k = {}
        for s in range(n_seg):
            m = masks_a[s] & masks_b[s]
            if m:
                self_k[s] = _max_threshold(m, thresholds)

        # 扫描实际事实：够批次的从应测对中扣除，不够批次的留作候选。
        v = snapshot._v
        I = v.I
        J = v.J
        ev_off = v.ev_off
        ev_batch = v.ev_batch
        # 已达标事实单独留作“从应测对中扣除”的稀疏集合（大小受实际事实数
        # 限制，不随隐藏对数增长）。
        satisfied = []
        existing = {}
        last_end = endpoints[-1]
        first_start = starts[0]

        def seg_of(x):
            if x < first_start or x >= last_end:
                return -1
            return bisect.bisect_right(starts, x) - 1

        for k in range(len(I)):
            i = I[k]
            j = J[k]
            if i == j:
                continue  # 自对不是待检对
            s = seg_of(i)
            t = seg_of(j)
            if s < 0 or t < 0:
                continue
            if s == t:
                need = self_k.get(s)
            else:
                need = blocks.get((s, t))
            if need is None:
                continue
            lo = ev_off[k]
            hi = ev_off[k + 1]
            fresh = len({ev_batch[x] for x in range(lo, hi)})
            if fresh >= need:
                satisfied.append((i, j))
            else:
                existing[(i, j)] = fresh

        # 行主序布局：每个有被覆盖对的行段，列出列段与三角块。
        active = []
        prefix = [0]
        for s in range(n_seg):
            tri = s in self_k
            if not block_ts[s] and not tri:
                continue
            runs = [(starts[t], lengths[t]) for t in block_ts[s]]
            seg = _RowSegment(starts[s], lengths[s], runs, tri)
            if seg.total <= 0:
                continue
            active.append(seg)
            prefix.append(prefix[-1] + seg.total)

        self._blocks = blocks
        self._self_k = self_k
        self._starts = starts
        self._existing = existing
        self._satisfied = sorted(satisfied)
        self._active = active
        self._prefix = prefix
        self._rect_total = prefix[-1]
        # 缺口 = 应测对 - 已达标的实际事实；两者都只依赖实际事实与规则。
        self._total = self._rect_total - len(self._satisfied)

    # ------------------------------------------------------------------ 属性

    @property
    def version_seq(self):
        return self._version_seq

    @property
    def requirements(self):
        return self._requirements

    @property
    def total_gaps(self):
        return self._total

    # ------------------------------------------------------------------ 取页

    def page(self, index, size):
        """返回从缺口序号 ``index`` 起、至多 ``size`` 条的一页。

        ``size`` 为 1..1000 的整数；``index >= total_gaps`` 返回空页。
        """
        self._ensure_live()
        if not _is_int(index):
            raise ValidationError("缺口序号必须为整数")
        if index < 0:
            raise ValidationError("缺口序号不能为负")
        if not _is_int(size):
            raise ValidationError("页长必须为整数")
        if not (MIN_PAGE_SIZE <= size <= MAX_PAGE_SIZE):
            raise ValidationError(
                f"页长必须在 {MIN_PAGE_SIZE}..{MAX_PAGE_SIZE} 之间，收到 {size}"
            )
        if index >= self._total:
            return GapPage(index, size, self._version_seq, self._total, ())
        end = min(index + size, self._total)
        entries = [self._entry_at(pos) for pos in range(index, end)]
        return GapPage(index, size, self._version_seq, self._total, entries)

    def _entry_at(self, pos):
        """第 pos 个缺口（0 起），顺序统计直接定位，不遍历前面的缺口。

        用有序的已达标事实集合做前缀计数；二分次数是 O(log 应测对数)，
        与缺口总数和前面的缺口个数无关。
        """
        satisfied = self._satisfied
        want = pos + 1
        lo = 0
        hi = self._rect_total - 1
        ans = hi
        while lo <= hi:
            mid = (lo + hi) >> 1
            row, col = self._key_at(mid)
            if (mid + 1) - bisect.bisect_right(satisfied, (row, col)) >= want:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        row, col = self._key_at(ans)
        required = self._required_of(row, col)
        stations = self._snap._v.stations
        return GapEntry(
            row,
            col,
            required,
            self._existing.get((row, col), 0),
            stations[row],
            stations[col],
        )

    def _seg_of(self, pos):
        return bisect.bisect_right(self._starts, pos) - 1

    def _required_of(self, row, col):
        s = self._seg_of(row)
        t = self._seg_of(col)
        if s == t:
            return self._self_k[s]
        return self._blocks[(s, t)]

    def _key_at(self, rank):
        """被覆盖对按 (row, col) 排序后的第 rank 个（0 起）。"""
        prefix = self._prefix
        ai = bisect.bisect_right(prefix, rank) - 1
        seg = self._active[ai]
        rem = rank - prefix[ai]
        # 本段前 q 行的数量 q*base + q*(m-1) - q*(q-1)/2 单调不减
        lo, hi = 0, seg.length - 1
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if seg.row_prefix(mid) <= rem:
                lo = mid
            else:
                hi = mid - 1
        q = lo
        rem -= seg.row_prefix(q)
        row = seg.start + q
        if seg.tri:
            tri_len = seg.length - 1 - q
            if rem < tri_len:
                return row, row + 1 + rem
            rem -= tri_len
        rj = bisect.bisect_right(seg.run_pref, rem) - 1
        return row, seg.run_starts[rj] + (rem - seg.run_pref[rj])

    def _ensure_live(self):
        if self._snap.closed:
            raise DatasetError("读快照已关闭，不能再从该工单取页")

    def __repr__(self):
        return (
            f"WorkOrder(version={self._version_seq}, "
            f"requirements={len(self._requirements)}, total_gaps={self._total})"
        )


def build_work_order(snapshot, requirements):
    """在打开的读快照上建立工单（Snapshot.backfill 的实现）。"""
    if snapshot.closed:
        raise DatasetError("读快照已关闭，不能建立工单")
    return WorkOrder(snapshot, requirements)
