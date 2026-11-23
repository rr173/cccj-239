"""对外交付的数据模型。

方向约定：
- 事实在盘上只按规范方向（排序位置小 -> 大）存一份；
- 读出的单元带 orientation 标记：
  - "forward"：单元 (row, col) 与规范方向一致（row < col）；
  - "reverse"：单元是规范方向的反向（row > col），d 已折算为 -d；
  - "self"：对角单元（row == col），只允许 d == 0。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

FORWARD = "forward"
REVERSE = "reverse"
SELF = "self"

ORIENTATIONS = (FORWARD, REVERSE, SELF)


@dataclass(frozen=True)
class EvidenceRef:
    """一条支持证据：原始记录身份 + 有效批次 + 原始方向与原值。

    src/dst 为原始记录“从 src 到 dst”的测点排序位置；
    raw_d/raw_e 为原始记录值（未折算）。
    """

    record_id: str
    batch: str
    src: int
    dst: int
    raw_d: int
    raw_e: float


@dataclass(frozen=True)
class Cell:
    """窗口内交付的一个有向单元。"""

    row: int
    col: int
    d: int
    e: float
    orientation: str
    evidence: Tuple[EvidenceRef, ...]
    row_code: str = ""
    col_code: str = ""


@dataclass(frozen=True)
class GapEntry:
    """补采工单中的一个缺口。

    row/col 为排序位置（row < col）；required_batches 为该测点对命中多条
    覆盖要求后取到的最大批次门槛；existing_batches 为现有证据中不同有效
    批次的个数（同一批次的多条证据只算一次，不分原始方向）。
    """

    row: int
    col: int
    required_batches: int
    existing_batches: int
    row_code: str = ""
    col_code: str = ""


@dataclass(frozen=True)
class CatchupChange:
    """一条记录在一次追补跨度内的净变更。

    old/new 为该记录 ID 在游标端/目标端的原始证据内容（EvidenceRef：
    批次、原始方向的两个测点排序位置、原始偏差与误差，未折算）；
    不存在的一端为 None。误差按数值比较（如 -0.0 与 0.0 视为相同）。
    """

    record_id: str
    old: Optional[EvidenceRef]
    new: Optional[EvidenceRef]


class CatchupResult:
    """一次追补的结果。

    changes 按记录 ID 的字符串顺序交付，每个有差异的 ID 只出现一次；
    to_seq 为目标快照版本号，可作为下次追补的游标；
    stats 暴露本次追补处理的记录规模（用于验证不做整仓扫描）。
    """

    __slots__ = ("from_seq", "to_seq", "changes", "stats")

    def __init__(self, from_seq, to_seq, changes, stats):
        self.from_seq = from_seq
        self.to_seq = to_seq
        self.changes: Tuple[CatchupChange, ...] = tuple(changes)
        self.stats = stats

    def __len__(self):
        return len(self.changes)

    def __iter__(self):
        return iter(self.changes)

    def __repr__(self):
        return (
            f"CatchupResult(from_seq={self.from_seq}, to_seq={self.to_seq}, "
            f"changes={len(self.changes)})"
        )


class WindowResult:
    """一次矩形窗口读取的结果。

    cells 按 (row, col) 排序；evidence_index 为窗内证据索引
    （record_id -> EvidenceRef），反向单元与正向单元指向同一批证据。
    stats 含本次查询的节点访问计数，用于验证两轴选择性。
    """

    __slots__ = ("r0", "r1", "c0", "c1", "version", "cells", "evidence_index", "stats")

    def __init__(self, r0, r1, c0, c1, version, cells, evidence_index, stats):
        self.r0 = r0
        self.r1 = r1
        self.c0 = c0
        self.c1 = c1
        self.version = version
        self.cells = cells
        self.evidence_index: Dict[str, EvidenceRef] = evidence_index
        self.stats = stats

    def __len__(self):
        return len(self.cells)

    def __iter__(self):
        return iter(self.cells)

    def __repr__(self):
        return (
            f"WindowResult(window=[{self.r0},{self.r1})x[{self.c0},{self.c1}), "
            f"version={self.version}, cells={len(self.cells)})"
        )
