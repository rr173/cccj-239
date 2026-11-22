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
from typing import Dict, Tuple

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
