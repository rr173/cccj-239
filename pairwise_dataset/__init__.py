"""可订正的落盘成对比测稀疏矩阵数据集。"""

from .dataset import Correction, Dataset, Snapshot
from .errors import (
    ConflictError,
    DatasetError,
    DuplicateRecordError,
    StaleTransactionError,
    UnknownRecordError,
    ValidationError,
)
from .model import FORWARD, REVERSE, SELF, Cell, EvidenceRef, GapEntry, WindowResult
from .workorder import (
    MAX_PAGE_SIZE,
    MAX_REQUIREMENTS,
    MIN_PAGE_SIZE,
    GapPage,
    WorkOrder,
)

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "Snapshot",
    "Correction",
    "Cell",
    "EvidenceRef",
    "WindowResult",
    "GapEntry",
    "GapPage",
    "WorkOrder",
    "FORWARD",
    "REVERSE",
    "SELF",
    "MAX_REQUIREMENTS",
    "MIN_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "DatasetError",
    "ValidationError",
    "ConflictError",
    "UnknownRecordError",
    "DuplicateRecordError",
    "StaleTransactionError",
]
