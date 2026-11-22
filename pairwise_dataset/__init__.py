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
from .model import FORWARD, REVERSE, SELF, Cell, EvidenceRef, WindowResult

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "Snapshot",
    "Correction",
    "Cell",
    "EvidenceRef",
    "WindowResult",
    "FORWARD",
    "REVERSE",
    "SELF",
    "DatasetError",
    "ValidationError",
    "ConflictError",
    "UnknownRecordError",
    "DuplicateRecordError",
    "StaleTransactionError",
]
