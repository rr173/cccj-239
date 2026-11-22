"""数据集错误类型。"""


class DatasetError(Exception):
    """数据集层面的一般错误。"""


class ValidationError(DatasetError):
    """输入记录或查询参数不合法。"""


class ConflictError(DatasetError):
    """同一测点对的剩余证据折算值不一致，拒绝发布。"""


class UnknownRecordError(DatasetError):
    """撤回引用了不存在的记录。"""


class DuplicateRecordError(DatasetError):
    """记录身份重复。"""


class StaleTransactionError(DatasetError):
    """订正事务基于过期版本。"""
