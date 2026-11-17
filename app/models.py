"""API 请求 / 响应模型。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

JobStatus = Literal[
    "pending",      # 已受理，等待运行名额
    "running",      # 运行中
    "cancelling",   # 已收到取消请求，正在中止本轮 SQL
    "succeeded",    # 终态：核对完成（可能一致，也可能有差异）
    "failed",       # 终态：任一侧失败
    "cancelled",    # 终态：用户取消
    "timeout",      # 终态：超过单轮时限
]

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "timeout"}
ACTIVE_STATUSES = {"pending", "running", "cancelling"}


class CreateJobRequest(BaseModel):
    left_sql: str = Field(..., min_length=1, description="左侧只读 SELECT（如改写后的日结 SQL）")
    right_sql: str = Field(..., min_length=1, description="右侧只读 SELECT（基准 SQL）")
    name: str | None = Field(None, max_length=128, description="可选备注名")


class TimeCorrectionRequest(BaseModel):
    """读数时刻校正请求：把窗口内该表号的读数时刻整体平移。"""

    meter_id: int = Field(..., description="表号（meters.meter_id）")
    start_utc: datetime = Field(
        ..., description="UTC 窗口起点（含），ISO 8601；建议显式带 Z 或 +00:00"
    )
    end_utc: datetime = Field(
        ..., description="UTC 窗口终点（不含），ISO 8601；建议显式带 Z 或 +00:00"
    )
    offset_minutes: int = Field(
        ..., description="分钟偏移，非零，绝对值不超过 120；正数向后、负数向前"
    )

    @field_validator("start_utc", "end_utc")
    @classmethod
    def _normalize_utc(cls, value: datetime) -> datetime:
        # 时刻语义必须明确：拒绝不带时区的输入，统一转成 UTC
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("时刻必须带时区信息（UTC，例如 2026-11-15T08:00:00Z）")
        return value.astimezone(timezone.utc)

    @field_validator("offset_minutes")
    @classmethod
    def _validate_offset(cls, value: int) -> int:
        if value == 0:
            raise ValueError("offset_minutes 必须非零")
        if abs(value) > 120:
            raise ValueError("offset_minutes 的绝对值不能超过 120")
        return value

    def validate_window(self) -> None:
        if not self.start_utc < self.end_utc:
            raise ValueError("start_utc 必须早于 end_utc（左闭右开区间）")


class TimeCorrectionResponse(BaseModel):
    moved_count: int = Field(..., description="实际移动的读数条数")
    meter_id: int
    start_utc: datetime
    end_utc: datetime
    offset_minutes: int


class DiffRow(BaseModel):
    # 与结果集列顺序对齐；NULL 用 null 表示，空串为 ""，二者不同
    values: list[Any]
    left_count: int
    right_count: int


class ReconcileResult(BaseModel):
    identical: bool
    columns: list[str]
    column_types: list[str]          # bigint | text
    left_row_total: int
    right_row_total: int
    matched_group_count: int         # 行内容与次数都相同的去重行数
    diff_group_total: int            # 存在差异的去重行种类总数（不受 20 条展示上限影响）
    diffs_shown: int
    diffs: list[DiffRow]             # 最多 20 种


class JobOut(BaseModel):
    job_id: str
    name: str | None
    status: JobStatus
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    deadline: datetime | None
    cancel_requested: bool
    error: str | None
    result: ReconcileResult | None

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(timezone.utc)
