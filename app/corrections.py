"""读数时刻批量校正（只移动 read_time，不改累计 Wh）。

事务与并发保证：
- 整次校正在单个事务内完成，任何一步失败（含与未移动读数冲突、
  并发校正 / 直连 SQL 写入撞主键、等锁超时）都整体回滚，
  不会覆盖其他读数，也不会丢掉其他会话已提交的 Wh 更正；
- 先用 SELECT ... FOR UPDATE 锁住本次窗口内该表号的全部选中行，
  再用一次只针对“非选中行”的 SELECT ... FOR UPDATE 探测目标时刻是否
  已被其他事务占用；DELETE + INSERT 不使用 ON CONFLICT，任何漏网冲突
  都会以主键 / 唯一约束冲突的形式让事务回滚，而不是静默覆盖；
- 目标时刻允许落在本批其他记录移走后腾出的位置（同批移动是 read_time
  上的整体平移，天然不冲突），也允许跨日。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import psycopg

from .config import settings


class CorrectionError(RuntimeError):
    """请求本身不合法（如选中行数超过上限）。"""


class CorrectionConflict(RuntimeError):
    """与未移动读数或并发写入冲突；整次校正已回滚。"""


@dataclass(frozen=True)
class CorrectionResult:
    moved_count: int
    meter_id: int
    start_utc: object
    end_utc: object
    offset_minutes: int


def _open_correction_connection() -> psycopg.Connection:
    """校正连接：需要写权限，单独设置更短的锁等待，避免长时间互相阻塞。"""
    conn = psycopg.connect(
        settings.pg_dsn,
        connect_timeout=settings.connect_timeout,
        autocommit=True,  # 事务边界用 with conn.transaction() 显式控制
        application_name="recon-api-correction",
    )
    # 均为内部 int 配置，直接内联安全
    conn.execute(f"SET SESSION statement_timeout = {int(settings.statement_timeout_ms)}")
    conn.execute(f"SET SESSION lock_timeout = {int(settings.correction_lock_timeout_ms)}")
    conn.execute(
        f"SET SESSION idle_in_transaction_session_timeout = "
        f"{int(settings.idle_in_tx_timeout_ms)}"
    )
    return conn


def apply_time_correction(
    *,
    meter_id: int,
    start_utc,
    end_utc,
    offset_minutes: int,
    max_rows: int,
) -> CorrectionResult:
    """把 [start_utc, end_utc) 内该表号的读数时刻整体平移 offset_minutes。

    start_utc / end_utc 必须是带 UTC 时区的 datetime；offset 已在 API 层
    校验为非零且绝对值不超过 120。返回实际移动条数。
    """
    conn = _open_correction_connection()
    try:
        try:
            with conn.transaction():
                # 1) 锁定并取出本次窗口内的全部选中行（左闭右开）。
                #    FOR UPDATE 会挡住其他事务对同一批行的 Wh 更正，
                #    保证本事务的 DELETE+INSERT 基于最新已提交状态。
                selected = conn.execute(
                    """
                    SELECT read_time, wh
                    FROM readings
                    WHERE meter_id = %s
                      AND read_time >= %s
                      AND read_time <  %s
                    ORDER BY read_time
                    FOR UPDATE
                    """,
                    (meter_id, start_utc, end_utc),
                ).fetchall()

                count = len(selected)
                if count == 0:
                    # 没有命中任何行：事务内未做任何修改，直接返回 0
                    return CorrectionResult(
                        moved_count=0,
                        meter_id=meter_id,
                        start_utc=start_utc,
                        end_utc=end_utc,
                        offset_minutes=offset_minutes,
                    )
                if count > max_rows:
                    raise CorrectionError(
                        f"窗口内命中 {count} 条读数，单次最多处理 {max_rows} 条，"
                        "请缩小时间范围后重试（本次未做任何修改）"
                    )

                source_times = [row[0] for row in selected]
                wh_values = [row[1] for row in selected]
                delta = timedelta(minutes=offset_minutes)
                target_times = [t + delta for t in source_times]

                # 2) 探测与“未移动读数”的冲突：只查不属于本批的行。
                #    同批目标时刻（含本批其他记录搬走后腾出的位置）放行；
                #    NOT IN 走数组参数，行数有 500 上限，开销可控。
                #    FOR UPDATE 让并发直连 INSERT/校正也在这里互相阻塞，
                #    对方若先提交则下一条语句的唯一约束兜底回滚。
                blocked = conn.execute(
                    """
                    SELECT read_time
                    FROM readings
                    WHERE meter_id = %s
                      AND read_time = ANY(%s::timestamptz[])
                      AND read_time <> ALL(%s::timestamptz[])
                    FOR UPDATE
                    """,
                    (meter_id, target_times, source_times),
                ).fetchall()
                if blocked:
                    clash = min(r[0] for r in blocked)
                    raise CorrectionConflict(
                        f"目标时刻 {clash.isoformat()} 已存在该表号的未移动读数，"
                        "整次校正已回滚，未覆盖任何读数"
                    )

                # 3) 删除旧时刻、插入新时刻。不带 ON CONFLICT：
                #    并发校正 / 直连 SQL 写入造成的撞键直接抛唯一约束冲突，
                #    由外层捕获并回滚，绝不静默覆盖他人数据。
                conn.execute(
                    """
                    DELETE FROM readings
                    WHERE meter_id = %s
                      AND read_time = ANY(%s::timestamptz[])
                    """,
                    (meter_id, source_times),
                )
                conn.execute(
                    """
                    INSERT INTO readings (meter_id, read_time, wh)
                    SELECT %s, x.read_time, x.wh
                    FROM ROWS FROM (
                        unnest(%s::timestamptz[]),
                        unnest(%s::bigint[])
                    ) AS x(read_time, wh)
                    """,
                    (meter_id, target_times, wh_values),
                )
            # with 退出且无异常 => COMMIT；任何异常 => ROLLBACK
        except CorrectionError:
            raise
        except CorrectionConflict:
            raise
        except psycopg.errors.UniqueViolation as exc:
            raise CorrectionConflict(
                "目标时刻与并发校正或直连 SQL 写入冲突，整次校正已回滚，"
                "未覆盖其他读数"
            ) from exc
        except psycopg.errors.LockNotAvailable as exc:
            raise CorrectionConflict(
                "等待读数行锁超时，可能存在并发校正或直连 SQL 写入，"
                "整次校正已回滚"
            ) from exc
        except psycopg.errors.SerializationFailure as exc:
            raise CorrectionConflict(
                "并发写入导致序列化失败，整次校正已回滚，请重试"
            ) from exc
        except psycopg.Error as exc:
            raise CorrectionError(f"数据库执行失败，整次校正已回滚：{exc}") from exc

        return CorrectionResult(
            moved_count=count,
            meter_id=meter_id,
            start_utc=start_utc,
            end_utc=end_utc,
            offset_minutes=offset_minutes,
        )
    finally:
        conn.close()
