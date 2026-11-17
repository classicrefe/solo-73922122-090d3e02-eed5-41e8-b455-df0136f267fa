"""时刻校正 API 的端到端集成测试（FastAPI + 真实 PostgreSQL）。

覆盖需求：
- 只改 read_time、不改累计 Wh；左闭右开窗口、跨日移动；
- 目标时刻可以是本批其他记录搬走腾出的位置（链式移动）；
- 与未移动读数冲突 / 超 500 条 / 参数非法时整次回滚或直接拒绝；
- 并发校正、直连 SQL 写入冲突时整次回滚，不覆盖、不丢 Wh 更正；
- 成功后 POST /jobs 的只读 SQL 读到新时刻，失败后仍是原记录。
"""
from __future__ import annotations

import socket
import threading
import time
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import config as config_mod

UTC = timezone.utc
TEST_METER = 9101
# 测试窗口附近的整点基线时刻
T0 = datetime(2026, 11, 15, 8, 0, tzinfo=UTC)


def _dsn() -> str:
    return config_mod.settings.pg_dsn


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_base_url(pg_dsn: str) -> str:
    """真实 uvicorn HTTP 服务（与 Docker 部署同形态）。

    并发用例必须走真实 TCP：starlette TestClient 的同步 portal 在
    “主线程持锁 + join 工作线程、工作线程同步等锁”时会自死锁，
    与被测应用无关。服务在守护线程内运行，模块结束自动停止。
    """
    import uvicorn
    import httpx

    config_mod.settings.pg_dsn = pg_dsn
    from app.main import app

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn-test", daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("测试用 uvicorn 未在时限内就绪")
    yield base
    server.should_exit = True
    thread.join(timeout=5)



@pytest.fixture()
def meter_rows(pg_dsn: str):
    """每用例独立的测试表号与读数；用例结束自动清理。"""
    conn = psycopg.connect(pg_dsn, autocommit=True)
    try:
        conn.execute(
            "INSERT INTO meters(meter_id, line_name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (TEST_METER, "测试线"),
        )
        conn.execute("DELETE FROM readings WHERE meter_id = %s", (TEST_METER,))
        yield conn
    finally:
        conn.execute("DELETE FROM readings WHERE meter_id = %s", (TEST_METER,))
        conn.execute("DELETE FROM meters WHERE meter_id = %s", (TEST_METER,))
        conn.close()


def _seed(conn, spec: list[tuple[datetime, int]]) -> None:
    """spec: [(read_time, wh), ...]"""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO readings(meter_id, read_time, wh) VALUES (%s, %s, %s)",
            [(TEST_METER, t, w) for t, w in spec],
        )


def _snapshot(conn) -> dict[datetime, int]:
    rows = conn.execute(
        "SELECT read_time, wh FROM readings WHERE meter_id = %s ORDER BY read_time",
        (TEST_METER,),
    ).fetchall()
    return {t.astimezone(UTC): w for t, w in rows}


def _correction(client: TestClient, start: datetime, end: datetime, offset: int,
                meter_id: int = TEST_METER):
    return client.post(
        "/time-corrections",
        json={
            "meter_id": meter_id,
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "offset_minutes": offset,
        },
    )


def _http_correction(base_url: str, start: datetime, end: datetime, offset: int,
                     meter_id: int = TEST_METER):
    import httpx

    # 并发用例会让该请求阻塞在行锁上，直到另一个事务放锁；超时给到 30s。
    # 注意：被测环境（嵌入式 PG）的 lock_timeout 后端中断可能不触发，
    # 因此用例不依赖超时，而是靠放锁后的唯一/主键冲突确定性产生 409。
    return httpx.post(
        f"{base_url}/time-corrections",
        json={
            "meter_id": meter_id,
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "offset_minutes": offset,
        },
        timeout=30,
    )


def _wait_for_correction_waiting_on_lock(timeout: float = 5.0) -> None:
    """等到应用的校正事务阻塞在行锁上（用独立监控连接查询，避免污染持锁连接）。"""
    monitor = psycopg.connect(_dsn(), autocommit=True)
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            blocked = monitor.execute(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE application_name = 'recon-api-correction' "
                "AND wait_event_type = 'Lock'"
            ).fetchone()[0]
            if blocked:
                return
            time.sleep(0.05)
        raise AssertionError("校正事务未在时限内进入等锁状态")
    finally:
        monitor.close()


# ---------------------------------------------------------------- 基本行为

def test_health_and_openapi_list_route(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/time-corrections" in paths


def test_shift_moves_time_but_keeps_wh(client: TestClient, meter_rows) -> None:
    _seed(meter_rows, [(T0, 100), (T0 + timedelta(hours=1), 200)])

    resp = _correction(client, T0, T0 + timedelta(hours=1), 30)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["moved_count"] == 1
    assert body["offset_minutes"] == 30

    after = _snapshot(meter_rows)
    assert after == {
        T0 + timedelta(minutes=30): 100,   # 时刻移动、Wh 原样
        T0 + timedelta(hours=1): 200,      # 窗口右开，第二条不动
    }


def test_negative_offset_and_half_open_window(client: TestClient, meter_rows) -> None:
    _seed(meter_rows, [(T0, 10), (T0 + timedelta(hours=1), 20),
                       (T0 + timedelta(hours=2), 30)])
    # 窗口 [T0, T0+2h)：恰好不含 T0+2h
    resp = _correction(client, T0, T0 + timedelta(hours=2), -15)
    assert resp.status_code == 200
    assert resp.json()["moved_count"] == 2

    assert _snapshot(meter_rows) == {
        T0 - timedelta(minutes=15): 10,
        T0 + timedelta(minutes=45): 20,
        T0 + timedelta(hours=2): 30,
    }


def test_cross_day_shift_takes_effect(client: TestClient, meter_rows) -> None:
    day_end = datetime(2026, 11, 15, 23, 30, tzinfo=UTC)
    _seed(meter_rows, [(day_end, 777)])
    resp = _correction(client, day_end, day_end + timedelta(hours=1), 45)
    assert resp.status_code == 200
    assert resp.json()["moved_count"] == 1
    after = _snapshot(meter_rows)
    assert list(after) == [datetime(2026, 11, 16, 0, 15, tzinfo=UTC)]
    assert after[datetime(2026, 11, 16, 0, 15, tzinfo=UTC)] == 777


def test_chain_shift_into_vacated_slot_within_batch(client: TestClient, meter_rows) -> None:
    # 三条连续读数整体后移 60 分钟：每条的目标时刻都是“本批前一条搬走的位置”
    # （首条目标为全新位置），必须全部成功而不是误判冲突；wh 随各自读数移动。
    spec = [(T0 + timedelta(hours=i), 100 + i * 10) for i in range(3)]
    _seed(meter_rows, spec)

    resp = _correction(client, T0, T0 + timedelta(hours=3), 60)
    assert resp.status_code == 200, resp.text
    assert resp.json()["moved_count"] == 3

    assert _snapshot(meter_rows) == {
        T0 + timedelta(hours=1): 100,
        T0 + timedelta(hours=2): 110,
        T0 + timedelta(hours=3): 120,
    }


def test_conflict_with_unmoved_reading_rolls_back(client: TestClient, meter_rows) -> None:
    # T0 后移 60 分钟会撞到未被本批选中的 T0+1h
    _seed(meter_rows, [(T0, 100), (T0 + timedelta(hours=1), 200)])
    before = _snapshot(meter_rows)

    resp = _correction(client, T0, T0 + timedelta(minutes=30), 60)
    assert resp.status_code == 409, resp.text
    assert "回滚" in resp.json()["detail"]["message"]

    # 失败后仍是原记录，一条都没动
    assert _snapshot(meter_rows) == before


def test_zero_match_returns_zero_and_changes_nothing(client: TestClient, meter_rows) -> None:
    _seed(meter_rows, [(T0, 100)])
    resp = _correction(client, T0 + timedelta(days=1), T0 + timedelta(days=2), 10)
    assert resp.status_code == 200
    assert resp.json()["moved_count"] == 0
    assert _snapshot(meter_rows) == {T0: 100}


# ---------------------------------------------------------------- 输入校验

def test_offset_zero_and_out_of_range_rejected(client: TestClient, meter_rows) -> None:
    _seed(meter_rows, [(T0, 1)])
    for bad_offset in (0, 121, -121):
        resp = _correction(client, T0, T0 + timedelta(hours=1), bad_offset)
        assert resp.status_code == 422, bad_offset

    # 边界值合法
    resp = _correction(client, T0, T0 + timedelta(hours=1), 120)
    assert resp.status_code == 200
    assert _snapshot(meter_rows) == {T0 + timedelta(hours=2): 1}


def test_naive_datetime_rejected(client: TestClient) -> None:
    resp = client.post(
        "/time-corrections",
        json={
            "meter_id": TEST_METER,
            "start_utc": "2026-11-15T08:00:00",       # 无时区
            "end_utc": "2026-11-15T09:00:00Z",
            "offset_minutes": 10,
        },
    )
    assert resp.status_code == 422


def test_inverted_window_rejected(client: TestClient, meter_rows) -> None:
    _seed(meter_rows, [(T0, 1)])
    resp = _correction(client, T0 + timedelta(hours=1), T0, 10)
    assert resp.status_code == 422


def test_more_than_500_rows_rejected_and_rolled_back(client: TestClient, meter_rows) -> None:
    # 501 条、间隔 1 分钟，窗口全覆盖
    spec = [(T0 + timedelta(minutes=i), i) for i in range(501)]
    _seed(meter_rows, spec)
    before = _snapshot(meter_rows)

    resp = _correction(client, T0, T0 + timedelta(minutes=501), 5)
    assert resp.status_code == 422
    assert "500" in resp.json()["detail"]["message"]
    # 超限在事务内判定，整体回滚：原记录一条不少
    assert _snapshot(meter_rows) == before


def test_exactly_500_rows_allowed(client: TestClient, meter_rows) -> None:
    # 边界值：恰好 500 条必须成功（上限判断为 > 500 而非 >= 500）
    spec = [(T0 + timedelta(minutes=i), i) for i in range(500)]
    _seed(meter_rows, spec)

    resp = _correction(client, T0, T0 + timedelta(minutes=500), 5)
    assert resp.status_code == 200, resp.text
    assert resp.json()["moved_count"] == 500

    after = _snapshot(meter_rows)
    assert len(after) == 500
    # 每条都后移 5 分钟，wh 保持原值
    assert after[T0 + timedelta(minutes=5)] == 0
    assert after[T0 + timedelta(minutes=499 + 5)] == 499


# ---------------------------------------------------------------- 并发与直连 SQL

def test_concurrent_correction_converging_targets_one_rolls_back(
    http_base_url: str, meter_rows
) -> None:
    """两个并发校正奔向同一目标时刻：一个成功、另一个必须 409 回滚。

    确定性时序（同表号，源行互不重叠）：
      读数 R1=(T0,wh100)、R3=(T0+2h,wh200)，目标位都是 T0+1h。
      管理员事务先锁住 A 的选中行 R1，A（R1: T0->T0+1h）在
      SELECT ... FOR UPDATE 阶段阻塞；
      此时 B（R3: T0+2h->T0+1h）的源行与 A 不重叠，可正常移动并提交，
      先占用目标位 T0+1h；
      管理员 ROLLBACK 放锁后，A 继续，在冲突探测阶段发现目标位已被
      B 占用（或在 INSERT 时撞唯一约束），整次 409 回滚。
      最终：B 的移动生效、A 的 R1 原样保留，无覆盖、无丢失。
    """
    _seed(meter_rows, [(T0, 100), (T0 + timedelta(hours=2), 200)])
    admin = psycopg.connect(_dsn(), autocommit=True)
    try:
        admin.execute("BEGIN")
        # 锁住 A 的选中行 R1，迫使 A 停在第一步 SELECT ... FOR UPDATE
        locked = admin.execute(
            "SELECT read_time FROM readings WHERE meter_id = %s "
            "AND read_time = %s FOR UPDATE",
            (TEST_METER, T0),
        ).fetchall()
        assert locked

        results: dict[str, int] = {}

        def call(name: str, start: datetime, end: datetime, offset: int) -> None:
            r = _http_correction(http_base_url, start, end, offset)
            results[name] = r.status_code

        t_a = threading.Thread(
            target=call, args=("A", T0, T0 + timedelta(hours=1), 60)
        )
        t_a.start()
        _wait_for_correction_waiting_on_lock()

        # B 的源行 R3 不被任何锁挡住，应直接成功并先占目标位 T0+1h
        rb = _http_correction(
            http_base_url,
            T0 + timedelta(hours=2),
            T0 + timedelta(hours=3),
            -60,
        )
        assert rb.status_code == 200, rb.text

        admin.execute("ROLLBACK")  # 放开 A 等待的 R1 行锁
        t_a.join(timeout=20)
        assert not t_a.is_alive()

        # A 必须因冲突而回滚；B 已成功
        assert results == {"A": 409}, results

        # 最终状态：A 的 R1 原样在 T0；B 把 R3 搬到了 T0+1h
        assert _snapshot(meter_rows) == {
            T0: 100,
            T0 + timedelta(hours=1): 200,
        }
    finally:
        admin.close()


def test_direct_sql_insert_conflict_rolls_back(http_base_url: str, meter_rows) -> None:
    """校正进行中，直连 SQL 抢先占用目标时刻：校正必须回滚，不覆盖新读数。"""
    _seed(meter_rows, [(T0, 100)])
    admin = psycopg.connect(_dsn(), autocommit=True)
    try:
        admin.execute("BEGIN")
        # 占住选中行的锁，让校正事务停在 SELECT ... FOR UPDATE
        locked = admin.execute(
            "SELECT read_time FROM readings WHERE meter_id = %s FOR UPDATE",
            (TEST_METER,),
        ).fetchall()
        assert locked

        outcome: dict[str, int] = {}

        def call_api() -> None:
            r = _http_correction(http_base_url, T0, T0 + timedelta(hours=1), 60)
            outcome["code"] = r.status_code

        t = threading.Thread(target=call_api)
        t.start()
        _wait_for_correction_waiting_on_lock()

        # 直连 SQL 写入校正的目标时刻（不同源时刻，不被行锁挡住）
        admin.execute(
            "INSERT INTO readings(meter_id, read_time, wh) VALUES (%s, %s, %s)",
            (TEST_METER, T0 + timedelta(hours=1), 555),
        )
        admin.execute("COMMIT")  # 同时放行校正事务
        t.join(timeout=15)
        assert not t.is_alive()

        assert outcome["code"] == 409
        # 两条读数都在：直连写入不被覆盖，原读数不丢失
        assert _snapshot(meter_rows) == {T0: 100, T0 + timedelta(hours=1): 555}
    finally:
        admin.close()


def test_direct_sql_wh_correction_is_not_lost(client: TestClient, meter_rows) -> None:
    """直连 SQL 先提交 Wh 更正，随后的时刻校正必须带着新 Wh 移动，旧值不复活。"""
    _seed(meter_rows, [(T0, 100)])
    meter_rows.execute(
        "UPDATE readings SET wh = 123 WHERE meter_id = %s AND read_time = %s",
        (TEST_METER, T0),
    )
    resp = _correction(client, T0, T0 + timedelta(hours=1), 30)
    assert resp.status_code == 200
    assert _snapshot(meter_rows) == {T0 + timedelta(minutes=30): 123}


# ---------------------------------------------------------------- 与 /jobs 联动

def _wait_job_terminal(client: TestClient, job_id: str) -> dict:
    deadline = time.time() + 20
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed", "cancelled", "timeout"):
            return body
        time.sleep(0.05)
    raise AssertionError("任务未在时限内到达终态")


def test_jobs_sql_reads_new_time_after_success_and_old_after_failure(
    client: TestClient, meter_rows
) -> None:
    _seed(meter_rows, [(T0, 100), (T0 + timedelta(hours=1), 200)])

    def job_count_at(t: datetime) -> int:
        sql = (
            "SELECT meter_id::bigint, wh::bigint FROM readings "
            f"WHERE meter_id = {TEST_METER} AND read_time = TIMESTAMPTZ '{t.isoformat()}'"
        )
        resp = client.post("/jobs", json={"left_sql": sql, "right_sql": sql})
        assert resp.status_code == 201
        body = _wait_job_terminal(client, resp.json()["job_id"])
        assert body["status"] == "succeeded", body
        return body["result"]["left_row_total"]

    # 成功的校正：新时刻可读、旧时刻为空
    ok = _correction(client, T0, T0 + timedelta(minutes=30), 30)
    assert ok.status_code == 200
    assert job_count_at(T0 + timedelta(minutes=30)) == 1
    assert job_count_at(T0) == 0

    # 失败的校正：原记录仍在原时刻
    clash = _correction(client, T0 + timedelta(minutes=30),
                        T0 + timedelta(hours=1), 30)
    assert clash.status_code == 409
    assert job_count_at(T0 + timedelta(minutes=30)) == 1
