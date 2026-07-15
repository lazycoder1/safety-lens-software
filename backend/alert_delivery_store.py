"""PostgreSQL authority for restart-safe, per-target alert delivery work."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

from psycopg2.extras import Json, RealDictCursor

from db import get_conn


_get_conn = get_conn
_NOTIFY_CHANNEL = "safetylens_alert_delivery"
LeaseRenewalStatus = Literal["renewed", "inactive", "lost"]


@dataclass(frozen=True, slots=True)
class InitialDeliveryTiming:
    """Durable timing anchors emitted only for the first initial delivery."""

    alert_id: str
    camera_id: str
    first_positive_at: datetime | None
    confirmed_at: datetime | None
    persisted_at: datetime | None
    first_initial_delivery_at: datetime


@dataclass(frozen=True, slots=True)
class DeliveryCompletion:
    """Result of a fenced delivery completion transaction."""

    updated: bool
    initial_delivery_timing: InitialDeliveryTiming | None = None


def _utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _env_seconds(name: str, default: float, minimum: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, value)


def _delivery_lifetime_seconds() -> float:
    return _env_seconds("ALERT_DELIVERY_MAX_AGE_SECONDS", 15 * 60)


def _priority_aging_seconds() -> float:
    """Promote a waiting row one priority band at a bounded interval."""
    return _env_seconds("ALERT_DELIVERY_PRIORITY_AGING_SECONDS", 60.0, minimum=5.0)


def init_db() -> None:
    """Create and forward-migrate the outbox without dropping pending work."""
    lifetime = _delivery_lifetime_seconds()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_delivery_outbox (
                    id UUID PRIMARY KEY,
                    alert_id TEXT NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL DEFAULT 'initial',
                    target_key TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    context JSONB NOT NULL DEFAULT '{}'::jsonb,
                    priority SMALLINT NOT NULL DEFAULT 4,
                    state TEXT NOT NULL DEFAULT 'pending'
                        CHECK (state IN ('pending', 'delivered', 'terminal', 'cancelled')),
                    claim_count INTEGER NOT NULL DEFAULT 0 CHECK (claim_count >= 0),
                    internal_failure_count INTEGER NOT NULL DEFAULT 0
                        CHECK (internal_failure_count >= 0),
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                    eligible_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    next_attempt_at TIMESTAMPTZ NOT NULL,
                    leased_by TEXT,
                    lease_token UUID,
                    lease_expires_at TIMESTAMPTZ,
                    last_attempt_at TIMESTAMPTZ,
                    last_error_code TEXT,
                    last_error_message TEXT,
                    last_acceptance_unknown BOOLEAN NOT NULL DEFAULT FALSE,
                    ever_acceptance_unknown BOOLEAN NOT NULL DEFAULT FALSE,
                    send_in_flight BOOLEAN NOT NULL DEFAULT FALSE,
                    terminal_reason TEXT,
                    delivered_at TIMESTAMPTZ,
                    terminal_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                    UNIQUE (alert_id, kind, target_key),
                    CHECK ((lease_token IS NULL) = (lease_expires_at IS NULL))
                )
                """
            )
            # Forward migration for development/pre-release databases created
            # by earlier outbox iterations. Production never needs a table drop.
            cur.execute(
                "ALTER TABLE alert_delivery_outbox "
                "ADD COLUMN IF NOT EXISTS claim_count INTEGER NOT NULL DEFAULT 0"
            )
            cur.execute(
                "ALTER TABLE alert_delivery_outbox "
                "ADD COLUMN IF NOT EXISTS internal_failure_count INTEGER NOT NULL DEFAULT 0"
            )
            cur.execute(
                "ALTER TABLE alert_delivery_outbox "
                "ADD COLUMN IF NOT EXISTS eligible_at TIMESTAMPTZ"
            )
            cur.execute(
                "ALTER TABLE alert_delivery_outbox "
                "ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ"
            )
            cur.execute(
                "ALTER TABLE alert_delivery_outbox "
                "ADD COLUMN IF NOT EXISTS ever_acceptance_unknown BOOLEAN NOT NULL DEFAULT FALSE"
            )
            cur.execute(
                "ALTER TABLE alert_delivery_outbox "
                "ADD COLUMN IF NOT EXISTS send_in_flight BOOLEAN NOT NULL DEFAULT FALSE"
            )
            cur.execute(
                "ALTER TABLE alert_delivery_outbox "
                "ADD COLUMN IF NOT EXISTS terminal_reason TEXT"
            )
            cur.execute(
                """
                UPDATE alert_delivery_outbox
                SET eligible_at = COALESCE(eligible_at, next_attempt_at, created_at)
                WHERE eligible_at IS NULL
                """
            )
            cur.execute(
                """
                UPDATE alert_delivery_outbox
                SET expires_at = eligible_at + (%s * interval '1 second')
                WHERE expires_at IS NULL
                """,
                (lifetime,),
            )
            cur.execute(
                "ALTER TABLE alert_delivery_outbox ALTER COLUMN eligible_at SET NOT NULL"
            )
            cur.execute(
                "ALTER TABLE alert_delivery_outbox ALTER COLUMN expires_at SET NOT NULL"
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alert_delivery_due_v2
                ON alert_delivery_outbox (next_attempt_at, priority, id)
                WHERE state = 'pending'
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alert_delivery_alert
                ON alert_delivery_outbox (alert_id, kind, state)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alert_delivery_completed
                ON alert_delivery_outbox (state, updated_at)
                WHERE state <> 'pending'
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_alert_delivery_expiry
                ON alert_delivery_outbox (expires_at, id)
                WHERE state = 'pending'
                """
            )
        conn.commit()


def insert_targets(
    cursor,
    *,
    alert_id: str,
    alert_timestamp: str,
    targets: list[dict] | None,
) -> int:
    """Insert snapshotted obligations inside the alert transaction."""
    inserted = 0
    default_lifetime = _delivery_lifetime_seconds()
    for target in targets or []:
        if not isinstance(target, dict):
            continue
        delay_seconds = _safe_seconds(
            target.get("delay_seconds", 0.0),
            default=0.0,
            minimum=0.0,
            maximum=365 * 24 * 60 * 60,
        )
        lifetime_seconds = _safe_seconds(
            target.get("max_age_seconds", default_lifetime),
            default=default_lifetime,
            minimum=1.0,
            maximum=30 * 24 * 60 * 60,
        )
        target_key = target.get("target_key")
        channel = target.get("channel")
        if not target_key or not channel:
            continue
        cursor.execute(
            """
            INSERT INTO alert_delivery_outbox (
                id, alert_id, kind, target_key, channel, context, priority,
                eligible_at, expires_at, next_attempt_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s::timestamptz + (%s * interval '1 second'),
                %s::timestamptz + (%s * interval '1 second')
                    + (%s * interval '1 second'),
                %s::timestamptz + (%s * interval '1 second')
            )
            ON CONFLICT (alert_id, kind, target_key) DO NOTHING
            """,
            (
                str(uuid4()),
                alert_id,
                str(target.get("kind") or "initial"),
                str(target_key),
                str(channel),
                Json(target.get("context") or {}),
                int(target.get("priority", 4)),
                alert_timestamp,
                delay_seconds,
                alert_timestamp,
                delay_seconds,
                lifetime_seconds,
                alert_timestamp,
                delay_seconds,
            ),
        )
        inserted += cur_rowcount(cursor)
    if inserted:
        # NOTIFY is delivered only if the surrounding alert transaction commits.
        cursor.execute("SELECT pg_notify(%s, %s)", (_NOTIFY_CHANNEL, alert_id))
    return inserted


def insert_existing_targets(
    *,
    alert_id: str,
    alert_timestamp: str,
    targets: list[dict],
) -> int:
    """Idempotently add upgrade/backfill targets for an existing alert."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            inserted = insert_targets(
                cur,
                alert_id=alert_id,
                alert_timestamp=alert_timestamp,
                targets=targets,
            )
        conn.commit()
    return inserted


def backfill_active_escalations_batch(
    config: dict,
    target_resolver,
    *,
    after_id: str | None = None,
    batch_size: int = 250,
    minimum_timestamp: str | None = None,
) -> tuple[int, str | None]:
    """Backfill one keyset page of future escalation obligations.

    Initial sends and already-due escalation steps are intentionally excluded:
    a legacy deployment has no durable evidence that those sends did not happen.
    """
    inserted = 0
    batch_size = max(1, min(int(batch_size), 2000))
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, severity, timestamp, camera_id,
                       clock_timestamp() AS database_now
                FROM alerts
                WHERE status = 'active'
                  AND (%s::text IS NULL OR timestamp >= %s::text)
                  AND (%s::text IS NULL OR id > %s::text)
                ORDER BY id
                LIMIT %s
                """,
                (
                    minimum_timestamp,
                    minimum_timestamp,
                    after_id,
                    after_id,
                    batch_size + 1,
                ),
            )
            alerts = cur.fetchall()
            page = alerts[:batch_size]
            for row in page:
                cur.execute("SAVEPOINT alert_escalation_backfill")
                try:
                    targets = target_resolver(
                        config,
                        {
                            "id": row["id"],
                            "severity": row["severity"],
                            "timestamp": row["timestamp"],
                            "cameraId": row["camera_id"],
                        },
                    )
                    alert_time = _coerce_datetime(row["timestamp"])
                    database_now = _coerce_datetime(row["database_now"])
                    escalation_targets = []
                    for target in targets:
                        if not isinstance(target, dict) or target.get("kind") != "escalation":
                            continue
                        delay = _safe_seconds(
                            target.get("delay_seconds", 0.0),
                            default=0.0,
                            minimum=0.0,
                            maximum=365 * 24 * 60 * 60,
                        )
                        if alert_time + timedelta(seconds=delay) <= database_now:
                            continue
                        escalation_targets.append(target)
                    inserted += insert_targets(
                        cur,
                        alert_id=str(row["id"]),
                        alert_timestamp=str(row["timestamp"]),
                        targets=escalation_targets,
                    )
                except Exception:
                    # A malformed legacy row/routing value must never prevent
                    # service startup or roll back other valid backfill rows.
                    cur.execute("ROLLBACK TO SAVEPOINT alert_escalation_backfill")
                finally:
                    cur.execute("RELEASE SAVEPOINT alert_escalation_backfill")
        conn.commit()
    next_cursor = str(page[-1]["id"]) if len(alerts) > batch_size and page else None
    return inserted, next_cursor


def backfill_active_escalations(config: dict, target_resolver) -> int:
    """Idempotently scan all active rows in bounded transactions."""
    inserted = 0
    after_id = None
    while True:
        batch_inserted, after_id = backfill_active_escalations_batch(
            config,
            target_resolver,
            after_id=after_id,
        )
        inserted += batch_inserted
        if after_id is None:
            return inserted


def claim_due(
    *,
    worker_id: str,
    lease_seconds: float,
    excluded_channels: list[str] | None = None,
    include_terminalized: bool = False,
) -> dict | None:
    """Lease one live due row; claiming does not consume a provider attempt."""
    lease_token = str(uuid4())
    lease_seconds = _safe_seconds(lease_seconds, default=60.0, minimum=0.1, maximum=3600.0)
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Bound stale cleanup per claim so a long outage cannot create one
            # unbounded startup transaction.
            cur.execute(
                """
                WITH stale AS (
                    SELECT id
                    FROM alert_delivery_outbox
                    WHERE state = 'pending'
                      AND expires_at <= clock_timestamp()
                      AND (lease_token IS NULL OR lease_expires_at <= clock_timestamp())
                    ORDER BY expires_at, id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 500
                )
                UPDATE alert_delivery_outbox AS delivery
                SET state = 'terminal',
                    terminal_reason = 'delivery_expired',
                    terminal_at = clock_timestamp(),
                    last_acceptance_unknown = (
                        delivery.last_acceptance_unknown OR delivery.send_in_flight
                    ),
                    ever_acceptance_unknown = (
                        delivery.ever_acceptance_unknown OR delivery.send_in_flight
                    ),
                    last_error_code = CASE
                        WHEN delivery.send_in_flight THEN 'interrupted_in_flight_attempt'
                        ELSE delivery.last_error_code
                    END,
                    send_in_flight = FALSE,
                    leased_by = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    updated_at = clock_timestamp()
                FROM stale
                WHERE delivery.id = stale.id
                RETURNING delivery.alert_id,
                          delivery.kind,
                          delivery.target_key
                """
            )
            terminalized = [dict(item) for item in cur.fetchall()]
            cur.execute(
                """
                WITH candidate AS (
                    SELECT id
                    FROM alert_delivery_outbox
                    WHERE state = 'pending'
                      AND next_attempt_at <= clock_timestamp()
                      AND expires_at > clock_timestamp()
                      AND (lease_token IS NULL OR lease_expires_at <= clock_timestamp())
                      AND NOT (channel = ANY(%s::text[]))
                    ORDER BY
                        GREATEST(
                            1,
                            priority - FLOOR(
                                EXTRACT(EPOCH FROM (
                                    clock_timestamp() - next_attempt_at
                                )) / %s
                            )::integer
                        ),
                        next_attempt_at,
                        id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE alert_delivery_outbox AS delivery
                SET leased_by = %s,
                    lease_token = %s,
                    lease_expires_at = clock_timestamp() + (%s * interval '1 second'),
                    claim_count = delivery.claim_count + 1,
                    last_acceptance_unknown = (
                        delivery.last_acceptance_unknown OR delivery.send_in_flight
                    ),
                    ever_acceptance_unknown = (
                        delivery.ever_acceptance_unknown OR delivery.send_in_flight
                    ),
                    last_error_code = CASE
                        WHEN delivery.send_in_flight THEN 'interrupted_in_flight_attempt'
                        ELSE delivery.last_error_code
                    END,
                    send_in_flight = FALSE,
                    updated_at = clock_timestamp()
                FROM candidate
                WHERE delivery.id = candidate.id
                RETURNING delivery.*,
                    EXTRACT(EPOCH FROM (clock_timestamp() - delivery.eligible_at))
                        AS age_seconds,
                    EXTRACT(EPOCH FROM (delivery.expires_at - clock_timestamp()))
                        AS remaining_lifetime_seconds
                """,
                (
                    list(excluded_channels or []),
                    _priority_aging_seconds(),
                    worker_id,
                    lease_token,
                    lease_seconds,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    claimed = _row_to_dict(row) if row else None
    if include_terminalized:
        return {
            "delivery": claimed,
            "terminalized": terminalized,
        }
    return claimed


def begin_send(
    delivery_id: str,
    lease_token: str,
    *,
    max_attempts: int,
    lease_seconds: float = 60.0,
) -> dict:
    """Fence cancellation/expiry and consume one provider attempt atomically."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT delivery.*, alerts.status AS alert_status,
                       clock_timestamp() AS database_now
                FROM alert_delivery_outbox AS delivery
                JOIN alerts ON alerts.id = delivery.alert_id
                WHERE delivery.id = %s
                FOR UPDATE OF delivery
                """,
                (delivery_id,),
            )
            row = cur.fetchone()
            if (
                row is None
                or row.get("state") != "pending"
                or str(row.get("lease_token") or "") != str(lease_token)
            ):
                conn.commit()
                return {"started": False, "reason": "lease_lost"}

            reason = None
            terminal = False
            cancelled = False
            release_only = False
            if row["expires_at"] <= row["database_now"]:
                reason = "delivery_expired"
                terminal = True
            elif row["lease_expires_at"] <= row["database_now"]:
                reason = "lease_expired"
                release_only = True
            elif int(row.get("attempt_count") or 0) >= max(1, int(max_attempts)):
                reason = "retry_attempts_exhausted"
                terminal = True
            elif row.get("kind") == "escalation" and row.get("alert_status") != "active":
                reason = "alert_inactive"
                cancelled = True

            if reason is not None:
                final_state = (
                    None
                    if release_only
                    else ("cancelled" if cancelled else "terminal")
                )
                cur.execute(
                    """
                    UPDATE alert_delivery_outbox
                    SET state = %s,
                        terminal_reason = %s,
                        terminal_at = CASE WHEN %s THEN clock_timestamp() ELSE terminal_at END,
                        send_in_flight = FALSE,
                        leased_by = NULL,
                        lease_token = NULL,
                        lease_expires_at = NULL,
                        updated_at = clock_timestamp()
                    WHERE id = %s AND lease_token = %s AND state = 'pending'
                    """,
                    (
                        "pending" if release_only else final_state,
                        None if release_only else reason,
                        terminal,
                        delivery_id,
                        lease_token,
                    ),
                )
                updated = cur.rowcount == 1
                conn.commit()
                result = {
                    "started": False,
                    "reason": reason,
                }
                if final_state is not None and updated:
                    result.update(
                        {
                            "final_state": final_state,
                            "alert_id": str(row.get("alert_id") or ""),
                            "kind": str(row.get("kind") or "initial"),
                            "target_key": str(row.get("target_key") or ""),
                        }
                    )
                return result

            cur.execute(
                """
                UPDATE alert_delivery_outbox
                SET attempt_count = attempt_count + 1,
                    last_attempt_at = clock_timestamp(),
                    send_in_flight = TRUE,
                    lease_expires_at = clock_timestamp() + (%s * interval '1 second'),
                    updated_at = clock_timestamp()
                WHERE id = %s AND lease_token = %s AND state = 'pending'
                RETURNING attempt_count,
                    EXTRACT(EPOCH FROM (expires_at - clock_timestamp()))
                        AS remaining_lifetime_seconds
                """,
                (
                    _safe_seconds(
                        lease_seconds,
                        default=60.0,
                        minimum=0.1,
                        maximum=3600.0,
                    ),
                    delivery_id,
                    lease_token,
                ),
            )
            started = cur.fetchone()
        conn.commit()
    if started is None:
        return {"started": False, "reason": "lease_lost"}
    return {
        "started": True,
        "attempt_count": int(started["attempt_count"]),
        "remaining_lifetime_seconds": max(
            0.0, float(started["remaining_lifetime_seconds"] or 0.0)
        ),
    }


def renew_lease_with_status(
    delivery_id: str,
    lease_token: str,
    *,
    lease_seconds: float,
) -> LeaseRenewalStatus:
    """Renew a lease and distinguish normal completion from fencing loss."""
    lease_seconds = _safe_seconds(lease_seconds, default=60.0, minimum=0.1, maximum=3600.0)
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT state,
                       lease_token::text AS lease_token,
                       expires_at > clock_timestamp() AS within_lifetime
                FROM alert_delivery_outbox
                WHERE id = %s
                FOR UPDATE
                """,
                (delivery_id,),
            )
            row = cur.fetchone()
            status: LeaseRenewalStatus
            if row is None:
                status = "lost"
            elif row["state"] != "pending":
                status = "inactive"
            elif not row["within_lifetime"]:
                status = "lost"
            elif row["lease_token"] is None:
                # Normal retry/release transitions keep the work pending but
                # atomically clear its lease before the process-local active
                # tuple is removed.
                status = "inactive"
            elif str(row["lease_token"]) != str(lease_token):
                status = "lost"
            else:
                # The row lock keeps completion and competing claimers from
                # changing the lifecycle classification between inspection and
                # renewal. An elapsed renewable lease can still be extended if
                # no competitor has replaced its token, preserving prior
                # behavior for a delayed renewer.
                cur.execute(
                    """
                    UPDATE alert_delivery_outbox
                    SET lease_expires_at = clock_timestamp() + (%s * interval '1 second'),
                        updated_at = clock_timestamp()
                    WHERE id = %s
                      AND lease_token = %s
                      AND state = 'pending'
                      AND expires_at > clock_timestamp()
                    """,
                    (lease_seconds, delivery_id, lease_token),
                )
                status = "renewed" if cur.rowcount == 1 else "lost"
        conn.commit()
    return status


def renew_lease(delivery_id: str, lease_token: str, *, lease_seconds: float) -> bool:
    """Preserve the original boolean API for callers that only need success."""
    return (
        renew_lease_with_status(
            delivery_id,
            lease_token,
            lease_seconds=lease_seconds,
        )
        == "renewed"
    )


def mark_delivered(
    delivery_id: str,
    lease_token: str,
    *,
    acceptance_unknown: bool = False,
) -> bool:
    """Complete a delivery while preserving the original boolean API."""
    return mark_delivered_with_timing(
        delivery_id,
        lease_token,
        acceptance_unknown=acceptance_unknown,
    ).updated


def mark_delivered_with_timing(
    delivery_id: str,
    lease_token: str,
    *,
    acceptance_unknown: bool = False,
) -> DeliveryCompletion:
    """Complete a target and durably stamp the first initial delivery.

    The outbox lease fence and alert timestamp are committed together. Only
    the transaction that changes ``first_initial_delivery_at`` receives a
    timing record, so concurrent initial targets cannot double-report it.
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE alert_delivery_outbox
                SET state = 'delivered',
                    delivered_at = clock_timestamp(),
                    terminal_reason = NULL,
                    last_error_code = NULL,
                    last_error_message = NULL,
                    last_acceptance_unknown = %s,
                    ever_acceptance_unknown = ever_acceptance_unknown OR %s,
                    send_in_flight = FALSE,
                    leased_by = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    updated_at = clock_timestamp()
                WHERE id = %s
                  AND lease_token = %s
                  AND state = 'pending'
                RETURNING alert_id, kind, delivered_at
                """,
                (
                    bool(acceptance_unknown),
                    bool(acceptance_unknown),
                    delivery_id,
                    lease_token,
                ),
            )
            completed = cur.fetchone()
            timing = None
            if completed and completed.get("kind") == "initial":
                cur.execute(
                    """
                    UPDATE alerts
                    SET first_initial_delivery_at = %s
                    WHERE id = %s
                      AND first_initial_delivery_at IS NULL
                    RETURNING id AS alert_id,
                              camera_id,
                              first_positive_at,
                              confirmed_at,
                              persisted_at,
                              first_initial_delivery_at
                    """,
                    (completed["delivered_at"], completed["alert_id"]),
                )
                stamped = cur.fetchone()
                if stamped is not None:
                    first_initial_delivery_at = _utc_datetime(
                        stamped["first_initial_delivery_at"]
                    )
                    if first_initial_delivery_at is None:
                        raise RuntimeError("initial delivery timestamp was not persisted")
                    timing = InitialDeliveryTiming(
                        alert_id=str(stamped["alert_id"]),
                        camera_id=str(stamped["camera_id"]),
                        first_positive_at=_utc_datetime(stamped.get("first_positive_at")),
                        confirmed_at=_utc_datetime(stamped.get("confirmed_at")),
                        persisted_at=_utc_datetime(stamped.get("persisted_at")),
                        first_initial_delivery_at=first_initial_delivery_at,
                    )
        conn.commit()
    return DeliveryCompletion(
        updated=completed is not None,
        initial_delivery_timing=timing,
    )


def mark_terminal(
    delivery_id: str,
    lease_token: str,
    *,
    error_code: str,
    error_message: str,
    terminal_reason: str | None = None,
    acceptance_unknown: bool = False,
) -> bool:
    return _finish(
        delivery_id,
        lease_token,
        state="terminal",
        error_code=error_code,
        error_message=error_message,
        terminal_reason=terminal_reason or error_code,
        acceptance_unknown=acceptance_unknown,
    )


def _finish(
    delivery_id: str,
    lease_token: str,
    *,
    state: str,
    error_code: str | None,
    error_message: str | None,
    terminal_reason: str | None,
    acceptance_unknown: bool,
) -> bool:
    timestamp_column = "delivered_at" if state == "delivered" else "terminal_at"
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE alert_delivery_outbox
                SET state = %s,
                    {timestamp_column} = clock_timestamp(),
                    terminal_reason = %s,
                    last_error_code = %s,
                    last_error_message = %s,
                    last_acceptance_unknown = %s,
                    ever_acceptance_unknown = ever_acceptance_unknown OR %s,
                    send_in_flight = FALSE,
                    leased_by = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    updated_at = clock_timestamp()
                WHERE id = %s
                  AND lease_token = %s
                  AND state = 'pending'
                """,
                (
                    state,
                    _bounded(terminal_reason),
                    _bounded(error_code),
                    _bounded(error_message),
                    bool(acceptance_unknown),
                    bool(acceptance_unknown),
                    delivery_id,
                    lease_token,
                ),
            )
            updated = cur.rowcount == 1
        conn.commit()
    return updated


def schedule_retry(
    delivery_id: str,
    lease_token: str,
    *,
    delay_seconds: float,
    max_attempts: int,
    error_code: str,
    error_message: str,
    acceptance_unknown: bool,
) -> str | None:
    """Persist a retry or atomically terminalize/cancel the leased row."""
    delay_seconds = _safe_seconds(
        delay_seconds,
        default=0.0,
        minimum=0.0,
        maximum=30 * 24 * 60 * 60,
    )
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT delivery.*, alerts.status AS alert_status,
                       clock_timestamp() AS database_now
                FROM alert_delivery_outbox AS delivery
                JOIN alerts ON alerts.id = delivery.alert_id
                WHERE delivery.id = %s
                FOR UPDATE OF delivery
                """,
                (delivery_id,),
            )
            row = cur.fetchone()
            if (
                row is None
                or row.get("state") != "pending"
                or str(row.get("lease_token") or "") != str(lease_token)
            ):
                conn.commit()
                return None

            next_attempt = row["database_now"] + timedelta(seconds=delay_seconds)
            cancelled = row.get("kind") == "escalation" and row.get("alert_status") != "active"
            attempts_exhausted = int(row.get("attempt_count") or 0) >= max(1, int(max_attempts))
            deadline_exhausted = next_attempt >= row["expires_at"]
            if cancelled or attempts_exhausted or deadline_exhausted:
                if cancelled:
                    state = "cancelled"
                    reason = "alert_inactive"
                else:
                    state = "terminal"
                    reason = (
                        "retry_attempts_exhausted"
                        if attempts_exhausted
                        else "retry_deadline_exceeds_lifetime"
                    )
                cur.execute(
                    """
                    UPDATE alert_delivery_outbox
                    SET state = %s,
                        terminal_reason = %s,
                        terminal_at = CASE WHEN %s = 'terminal'
                            THEN clock_timestamp() ELSE terminal_at END,
                        last_error_code = %s,
                        last_error_message = %s,
                        last_acceptance_unknown = %s,
                        ever_acceptance_unknown = ever_acceptance_unknown OR %s,
                        send_in_flight = FALSE,
                        leased_by = NULL,
                        lease_token = NULL,
                        lease_expires_at = NULL,
                        updated_at = clock_timestamp()
                    WHERE id = %s AND lease_token = %s AND state = 'pending'
                    """,
                    (
                        state,
                        reason,
                        state,
                        _bounded(error_code),
                        _bounded(error_message),
                        bool(acceptance_unknown),
                        bool(acceptance_unknown),
                        delivery_id,
                        lease_token,
                    ),
                )
                result = state if cur.rowcount == 1 else None
            else:
                cur.execute(
                    """
                    UPDATE alert_delivery_outbox
                    SET next_attempt_at = %s,
                        last_error_code = %s,
                        last_error_message = %s,
                        last_acceptance_unknown = %s,
                        ever_acceptance_unknown = ever_acceptance_unknown OR %s,
                        send_in_flight = FALSE,
                        leased_by = NULL,
                        lease_token = NULL,
                        lease_expires_at = NULL,
                        updated_at = clock_timestamp()
                    WHERE id = %s AND lease_token = %s AND state = 'pending'
                    """,
                    (
                        next_attempt,
                        _bounded(error_code),
                        _bounded(error_message),
                        bool(acceptance_unknown),
                        bool(acceptance_unknown),
                        delivery_id,
                        lease_token,
                    ),
                )
                result = "pending" if cur.rowcount == 1 else None
                if result:
                    cur.execute("SELECT pg_notify(%s, %s)", (_NOTIFY_CHANNEL, delivery_id))
        conn.commit()
    return result


def schedule_internal_retry(
    delivery_id: str,
    lease_token: str,
    *,
    delay_seconds: float,
    max_claims: int,
    error_code: str = "worker_error",
) -> str | None:
    """Bound poison/pre-send failures without consuming provider attempts."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT internal_failure_count, expires_at,
                       clock_timestamp() AS database_now
                FROM alert_delivery_outbox
                WHERE id = %s AND lease_token = %s AND state = 'pending'
                FOR UPDATE
                """,
                (delivery_id, lease_token),
            )
            row = cur.fetchone()
            if row is None:
                conn.commit()
                return None
            delay = _safe_seconds(delay_seconds, default=1.0, minimum=0.0, maximum=60.0)
            next_attempt = row["database_now"] + timedelta(seconds=delay)
            failure_count = int(row.get("internal_failure_count") or 0) + 1
            exhausted = (
                failure_count >= max(1, int(max_claims))
                or next_attempt >= row["expires_at"]
            )
            if exhausted:
                cur.execute(
                    """
                    UPDATE alert_delivery_outbox
                    SET state = 'terminal',
                        terminal_reason = 'worker_failures_exhausted',
                        terminal_at = clock_timestamp(),
                        internal_failure_count = %s,
                        last_error_code = %s,
                        last_error_message = 'Delivery worker failed before provider dispatch',
                        send_in_flight = FALSE,
                        leased_by = NULL,
                        lease_token = NULL,
                        lease_expires_at = NULL,
                        updated_at = clock_timestamp()
                    WHERE id = %s AND lease_token = %s AND state = 'pending'
                    """,
                    (failure_count, _bounded(error_code), delivery_id, lease_token),
                )
                result = "terminal" if cur.rowcount == 1 else None
            else:
                cur.execute(
                    """
                    UPDATE alert_delivery_outbox
                    SET next_attempt_at = %s,
                        internal_failure_count = %s,
                        last_error_code = %s,
                        last_error_message = 'Delivery worker failed before provider dispatch',
                        send_in_flight = FALSE,
                        leased_by = NULL,
                        lease_token = NULL,
                        lease_expires_at = NULL,
                        updated_at = clock_timestamp()
                    WHERE id = %s AND lease_token = %s AND state = 'pending'
                    """,
                    (
                        next_attempt,
                        failure_count,
                        _bounded(error_code),
                        delivery_id,
                        lease_token,
                    ),
                )
                result = "pending" if cur.rowcount == 1 else None
        conn.commit()
    return result


def release_unstarted_claim(
    delivery_id: str,
    lease_token: str,
    *,
    delay_seconds: float = 0.0,
    error_code: str = "shutdown_before_send",
) -> bool:
    """Return a never-started claim without consuming any failure budget."""
    delay = _safe_seconds(delay_seconds, default=0.0, minimum=0.0, maximum=60.0)
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE alert_delivery_outbox
                SET next_attempt_at = LEAST(
                        expires_at - interval '1 millisecond',
                        clock_timestamp() + (%s * interval '1 second')
                    ),
                    last_error_code = %s,
                    last_error_message = 'Delivery claim released before provider dispatch',
                    send_in_flight = FALSE,
                    leased_by = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    updated_at = clock_timestamp()
                WHERE id = %s
                  AND lease_token = %s
                  AND state = 'pending'
                  AND send_in_flight = FALSE
                  AND expires_at > clock_timestamp()
                """,
                (delay, _bounded(error_code), delivery_id, lease_token),
            )
            updated = cur.rowcount == 1
            if updated:
                cur.execute("SELECT pg_notify(%s, %s)", (_NOTIFY_CHANNEL, delivery_id))
        conn.commit()
    return updated


def cancel_escalations(alert_id: str, *, cursor=None) -> int:
    """Cancel unstarted and leased escalation work, fencing stale workers."""
    if cursor is not None:
        return _cancel_escalations(cursor, alert_id)
    with _get_conn() as conn:
        with conn.cursor() as cur:
            count = _cancel_escalations(cur, alert_id)
        conn.commit()
    return count


def cancel_escalations_many(alert_ids: list[str], *, cursor=None) -> int:
    """Cancel a bounded set of escalation obligations in one SQL update."""
    if not alert_ids:
        return 0
    if cursor is not None:
        return _cancel_escalations_many(cursor, alert_ids)
    with _get_conn() as conn:
        with conn.cursor() as cur:
            count = _cancel_escalations_many(cur, alert_ids)
        conn.commit()
    return count


def _cancel_escalations(cursor, alert_id: str) -> int:
    return _cancel_escalations_many(cursor, [alert_id])


def _cancel_escalations_many(cursor, alert_ids: list[str]) -> int:
    cursor.execute(
        """
        UPDATE alert_delivery_outbox
        SET state = 'cancelled',
            terminal_reason = 'alert_inactive',
            last_acceptance_unknown = last_acceptance_unknown OR send_in_flight,
            ever_acceptance_unknown = ever_acceptance_unknown OR send_in_flight,
            last_error_code = CASE
                WHEN send_in_flight THEN 'cancelled_in_flight'
                ELSE last_error_code
            END,
            send_in_flight = FALSE,
            leased_by = NULL,
            lease_token = NULL,
            lease_expires_at = NULL,
            updated_at = clock_timestamp()
        WHERE alert_id = ANY(%s)
          AND kind = 'escalation'
          AND state = 'pending'
        """,
        (alert_ids,),
    )
    return cursor.rowcount


def cancel_inactive_escalations() -> int:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE alert_delivery_outbox AS delivery
                SET state = 'cancelled',
                    terminal_reason = 'alert_inactive',
                    last_acceptance_unknown = (
                        delivery.last_acceptance_unknown OR delivery.send_in_flight
                    ),
                    ever_acceptance_unknown = (
                        delivery.ever_acceptance_unknown OR delivery.send_in_flight
                    ),
                    last_error_code = CASE
                        WHEN delivery.send_in_flight THEN 'cancelled_in_flight'
                        ELSE delivery.last_error_code
                    END,
                    send_in_flight = FALSE,
                    leased_by = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    updated_at = clock_timestamp()
                FROM alerts
                WHERE delivery.alert_id = alerts.id
                  AND delivery.kind = 'escalation'
                  AND delivery.state = 'pending'
                  AND alerts.status <> 'active'
                """
            )
            count = cur.rowcount
        conn.commit()
    return count


def requeue_terminal(
    delivery_id: str,
    *,
    allow_ambiguous: bool = False,
    max_replay_age_seconds: float = 24 * 60 * 60,
) -> bool:
    """Age-bound operator replay after configuration is repaired."""
    lifetime = _delivery_lifetime_seconds()
    replay_age = _safe_seconds(
        max_replay_age_seconds,
        default=24 * 60 * 60,
        minimum=1.0,
        maximum=30 * 24 * 60 * 60,
    )
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE alert_delivery_outbox
                SET state = 'pending',
                    claim_count = 0,
                    internal_failure_count = 0,
                    attempt_count = 0,
                    eligible_at = clock_timestamp(),
                    expires_at = clock_timestamp() + (%s * interval '1 second'),
                    next_attempt_at = clock_timestamp(),
                    leased_by = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL,
                    last_attempt_at = NULL,
                    last_error_code = NULL,
                    last_error_message = NULL,
                    last_acceptance_unknown = FALSE,
                    send_in_flight = FALSE,
                    terminal_reason = NULL,
                    terminal_at = NULL,
                    updated_at = clock_timestamp()
                WHERE id = %s
                  AND state = 'terminal'
                  AND updated_at >= clock_timestamp() - (%s * interval '1 second')
                  AND (%s OR NOT ever_acceptance_unknown)
                """,
                (lifetime, delivery_id, replay_age, bool(allow_ambiguous)),
            )
            updated = cur.rowcount == 1
            if updated:
                cur.execute("SELECT pg_notify(%s, %s)", (_NOTIFY_CHANNEL, delivery_id))
        conn.commit()
    return updated


def cleanup_completed(
    *,
    delivered_days: int | None = None,
    terminal_days: int | None = None,
    batch_size: int = 2000,
) -> int:
    """Bound outbox history without ever deleting pending/leased work."""
    delivered_days = delivered_days or int(os.getenv("ALERT_DELIVERY_RETENTION_DAYS", "14"))
    terminal_days = terminal_days or int(os.getenv("ALERT_DELIVERY_TERMINAL_RETENTION_DAYS", "90"))
    batch_size = max(1, min(int(batch_size), 10000))
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH expired AS (
                    SELECT id
                    FROM alert_delivery_outbox
                    WHERE (
                        state IN ('delivered', 'cancelled')
                        AND updated_at < clock_timestamp() - (%s * interval '1 day')
                    ) OR (
                        state = 'terminal'
                        AND updated_at < clock_timestamp() - (%s * interval '1 day')
                    )
                    ORDER BY updated_at, id
                    LIMIT %s
                )
                DELETE FROM alert_delivery_outbox AS delivery
                USING expired
                WHERE delivery.id = expired.id
                """,
                (max(1, delivered_days), max(1, terminal_days), batch_size),
            )
            count = cur.rowcount
        conn.commit()
    return count


def get_stats() -> dict:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE state = 'pending') AS pending,
                    COUNT(*) FILTER (
                        WHERE state = 'pending'
                          AND next_attempt_at <= clock_timestamp()
                          AND expires_at > clock_timestamp()
                    ) AS due,
                    COUNT(*) FILTER (
                        WHERE state = 'pending' AND next_attempt_at > clock_timestamp()
                    ) AS scheduled,
                    COUNT(*) FILTER (
                        WHERE state = 'pending' AND lease_token IS NOT NULL
                          AND lease_expires_at > clock_timestamp()
                    ) AS leased,
                    COUNT(*) FILTER (
                        WHERE state = 'pending' AND lease_token IS NOT NULL
                          AND lease_expires_at <= clock_timestamp()
                    ) AS expired_leases,
                    COUNT(*) FILTER (WHERE state = 'delivered') AS delivered,
                    COUNT(*) FILTER (WHERE state = 'terminal') AS terminal,
                    COUNT(*) FILTER (WHERE state = 'cancelled') AS cancelled,
                    COUNT(*) FILTER (WHERE ever_acceptance_unknown) AS ambiguous_history,
                    EXTRACT(EPOCH FROM (
                        clock_timestamp() - MIN(next_attempt_at) FILTER (
                            WHERE state = 'pending'
                              AND next_attempt_at <= clock_timestamp()
                              AND expires_at > clock_timestamp()
                        )
                    )) AS oldest_due_age_seconds
                FROM alert_delivery_outbox
                """
            )
            row = cur.fetchone()
    oldest_due = row["oldest_due_age_seconds"]
    return {
        "pending": int(row["pending"] or 0),
        "due": int(row["due"] or 0),
        "scheduled": int(row["scheduled"] or 0),
        "leased": int(row["leased"] or 0),
        "expired_leases": int(row["expired_leases"] or 0),
        "delivered": int(row["delivered"] or 0),
        "terminal": int(row["terminal"] or 0),
        "cancelled": int(row["cancelled"] or 0),
        "ambiguous_history": int(row["ambiguous_history"] or 0),
        "oldest_due_age_seconds": round(float(oldest_due), 3) if oldest_due is not None else None,
        # Compatibility alias; unlike the old value this excludes future work.
        "oldest_pending_age_seconds": round(float(oldest_due), 3) if oldest_due is not None else None,
    }


def get_for_alert(alert_id: str) -> list[dict]:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM alert_delivery_outbox
                WHERE alert_id = %s
                ORDER BY kind, target_key
                """,
                (alert_id,),
            )
            rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]


def get_delivery(delivery_id: str) -> dict | None:
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM alert_delivery_outbox WHERE id = %s", (delivery_id,))
            row = cur.fetchone()
    return _row_to_dict(row) if row else None


def list_deliveries(
    *,
    state: str | None = None,
    alert_id: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Return recent delivery rows for a privileged, redacted API projection."""
    if state is not None and state not in {"pending", "delivered", "terminal", "cancelled"}:
        raise ValueError("invalid delivery state")
    limit = max(1, min(int(limit), 500))
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, alert_id, kind, channel, priority, state,
                       claim_count, internal_failure_count, attempt_count,
                       eligible_at, expires_at, next_attempt_at, last_attempt_at,
                       last_error_code, last_acceptance_unknown,
                       ever_acceptance_unknown, terminal_reason, delivered_at,
                       terminal_at, created_at, updated_at
                FROM alert_delivery_outbox
                WHERE (%s::text IS NULL OR state = %s::text)
                  AND (%s::text IS NULL OR alert_id = %s::text)
                ORDER BY updated_at DESC, id
                LIMIT %s
                """,
                (state, state, alert_id, alert_id, limit),
            )
            rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_seconds(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _bounded(value: object, limit: int = 200) -> str | None:
    if value is None:
        return None
    return str(value).replace("\r", " ").replace("\n", " ")[:limit]


def cur_rowcount(cursor) -> int:
    return max(0, int(cursor.rowcount or 0))


def _row_to_dict(row) -> dict:
    result = dict(row)
    context = result.get("context")
    if not isinstance(context, dict):
        result["context"] = {}
    return result
