"""Typed, provider-safe outcomes for external alert delivery."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum


class DeliveryDisposition(str, Enum):
    DELIVERED = "delivered"
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    SKIPPED = "skipped"


def stable_delivery_identity(alert: dict) -> str:
    """Return a fleet-safe identity that stays stable only for one obligation."""
    obligation = alert.get("deliveryId") or f"{alert.get('id', 'unknown')}:initial"
    raw = "\x1f".join(
        str(value or "")
        for value in (
            obligation,
            alert.get("id"),
            alert.get("timestamp"),
            alert.get("cameraId"),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderDeliveryResult:
    """A sanitized provider outcome with an explicit retry contract."""

    disposition: DeliveryDisposition
    message: str
    error_code: str | None = None
    provider_status: int | str | None = None
    retry_after_seconds: float | None = None
    acceptance_unknown: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, DeliveryDisposition):
            raise TypeError("disposition must be a DeliveryDisposition")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("delivery result message must be non-empty")
        if type(self.acceptance_unknown) is not bool:
            raise TypeError("acceptance_unknown must be a bool")
        if self.acceptance_unknown and self.disposition is not DeliveryDisposition.RETRYABLE:
            raise ValueError("acceptance_unknown is valid only for retryable delivery")
        if self.retry_after_seconds is None:
            return
        if self.disposition is not DeliveryDisposition.RETRYABLE:
            raise ValueError("retry_after_seconds is valid only for retryable delivery")
        delay = float(self.retry_after_seconds)
        if isinstance(self.retry_after_seconds, bool) or not math.isfinite(delay) or delay < 0:
            raise ValueError("retry_after_seconds must be finite and non-negative")
        object.__setattr__(
            self,
            "retry_after_seconds",
            delay,
        )

    @property
    def success(self) -> bool:
        return self.disposition is DeliveryDisposition.DELIVERED

    @property
    def retryable(self) -> bool:
        return self.disposition is DeliveryDisposition.RETRYABLE

    def to_dispatch_dict(self, channel: str) -> dict:
        result = {
            "channel": channel,
            "success": self.success,
            "status": self.disposition.value,
            "message": self.message,
        }
        if self.error_code is not None:
            result["errorCode"] = self.error_code
        if self.provider_status is not None:
            result["providerStatus"] = self.provider_status
        if self.retry_after_seconds is not None:
            result["retryAfterSeconds"] = self.retry_after_seconds
        if self.acceptance_unknown:
            result["acceptanceUnknown"] = True
        return result


def parse_retry_after(value: object, *, now: datetime | None = None) -> float | None:
    """Parse non-negative delta-seconds or an HTTP-date."""
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        reference = now or datetime.now(timezone.utc)
        seconds = (retry_at.astimezone(timezone.utc) - reference).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)
