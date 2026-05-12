"""Redis caching layer for StarLens.

Caches expensive computations (ephemeris) and LLM responses to avoid
redundant work. Gracefully degrades when Redis is unavailable.
"""

import hashlib
import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_redis = None
_available = False
_lock = threading.Lock()


def _connect() -> None:
    """Lazy-connect to Redis on first use."""
    global _redis, _available  # noqa: PLW0603  # pylint: disable=global-statement

    if _redis is not None:
        return

    with _lock:
        # Double-check after acquiring lock
        if _redis is not None:
            return

        from .settings import settings  # pylint: disable=import-outside-toplevel

        url = settings.redis.url
        if not url:
            _available = False
            return

        try:
            import redis  # pylint: disable=import-outside-toplevel

            _redis = redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            _redis.ping()
            _available = True
            logger.info("Redis cache connected: %s", url)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _redis = None
            _available = False
            logger.warning("Redis unavailable — caching disabled (%s: %s)", type(exc).__name__, exc)


def _cache_key(prefix: str, *parts: Any) -> str:
    """Build a deterministic cache key from arbitrary inputs."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"starlens:{prefix}:{digest}"


def get(prefix: str, *parts: Any) -> Any | None:
    """Retrieve a cached value, or None on miss / unavailable."""
    _connect()
    if not _available or _redis is None:
        return None
    try:
        data: str | None = _redis.get(_cache_key(prefix, *parts))  # type: ignore[assignment]
        if data is not None:
            return json.loads(data)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return None


def put(prefix: str, *parts: Any, value: Any, ttl: int = 300) -> None:
    """Store a value in cache with a TTL (seconds)."""
    _connect()
    if not _available or _redis is None:
        return
    try:
        _redis.setex(
            _cache_key(prefix, *parts),
            ttl,
            json.dumps(value, default=str),
        )
    except Exception:  # pylint: disable=broad-exception-caught
        pass
