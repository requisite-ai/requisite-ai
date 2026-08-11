"""
Redis-backed conversation memory.

Persists each session's messages as a Redis **list** (``RPUSH``/``LRANGE``)
under a namespaced key -- a natural fit for an append-only conversation
log, and avoids the read-modify-write race a single JSON-blob-per-session
key would have under concurrent writers. Unlike
:class:`~requisite.memory.sqlite.SQLiteMemory`, sessions are shared
across processes/machines, not just a single local file.

Install with: ``pip install requisite-ai[redis]``
"""

from __future__ import annotations

import os
from typing import Any, Optional

from requisite.core.exceptions import ConfigurationException, MemoryException
from requisite.core.interfaces import Message
from requisite.memory.base import BaseMemory

# Deliberately 127.0.0.1, not "localhost": on Windows, resolving
# "localhost" tries the IPv6 loopback (::1) first and stalls for ~2s
# before falling back to IPv4 -- confirmed by direct measurement against
# a local Redis-compatible server. That's a real, silent tax on every
# fresh connection for the zero-config default this backend is supposed
# to have. 127.0.0.1 skips DNS/AF resolution entirely, so it's strictly
# faster everywhere and never slower.
_DEFAULT_URL = "redis://127.0.0.1:6379/0"


class RedisMemory(BaseMemory):
    """Conversation memory backed by a Redis server.

    Parameters
    ----------
    url:
        Redis connection URL, e.g. ``"redis://127.0.0.1:6379/0"``. If not
        provided, falls back to the ``REDIS_URL`` environment variable,
        then ``"redis://127.0.0.1:6379/0"``. Note this fallback is
        implemented by ``RedisMemory`` itself -- unlike the Pinecone/OpenAI
        SDKs, ``redis-py`` does not read this env var on its own. Uses
        ``127.0.0.1`` rather than ``localhost`` deliberately -- see the
        module-level comment on ``_DEFAULT_URL``.
    key_prefix:
        Prefix applied to every Redis key this backend touches, so
        sessions don't collide with unrelated keys in a shared Redis
        instance.
    ttl_seconds:
        If set, a session's key is refreshed with ``EXPIRE`` on every
        append, so inactive sessions eventually expire instead of
        accumulating forever. ``None`` (default) means no expiry.

    Examples
    --------
    >>> memory = RedisMemory(url="redis://127.0.0.1:6379/0")  # doctest: +SKIP
    >>> memory.append("session-1", Message.user("hello"))  # doctest: +SKIP
    >>> memory.load("session-1")  # doctest: +SKIP
    [Message(role=<Role.USER: 'user'>, content='hello', ...)]
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        key_prefix: str = "requisite:memory:",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._url = url or os.environ.get("REDIS_URL", _DEFAULT_URL)
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds
        self._client: Any = None
        self._aclient: Any = None

    @property
    def name(self) -> str:
        return "redis"

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - exercised only without the dep
                raise ConfigurationException(
                    "The 'redis' package is required for RedisMemory. "
                    "Install it with: pip install requisite-ai[redis]",
                ) from exc
            self._client = redis.Redis.from_url(self._url, decode_responses=True)
        return self._client

    def _get_async_client(self) -> Any:
        if self._aclient is None:
            try:
                import redis.asyncio as redis_asyncio
            except ImportError as exc:  # pragma: no cover - exercised only without the dep
                raise ConfigurationException(
                    "The 'redis' package is required for RedisMemory. "
                    "Install it with: pip install requisite-ai[redis]",
                ) from exc
            self._aclient = redis_asyncio.Redis.from_url(self._url, decode_responses=True)
        return self._aclient

    def load(self, session_id: str) -> list[Message]:
        client = self._get_client()
        try:
            raw = client.lrange(self._key(session_id), 0, -1)
        except Exception as exc:  # noqa: BLE001
            raise MemoryException(
                f"Redis load failed for session '{session_id}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc
        return [Message.model_validate_json(item) for item in raw]

    def append(self, session_id: str, message: Message) -> None:
        client = self._get_client()
        key = self._key(session_id)
        try:
            client.rpush(key, message.model_dump_json())
            if self._ttl_seconds is not None:
                client.expire(key, self._ttl_seconds)
        except Exception as exc:  # noqa: BLE001
            raise MemoryException(
                f"Redis append failed for session '{session_id}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc

    def clear(self, session_id: str) -> None:
        client = self._get_client()
        try:
            client.delete(self._key(session_id))
        except Exception as exc:  # noqa: BLE001
            raise MemoryException(
                f"Redis clear failed for session '{session_id}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc

    async def aload(self, session_id: str) -> list[Message]:
        client = self._get_async_client()
        try:
            raw = await client.lrange(self._key(session_id), 0, -1)
        except Exception as exc:  # noqa: BLE001
            raise MemoryException(
                f"Redis load failed for session '{session_id}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc
        return [Message.model_validate_json(item) for item in raw]

    async def aappend(self, session_id: str, message: Message) -> None:
        client = self._get_async_client()
        key = self._key(session_id)
        try:
            await client.rpush(key, message.model_dump_json())
            if self._ttl_seconds is not None:
                await client.expire(key, self._ttl_seconds)
        except Exception as exc:  # noqa: BLE001
            raise MemoryException(
                f"Redis append failed for session '{session_id}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc

    async def aclear(self, session_id: str) -> None:
        client = self._get_async_client()
        try:
            await client.delete(self._key(session_id))
        except Exception as exc:  # noqa: BLE001
            raise MemoryException(
                f"Redis clear failed for session '{session_id}': {exc}",
                backend=self.name,
                original_error=exc,
            ) from exc
