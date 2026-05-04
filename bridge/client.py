import asyncio
import json
import uuid
from collections import deque
from contextlib import suppress
from typing import Any, Deque, Dict, Optional


class BridgeError(RuntimeError):
    """Raised when the Minecraft Bridge cannot satisfy a request."""


class BridgeClient:
    """
    Async JSON-over-TCP client for the Minecraft Bridge mod.

    The bridge also broadcasts game_state snapshots; request/response matching is
    therefore done with request_id when the mod supports it, with FIFO fallback for
    older bridge jars.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 25575,
        auth_token: str = "",
        timeout: float = 5.0,
        reconnect_delay: float = 3.0,
        max_reconnect_delay: float = 12.0,
        max_line_bytes: int = 64_000_000,
    ) -> None:
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max(reconnect_delay, max_reconnect_delay)
        self.max_line_bytes = max_line_bytes
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._pending: Dict[str, asyncio.Future] = {}
        self._pending_fifo: Deque[asyncio.Future] = deque()
        self._broadcasts: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._system_messages: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=20)
        self.latest_state: Optional[Dict[str, Any]] = None
        self.connected = False
        self._disconnect_reason = "Bridge disconnected"

    async def connect(self) -> None:
        await self.close()
        self.reader, self.writer = await asyncio.open_connection(
            self.host,
            self.port,
            limit=self.max_line_bytes + 1,
        )
        self.connected = True
        self._disconnect_reason = "Bridge disconnected"
        if self.auth_token:
            await self._write_json({"auth": self.auth_token})
        self._reader_task = asyncio.create_task(self._read_loop(), name="minecraft-bridge-reader")

    async def ensure_connected(self) -> None:
        if self.connected and self.writer is not None and not self.writer.is_closing():
            return
        delay = self.reconnect_delay
        while True:
            try:
                await self.connect()
                return
            except OSError:
                await asyncio.sleep(delay)
                delay = min(self.max_reconnect_delay, delay * 1.5)

    async def close(self) -> None:
        self.connected = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._reader_task
            self._reader_task = None
        if self.writer is not None:
            self.writer.close()
            with suppress(Exception):
                await self.writer.wait_closed()
        self.reader = None
        self.writer = None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(BridgeError(self._disconnect_reason))
        self._pending.clear()
        self._pending_fifo.clear()

    async def request(self, action: str, **payload: Any) -> Dict[str, Any]:
        await self.ensure_connected()
        request_id = payload.pop("request_id", str(uuid.uuid4()))
        command = {"action": action, "request_id": request_id, **payload}
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        self._pending_fifo.append(future)
        try:
            await self._write_json(command)
        except Exception as exc:
            self._pending.pop(request_id, None)
            with suppress(ValueError):
                self._pending_fifo.remove(future)
            if not future.done():
                future.set_exception(BridgeError(f"Bridge write failed: {exc}"))
            await self.close()
            raise BridgeError(f"Bridge write failed: {exc}") from exc
        try:
            response = await asyncio.wait_for(future, timeout=self.timeout)
        finally:
            self._pending.pop(request_id, None)
            with suppress(ValueError):
                self._pending_fifo.remove(future)
        if response.get("status") == "error":
            raise BridgeError(f"{response.get('code', 'ERROR')}: {response.get('message', response)}")
        return response

    async def next_broadcast(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        try:
            return await asyncio.wait_for(self._broadcasts.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _write_json(self, message: Dict[str, Any]) -> None:
        if self.writer is None:
            raise BridgeError("Bridge is not connected")
        self.writer.write((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
        await self.writer.drain()

    async def _read_loop(self) -> None:
        assert self.reader is not None
        try:
            while True:
                try:
                    line = await self.reader.readline()
                except ValueError as exc:
                    self._disconnect_reason = f"Bridge read failed: {exc}"
                    self._put_nowait_drop_oldest(
                        self._system_messages,
                        {"status": "error", "message": self._disconnect_reason},
                    )
                    break
                if not line:
                    break
                if len(line) > self.max_line_bytes:
                    self._put_nowait_drop_oldest(
                        self._system_messages,
                        {"status": "error", "message": "Bridge message exceeded max_line_bytes"},
                    )
                    continue
                try:
                    message = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                self._route_message(message)
        finally:
            self.connected = False
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(BridgeError(self._disconnect_reason))
            self._pending.clear()
            self._pending_fifo.clear()

    def _route_message(self, message: Dict[str, Any]) -> None:
        if message.get("type") == "game_state":
            self.latest_state = message
            self._put_nowait_drop_oldest(self._broadcasts, message)
            return

        request_id = message.get("request_id")
        if request_id is not None:
            future = self._pending.get(str(request_id))
            if future is not None and not future.done():
                future.set_result(message)
                return

        if "action_id" in message and self._pending_fifo:
            while self._pending_fifo:
                future = self._pending_fifo.popleft()
                if not future.done():
                    future.set_result(message)
                    return

        self._put_nowait_drop_oldest(self._system_messages, message)

    @staticmethod
    def _put_nowait_drop_oldest(queue: asyncio.Queue, message: Dict[str, Any]) -> None:
        if queue.full():
            with suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(message)
