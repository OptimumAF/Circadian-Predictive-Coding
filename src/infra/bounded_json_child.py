"""Owned, bounded JSON-line child transport using only standard-library IO.

Readers enforce byte/message caps; a writer thread bounds stalled stdin writes.
Why join before closing pipes: closing a buffered stream held by a blocked reader
can hang on its lock. Reap the owned process first, then join within one deadline.
Inherited descendant pipes can prevent cleanup; report failure rather than claim
success. No process-tree termination, registration, native state or authority.
"""

import json
from math import isfinite
from pathlib import Path
from queue import Empty, Full, Queue
import subprocess
import sys
from threading import Event, Lock, Thread, TIMEOUT_MAX
from time import monotonic
from typing import Any


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or not 0 < value < sys.maxsize:
        raise ValueError(f"{name} must be a bounded positive native integer")
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate child JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"nonfinite child JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not isfinite(number):
        raise ValueError("nonfinite child JSON number")
    return number


def _json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_nonfinite,
            parse_float=_finite_float,
        )
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid child JSON object") from error
    if type(value) is not dict:
        raise ValueError("child JSON message must be an object")
    return value


def _configuration(command, cwd, env, timeout_seconds, bounds):
    if (
        type(command) is not list
        or not command
        or any(type(part) is not str or not part or "\x00" in part for part in command)
    ):
        raise ValueError("child command requires a nonempty list of nonempty native strings")
    if (
        sum(len(part) + 1 for part in command) > 65536
        or sum(len(part.encode("utf-8")) + 1 for part in command) > 65536
    ):
        raise ValueError("child command exceeds 65536 bytes")
    if (
        not isinstance(cwd, Path)
        or type(env) is not dict
        or any(
            type(k) is not str
            or not k
            or "\x00" in k
            or "=" in k
            or type(v) is not str
            or "\x00" in v
            for k, v in env.items()
        )
    ):
        raise ValueError("child requires a Path cwd and exact string environment")
    if type(timeout_seconds) not in (int, float):
        raise ValueError("timeout_seconds must be native positive finite seconds")
    try:
        timeout = float(timeout_seconds)
    except OverflowError:
        raise ValueError("timeout_seconds must be finite") from None
    if not isfinite(timeout) or not 0 < timeout <= TIMEOUT_MAX:
        raise ValueError("timeout_seconds exceeds supported positive finite timeout")
    for name, value in bounds.items():
        _positive_int(value, name)
    return list(command), Path(cwd), dict(env), timeout


class BoundedJsonChild:
    """Single-owner child operations with finite deadlines and terminal faults.

    A failed operation has its own timeout plus a separate cleanup timeout;
    timeout_seconds is not a combined operation-and-cleanup deadline.
    """

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: float = 5.0,
        max_line_bytes: int = 65536,
        max_output_bytes: int = 262144,
        max_stderr_bytes: int = 16384,
        max_messages: int = 16,
    ) -> None:
        command, cwd, env, self._timeout = _configuration(
            command,
            cwd,
            env,
            timeout_seconds,
            dict(
                max_line_bytes=max_line_bytes,
                max_output_bytes=max_output_bytes,
                max_stderr_bytes=max_stderr_bytes,
                max_messages=max_messages,
            ),
        )
        self._line_cap, self._output_cap = max_line_bytes, max_output_bytes
        self._stderr_cap, self._message_cap = max_stderr_bytes, max_messages
        self._inbox: Queue[dict[str, Any]] = Queue(maxsize=max_messages)
        self._writes: Queue[Any] = Queue(maxsize=1)
        self._gate = Lock()
        self._failure: BaseException | None = None
        self._stderr = bytearray()
        self._eof = Event()
        self._closed = False
        self._writer_stopped = False
        self._threads: list[Thread] = []
        self._process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for target, name in (
                (self._read_stdout, "stdout"),
                (self._read_stderr, "stderr"),
                (self._write_stdin, "stdin"),
            ):
                thread = Thread(target=target, name=f"bounded-child-{self.pid}-{name}", daemon=True)
                thread.start()
                self._threads.append(thread)
        except BaseException as error:
            self._abort(error)
            raise

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    @property
    def stderr(self) -> bytes:
        with self._gate:
            return bytes(self._stderr)

    def _fail(self, error: BaseException) -> None:
        with self._gate:
            if self._failure is None:
                self._failure = error

    def _require_ready(self) -> None:
        with self._gate:
            failure = self._failure
        if failure is not None:
            raise failure
        if self._closed:
            raise ValueError("child session is closed")

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("bounded child operation timed out")
        return remaining

    def _read_stdout(self) -> None:
        try:
            pipe = self._process.stdout
            assert pipe is not None
            total = messages = 0
            while True:
                raw = pipe.readline(self._line_cap + 1)
                if not raw:
                    return
                total += len(raw)
                messages += 1
                if (
                    len(raw) > self._line_cap
                    or total > self._output_cap
                    or messages > self._message_cap
                ):
                    raise ValueError("child stdout exceeded byte or message bounds")
                if not raw.endswith(b"\n"):
                    raise ValueError("partial child JSON line")
                self._inbox.put_nowait(_json_object(raw))
        except Full:
            self._fail(ValueError("child JSON message queue exceeded its bound"))
        except BaseException as error:
            self._fail(error)
        finally:
            self._eof.set()

    def _read_stderr(self) -> None:
        try:
            pipe = self._process.stderr
            assert pipe is not None
            while True:
                with self._gate:
                    remaining = self._stderr_cap - len(self._stderr)
                raw = pipe.read(min(4096, remaining + 1))
                if not raw:
                    return
                with self._gate:
                    self._stderr.extend(raw[:remaining])
                if len(raw) > remaining:
                    raise ValueError("child stderr exceeded its byte bound")
        except BaseException as error:
            self._fail(error)

    def _write_stdin(self) -> None:
        while True:
            request = self._writes.get()
            if request is None:
                return
            payload, done, outcome = request
            try:
                pipe = self._process.stdin
                assert pipe is not None
                count = pipe.write(payload)
                if type(count) is not int or count != len(payload):
                    raise OSError("child stdin write was not complete")
                pipe.flush()
            except BaseException as error:
                outcome.append(error)
                self._fail(error)
            finally:
                done.set()

    def _abort(self, primary: BaseException) -> None:
        self._fail(primary)
        try:
            self.close()
        except BaseException as cleanup:
            BaseException.add_note(primary, f"owned child cleanup failed: {cleanup!r}")

    def send(self, message: dict[str, Any]) -> None:
        try:
            self._require_ready()
            if type(message) is not dict:
                raise ValueError("child command must be an exact JSON object")
            try:
                payload = (
                    json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                )
            except (ValueError, TypeError, RecursionError, UnicodeError) as error:
                raise ValueError("invalid child JSON command") from error
            if len(payload) > self._line_cap:
                raise ValueError("child command exceeds the JSON line byte bound")
            deadline = monotonic() + self._timeout
            done = Event()
            outcome: list[BaseException] = []
            self._writes.put_nowait((payload, done, outcome))
            while not done.wait(min(0.01, self._remaining(deadline))):
                self._require_ready()
            if outcome:
                raise outcome[0]
            self._require_ready()
        except BaseException as error:
            self._abort(error)
            raise

    def receive(self) -> dict[str, Any]:
        try:
            deadline = monotonic() + self._timeout
            while True:
                self._require_ready()
                try:
                    message = self._inbox.get(timeout=min(0.01, self._remaining(deadline)))
                    self._require_ready()
                    return message
                except Empty:
                    self._require_ready()
                    if self._eof.is_set():
                        raise ValueError(
                            "child stdout ended before the requested JSON message"
                        ) from None
        except BaseException as error:
            self._abort(error)
            raise

    def _join_and_close(self, deadline: float) -> None:
        if not self._writer_stopped:
            try:
                self._writes.put(None, timeout=self._remaining(deadline))
            except Full:
                raise TimeoutError("owned child writer did not accept shutdown") from None
            self._writer_stopped = True
        for thread in self._threads:
            thread.join(timeout=self._remaining(deadline))
            if thread.is_alive():
                raise TimeoutError("owned child IO thread did not stop; inherited pipes may remain")
        for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
            if pipe is not None:
                pipe.close()
        self._closed = True

    def wait(self) -> int:
        if self._closed:
            with self._gate:
                failure = self._failure
            if failure is not None:
                raise failure
            code = self.returncode
            if code is None:
                raise RuntimeError("closed child was not reaped")
            return code
        try:
            self._require_ready()
            deadline = monotonic() + self._timeout
            try:
                code = self._process.wait(timeout=self._remaining(deadline))
            except subprocess.TimeoutExpired as error:
                raise TimeoutError("bounded child wait timed out") from error
            self._join_and_close(deadline)
            with self._gate:
                failure = self._failure
            if failure is not None:
                raise failure
            if code != 0 or self.stderr:
                raise RuntimeError(f"child exit failed: returncode={code}, stderr={self.stderr!r}")
            return code
        except BaseException as error:
            self._abort(error)
            raise

    def kill(self) -> int:
        self.close()
        code = self.returncode
        if code is None:
            raise RuntimeError("owned child was not reaped")
        return code

    def close(self) -> None:
        if self._closed:
            return
        deadline = monotonic() + self._timeout
        if self._process.poll() is None:
            self._process.kill()
        try:
            self._process.wait(timeout=self._remaining(deadline))
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("owned child could not be reaped within cleanup deadline") from error
        self._join_and_close(deadline)

    def __enter__(self) -> "BoundedJsonChild":
        try:
            self._require_ready()
            return self
        except BaseException as error:
            # A failed __enter__ never registers __exit__ with ExitStack.
            self._abort(error)
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is None:
            self.close()
        else:
            self._abort(exc)
