"""FIFO (named-pipe) helpers for lightweight inter-process communication.

Each daemon owns a *command* FIFO and a *status* file::

    /run/ari/tts_cmd      <-- clients write commands here
    /run/ari/tts_status    <-- daemon writes status/responses here

Naming convention: the status path is derived automatically by replacing the
``_cmd`` suffix in the FIFO path with ``_status``.

Usage -- daemon side::

    srv = FifoServer("/run/ari/tts_cmd")
    for cmd in srv:          # blocks until a command arrives
        handle(cmd)
        srv.reply("ok")      # write to the status file

Usage -- client side::

    cli = FifoClient("/run/ari/tts_cmd")
    status = cli.send("speak Hello world")   # returns status string
"""

from __future__ import annotations

import errno
import logging
import os
import time
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


def _status_path_for(fifo_path: Path) -> Path:
    """Derive the status file path from the command FIFO path.

    Replaces a trailing ``_cmd`` with ``_status``.  If the name does not end
    with ``_cmd``, appends ``_status`` as a sibling file instead.
    """
    name = fifo_path.name
    if name.endswith("_cmd"):
        return fifo_path.with_name(name[:-4] + "_status")
    return fifo_path.with_name(name + "_status")


class FifoServer:
    """Daemon-side FIFO reader.

    Creates the named pipe (if it doesn't already exist) and exposes a
    blocking iterator that yields one command string per write from a client.

    Parameters
    ----------
    path:
        Filesystem path for the command FIFO.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.status_path = _status_path_for(self.path)
        self._ensure_fifo()

    # -- public API --------------------------------------------------------

    def read_command(self) -> str:
        """Block until a command is written to the FIFO, then return it.

        Returns the command string with leading/trailing whitespace stripped.
        """
        with open(self.path, "r") as fh:
            data = fh.read().strip()
        log.debug("FifoServer read: %r", data)
        return data

    def reply(self, status: str) -> None:
        """Write a status/response string to the status file.

        Overwrites any previous content so the client always sees the latest
        response.
        """
        self.status_path.write_text(status + "\n")
        log.debug("FifoServer replied: %r", status)

    def __iter__(self) -> Iterator[str]:
        """Yield commands forever (blocking between each)."""
        while True:
            cmd = self.read_command()
            if cmd:
                yield cmd

    def cleanup(self) -> None:
        """Remove the FIFO and status file (best-effort)."""
        for p in (self.path, self.status_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    # -- internals ---------------------------------------------------------

    def _ensure_fifo(self) -> None:
        """Create the FIFO if it doesn't already exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkfifo(self.path)
            log.info("Created FIFO %s", self.path)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                log.debug("FIFO already exists: %s", self.path)
            else:
                raise


class FifoClient:
    """Client-side FIFO writer with retry logic.

    Parameters
    ----------
    path:
        Filesystem path for the command FIFO (must already exist).
    retries:
        Number of write attempts before giving up (default 3).
    retry_delay:
        Seconds to wait between retries (default 0.3).
    status_timeout:
        Seconds to wait for a status reply before giving up (default 5.0).
    """

    def __init__(
        self,
        path: str | Path,
        retries: int = 3,
        retry_delay: float = 0.3,
        status_timeout: float = 5.0,
    ) -> None:
        self.path = Path(path)
        self.status_path = _status_path_for(self.path)
        self.retries = retries
        self.retry_delay = retry_delay
        self.status_timeout = status_timeout

    def send(self, command: str) -> str:
        """Send *command* to the daemon and return its status reply.

        Retries up to ``self.retries`` times on transient write failures.
        Raises ``RuntimeError`` if all attempts fail or the daemon never
        writes a status reply within ``self.status_timeout`` seconds.
        """
        # Clear any stale status before sending.
        self._clear_status()

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                self._write_command(command)
                break
            except OSError as exc:
                last_exc = exc
                log.warning(
                    "FIFO write attempt %d/%d failed: %s",
                    attempt,
                    self.retries,
                    exc,
                )
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
        else:
            raise RuntimeError(
                f"Failed to write to FIFO {self.path} after "
                f"{self.retries} attempts"
            ) from last_exc

        return self._wait_for_status()

    # -- internals ---------------------------------------------------------

    def _write_command(self, command: str) -> None:
        """Open the FIFO for writing and send the command string."""
        with open(self.path, "w") as fh:
            fh.write(command + "\n")
            fh.flush()
        log.debug("FifoClient wrote: %r", command)

    def _clear_status(self) -> None:
        """Remove the status file so we don't read a stale reply."""
        try:
            self.status_path.unlink()
        except FileNotFoundError:
            pass

    def _wait_for_status(self) -> str:
        """Poll for the status file until it appears or we time out."""
        deadline = time.monotonic() + self.status_timeout
        while time.monotonic() < deadline:
            if self.status_path.exists():
                text = self.status_path.read_text().strip()
                if text:
                    log.debug("FifoClient got status: %r", text)
                    return text
            time.sleep(0.05)
        raise RuntimeError(
            f"No status reply within {self.status_timeout}s "
            f"(expected at {self.status_path})"
        )
