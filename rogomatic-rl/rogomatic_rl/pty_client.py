"""PTY client for running Rogue 5.4 in a 24x80 virtual terminal."""

from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import tempfile
import termios
from pathlib import Path
from typing import Optional

import pyte


DEFAULT_ROGUE_BIN = Path(__file__).resolve().parents[2] / "rogue5.4" / "rogue"


class RoguePTYClient:
    """Manages a Rogue 5.4 child process over a 24x80 pseudo-terminal."""

    ROWS = 24
    COLS = 80

    def __init__(
        self,
        rogue_bin: str | Path = DEFAULT_ROGUE_BIN,
        rogue_opts: str = "terse,name=rogomatic,jump",
        term: str = "vt100",
        step_timeout: float = 0.02,
        idle_timeout: float = 0.001,
        startup_timeout: float = 0.5,
    ) -> None:
        self.rogue_bin = str(rogue_bin)
        self.rogue_opts = rogue_opts
        self.term = term
        self.step_timeout = step_timeout
        self.idle_timeout = idle_timeout
        self.startup_timeout = startup_timeout

        self.screen = pyte.Screen(self.COLS, self.ROWS)
        self.stream = pyte.Stream(self.screen)

        self._master_fd: Optional[int] = None
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._workdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self.alive: bool = False

    def start(self) -> list[str]:
        """Start a fresh Rogue process and return any initial messages."""
        self.close()
        self.screen.reset()

        self._workdir = tempfile.TemporaryDirectory(prefix="rogomatic_")
        master_fd, slave_fd = pty.openpty()
        winsize = struct.pack("HHHH", self.ROWS, self.COLS, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        env = os.environ.copy()
        env["TERM"] = self.term
        env["ROGUEOPTS"] = self.rogue_opts
        env["ESCDELAY"] = "0"

        try:
            self._proc = subprocess.Popen(
                [self.rogue_bin],
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=self._workdir.name,
                env=env,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(slave_fd)

        self._master_fd = master_fd
        self.alive = True
        self._drain_output(first_timeout=self.startup_timeout)
        return self._dismiss_more_prompts()

    def send_key(self, key: str) -> list[str]:
        """Send a keystroke (or key sequence) to Rogue and return any messages."""
        if not self.alive or self._master_fd is None:
            return []

        try:
            os.write(self._master_fd, key.encode("ascii"))
        except OSError:
            self.alive = False
            return []

        self._drain_output(first_timeout=self.step_timeout)
        return self._dismiss_more_prompts()

    def get_lines(self) -> list[str]:
        """Return the current 24x80 virtual screen as 24 strings of length 80."""
        return [
            "".join(self.screen.buffer[row][col].data for col in range(self.COLS))
            for row in range(self.ROWS)
        ]

    def _dismiss_more_prompts(self, max_prompts: int = 10) -> list[str]:
        """Automatically dismiss '--More--' prompts while collecting top-line messages."""
        messages: list[str] = []
        for _ in range(max_prompts):
            top_line = "".join(
                self.screen.buffer[0][col].data for col in range(self.COLS)
            ).strip()
            if top_line:
                messages.append(top_line.replace("--More--", "").strip())
            if "--More--" not in top_line or not self.alive or self._master_fd is None:
                break
            try:
                os.write(self._master_fd, b" ")
            except OSError:
                self.alive = False
                break
            self._drain_output(first_timeout=self.step_timeout)
        return [m for m in messages if m]

    def _drain_output(self, first_timeout: float) -> None:
        """Read available PTY bytes into the pyte stream."""
        if self._master_fd is None:
            return

        timeout = first_timeout
        while True:
            rlist, _, _ = select.select([self._master_fd], [], [], timeout)
            if not rlist:
                break
            try:
                data = os.read(self._master_fd, 4096)
            except OSError:
                self.alive = False
                break
            if not data:
                self.alive = False
                break
            self.stream.feed(data.decode("latin-1", errors="replace"))
            timeout = self.idle_timeout

        if self._proc is not None and self._proc.poll() is not None:
            self.alive = False

    def close(self) -> None:
        """Terminate the Rogue process and release PTY/temp resources."""
        self.alive = False

        # Kill Rogue before closing the PTY master, and use SIGKILL so it
        # cannot react.  If the master closes first, Rogue's blocked read
        # fails and it auto-saves the game to ~/rogue.save (mdport.c
        # md_readchar), clobbering any real saved game there.
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    self._proc.wait(timeout=1.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._proc = None

        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None
