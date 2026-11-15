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
import time
from pathlib import Path
from typing import Optional

import pyte


DEFAULT_ROGUE_BIN = Path(__file__).resolve().parents[2] / "rogue5.4" / "rogue"
# Shared high score list for the agent's games (untracked; see .gitignore).
DEFAULT_SCORE_FILE = Path(__file__).resolve().parents[1] / "scores" / "rogue.scr"
# Name of the score file compiled into the Rogue binary, relative to its cwd.
_SCORE_FILE_NAME = "rogue.scr"


class _VT100Screen(pyte.Screen):
    """pyte.Screen with standard VT100 margin-clamping rules when DECOM is off.

    pyte unconditionally clamps cursor.y to ``margins`` in ``restore_cursor``,
    ``cursor_up``, and ``cursor_down`` even when Origin Mode (DECOM) is off.
    When ncurses uses ``csr`` (``\\x1b7\\x1b[top;botr\\x1b8``) with a cursor outside
    the new scroll region, clamping ``\\x1b8`` shifts the cursor by a row so the
    subsequent ``\\x1bM`` / ``\\n`` misses the margin boundary and fails to scroll.
    """

    def restore_cursor(self) -> None:
        saved_y = self.savepoints[-1].cursor.y if self.savepoints else None
        super().restore_cursor()
        if saved_y is not None and pyte.modes.DECOM not in self.mode:
            self.cursor.y = min(max(0, saved_y), self.lines - 1)

    def cursor_up(self, count: Optional[int] = None) -> None:
        top, _ = self.margins or pyte.screens.Margins(0, self.lines - 1)
        limit = top if self.cursor.y >= top else 0
        self.cursor.y = max(self.cursor.y - (count or 1), limit)

    def cursor_down(self, count: Optional[int] = None) -> None:
        _, bottom = self.margins or pyte.screens.Margins(0, self.lines - 1)
        limit = bottom if self.cursor.y <= bottom else self.lines - 1
        self.cursor.y = min(self.cursor.y + (count or 1), limit)


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
        score_file: Optional[str | Path] = None,
    ) -> None:
        self.rogue_bin = str(rogue_bin)
        self.rogue_opts = rogue_opts
        self.term = term
        self.step_timeout = step_timeout
        self.idle_timeout = idle_timeout
        self.startup_timeout = startup_timeout
        # When set, every game shares this score file and record_score()
        # can post to it.  When None, scores are discarded with the workdir.
        self.score_file: Optional[Path] = (
            Path(score_file).resolve() if score_file is not None else None
        )

        self.screen = _VT100Screen(self.COLS, self.ROWS)
        self.stream = pyte.Stream(self.screen)

        self._master_fd: Optional[int] = None
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._workdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self.alive: bool = False

    def start(self, seed: Optional[int] = None) -> list[str]:
        """Start a fresh Rogue process and return any initial messages.

        With ``seed``, Rogue generates that numbered dungeon (its ``SEED``
        environment variable) instead of a random one.  Rogue never posts
        a seeded game to the high score list.
        """
        self.close()
        self.screen.reset()

        self._workdir = tempfile.TemporaryDirectory(prefix="rogomatic_")
        if self.score_file is not None:
            # Rogue opens "rogue.scr" in its cwd at startup; point that at
            # the shared file (Rogue creates the target if it is missing).
            self.score_file.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(self.score_file, Path(self._workdir.name) / _SCORE_FILE_NAME)
        master_fd, slave_fd = pty.openpty()
        winsize = struct.pack("HHHH", self.ROWS, self.COLS, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        env = os.environ.copy()
        env["TERM"] = self.term
        env["ROGUEOPTS"] = self.rogue_opts
        env["ESCDELAY"] = "0"
        # Never inherit a seed from the caller's shell: an unseeded game
        # must be random and must be able to post its score.
        env.pop("SEED", None)
        if seed is not None:
            env["SEED"] = str(int(seed))

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

    def record_score(self, quit_game: bool = False, timeout: float = 2.0) -> bool:
        """Let Rogue post this game to the shared score file, then exit.

        Call at the tombstone, or with ``quit_game=True`` while the player is
        still alive to quit first (recorded as "quit").  Returns True if
        Rogue wrote the score and exited cleanly.  Rogue only lists games
        that beat an existing entry, so a zero-gold game is never shown.
        """
        if self.score_file is None or not self.alive or self._master_fd is None:
            return False

        # Rogue's own lock file does not exclude other writers, so hold a
        # real lock across its read-modify-write of the score file.
        lock_path = self.score_file.with_name(self.score_file.name + ".lock")
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            if quit_game:
                self.send_key("Q")
                if "really quit" not in self.get_lines()[0].lower():
                    return False
                self.send_key("y")

            screen_text = "\n".join(self.get_lines()).lower()
            if "press return to continue" not in screen_text:
                return False
            try:
                os.write(self._master_fd, b"\n")
            except OSError:
                self.alive = False
                return False

            deadline = time.monotonic() + timeout
            while self._proc is not None and self._proc.poll() is None:
                if time.monotonic() > deadline:
                    return False
                self._drain_output(first_timeout=self.step_timeout)
            self.alive = False
            return self._proc is not None and self._proc.returncode == 0

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
