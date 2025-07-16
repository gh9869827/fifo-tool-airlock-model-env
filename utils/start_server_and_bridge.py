"""
start_server_and_bridge.py

This script manages and displays logs for two uvicorn servers:
one running inside a Docker container (`phi`) and one running on the local host.
The uvicorn servers are both bound to 127.0.0.1:8000, matching the documentation.

- The Docker container is expected to run a single instance of
  `fifo_tool_airlock_model_env.bridge.fastapi_server:app`. Termination uses
  `pkill -SIGINT -f fifo_tool_airlock_model_env.bridge.fastapi_server:app` inside the container.
- The script works only on Windows, consistent with the documented and supported platform.
- Server process names, container name, and port configuration all match the current README.
"""

from dataclasses import dataclass, field
import platform
import subprocess
import threading
import signal
import time
import sys
from collections import deque
import re
import shutil
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

def _split_panel_heights(total_height: int, n_panels: int) -> list[int]:
    """
    Evenly split a total number of lines into N panels, giving any extra lines
    to the bottom panels (i.e., the last panels in the list).

    Example:
        split_panel_heights(5, 3) -> [1, 2, 2]
        split_panel_heights(10, 3) -> [3, 3, 4]
        split_panel_heights(7, 2) -> [3, 4]

    Args:
        total_height (int):
            The total number of lines to split.

        n_panels (int):
            The number of panels.

    Returns:
        A list of integers, where each value is the height of a panel,
        and sum(list) == total_height.
    """
    base = total_height // n_panels
    extra = total_height % n_panels
    return [base] * (n_panels - extra) + [base + 1] * extra


def _get_terminal_height() -> int:
    """
    Returns the number of lines in the current terminal.
    Falls back to 10 if terminal size can't be determined.

    Returns:
        int:
            Number of terminal lines (height), or 10 if unavailable.
    """
    try:
        return shutil.get_terminal_size().lines
    except OSError:
        return 10


@dataclass
class UvicornProcessInfo:
    """
    Manages a single process for log streaming and clean shutdown.

    Parameters:
        name (str):
            Process label, used for UI panel naming.

        cmd_start (list[str]):
            List of command line arguments to start the process.
            Passed to `subprocess.Popen` to launch the `uvicorn` process.

        cmd_stop (list[str] | None):
            List of command line arguments to stop the process, or None to use a signal.
            When provided, it is passed to `subprocess.run` to initiate shutdown of the process.
            If None, `signal.CTRL_BREAK_EVENT` is sent to terminate the process.

        log_buffer (deque[str]):
            Internal: Rolling buffer (maxlen=50) storing recent log lines.

        lock (threading.Lock):
            Internal: Thread lock to guard `log_buffer`.

        process (subprocess.Popen[str]):
            Underlying `Popen` object for the process (set in `__post_init__`).

        thread (threading.Thread):
            Thread object for log reading (set in `__post_init__`).
    """

    name: str
    cmd_start: list[str]
    cmd_stop: list[str] | None
    log_buffer: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    lock: threading.Lock = field(default_factory=threading.Lock)
    process: subprocess.Popen[str] = field(init=False)
    thread: threading.Thread = field(init=False)

    def __post_init__(self):
        """
        Starts the subprocess and the log reading thread.
        """
        self.process = subprocess.Popen(
            self.cmd_start,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        self.thread = threading.Thread(target=self._stream_output, name=f"{self.name}_stream")
        self.thread.daemon = True
        self.thread.start()

    def _sanitize_log(self, text: str) -> str:
        """
        Strips ANSI escape codes and filters characters for safe UI display, keeping only
        approved emoji and ASCII characters expected in log outputs.

        Arguments:
            text:
                The log line to sanitize.

        Returns:
            str:
                Sanitized log line containing only whitelisted characters.
        """
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        text = ansi_escape.sub('', text)
        allowed_ascii = set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            " .,'<>[]()/\\-_+|=:"  
            "[]{}<>?:;!@#$%^&*~`\""
            "|"
            "\t"
            " "
        )
        allowed_unicode = set("✅🚀📦🔧│█")
        return ''.join(c for c in text if c in allowed_ascii or c in allowed_unicode)

    def _stream_output(self):
        """
        Reads process output and appends sanitized lines to the log buffer.
        Handles OSError and RuntimeError gracefully.
        """
        try:
            if self.process.stdout is None:
                raise RuntimeError("Cannot read process output")
            for line in self.process.stdout:
                clean_line = self._sanitize_log(line.rstrip('\n'))
                with self.lock:
                    self.log_buffer.append(clean_line)
        except (OSError, RuntimeError):
            print(f"[{self.name}] ERROR in log thread", file=sys.stderr)

class UvicornProcessManager:
    """
    Manages multiple UvicornProcessInfo objects for:
    - Process lifecycle: processes are individually started with `start_process`
      and all terminated with `stop_all`.
    - Log streaming to UI display in the terminal.

    Handles concurrent process output and graceful shutdown via SIGINT (Ctrl-C).

    Parameters:
        _procs (dict[str, UvicornProcessInfo]):
            Dictionary mapping process names to their UvicornProcessInfo objects.

        _stop_requested_time (float | None):
            Time (epoch seconds) when shutdown was requested, or None if still running.

        _stop_requested (bool):
            Set to True after shutdown has started to prevent duplicate stop logic.
    """

    _procs: dict[str, UvicornProcessInfo]
    _stop_requested_time: float | None
    _stop_requested: bool

    def __init__(self) -> None:
        """
        Initializes the manager with empty process list and reset shutdown state.
        """
        self._procs = {}
        self._stop_requested_time = None
        self._stop_requested = False

    def start_process(self, name: str, cmd_start: list[str], cmd_stop: list[str] | None):
        """
        Launches a new process and begins streaming its logs.

        Arguments:
            name:
                Process label for display and internal lookup.

            cmd_start:
                List of arguments to start the process.
                Passed to `subprocess.Popen` to launch the `uvicorn` process.

            cmd_stop:
                List of arguments to stop the process, or None to use signal.
                When provided, it is passed to `subprocess.run` to initiate shutdown of the process.
                If None, `signal.CTRL_BREAK_EVENT` is sent to terminate the process.
        """
        info = UvicornProcessInfo(
            name=name,
            cmd_start=cmd_start,
            cmd_stop=cmd_stop
        )
        self._procs[name] = info

    def stop_all(self):
        """
        Attempts to gracefully stop all managed processes and joins their log threads.
        """
        if self._stop_requested:
            # Ctrl-C already pressed; skipping subsequent call
            return

        print("\n[MAIN] Sending SIGTERM to all processes...")
        for name, info in self._procs.items():
            proc, cmd_stop = info.process, info.cmd_stop
            if cmd_stop:
                subprocess.run(cmd_stop, check=True)
            else:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
        for name, info in self._procs.items():
            proc = info.process
            try:
                proc.wait(timeout=5)
                print(f"[{name}] exited cleanly.")
            except subprocess.TimeoutExpired:
                print(f"[{name}] did not exit in time. Forcing kill.")
                proc.kill()
                proc.wait()
        for info in self._procs.values():
            info.thread.join(timeout=3)
            if info.thread.is_alive():
                print(f"[{info.name}] WARNING: Thread did not exit cleanly.")
        self._stop_requested_time = time.time()

    def run_forever(self):
        """
        Runs the terminal UI loop for log display. Handles shutdown on Ctrl-C.
        Cleans up all processes and clears resources when done.
        """
        signal.signal(signal.SIGINT, lambda s, f: self.stop_all())

        layout = Layout()
        layout.split_column(*[Layout(name=name, ratio=1) for name in self._procs])

        with Live(layout, refresh_per_second=10, screen=True):
            while self._stop_requested_time is None or time.time() - self._stop_requested_time < 6:
                terminal_height = _get_terminal_height()

                for name, panel_height in zip(
                    self._procs,
                    _split_panel_heights(terminal_height - 4, len(self._procs))
                ):
                    info = self._procs[name]
                    with info.lock:
                        log_lines = list(info.log_buffer)
                    log_display = log_lines[-panel_height:]
                    log_str = "\n".join(log_display)
                    layout[name].update(
                        Panel(
                            Text(log_str, overflow="crop", no_wrap=False),
                            title=f"[bold yellow]{name}[/bold yellow]",
                            border_style="green" if self._stop_requested_time is None else "red"
                        )
                    )

            time.sleep(2)

        self._procs.clear()  # Clean up after log display is fully finished

if __name__ == "__main__":

    assert platform.system() == "Windows"

    manager = UvicornProcessManager()
    cmd_server = [
        "docker", "exec", "--workdir", "/home/fifodev/fifo-tool-airlock-model-env", "phi",
        "uvicorn", "fifo_tool_airlock_model_env.server.fastapi_server:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    cmd_kill_server = [
        "docker", "exec", "phi",
        "pkill", "-SIGINT", "-f", "fifo_tool_airlock_model_env.server.fastapi_server:app"
    ]
    cmd_bridge = [
        "uvicorn", "fifo_tool_airlock_model_env.bridge.fastapi_server:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    manager.start_process("SERVER", cmd_server, cmd_kill_server)
    manager.start_process("BRIDGE", cmd_bridge, None)
    manager.run_forever()
