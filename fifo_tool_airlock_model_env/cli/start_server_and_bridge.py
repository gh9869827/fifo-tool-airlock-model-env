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

from abc import ABC, abstractmethod
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
from typing import Any, Iterator, cast
import docker  # type: ignore[import]
from docker.errors import NotFound, APIError  # type: ignore[import]
from docker.models.containers import Container  # type: ignore[import]
from docker.client import DockerClient  # type: ignore[import]

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


@dataclass(kw_only=True)
class Command(ABC):
    """
    Abstract base class for commands executed locally or in a Docker container.
    Commands are executed in a separate process.

    Provides a common interface for background process management, log streaming,
    and thread-safe logging. Subclasses must implement process-specific details.
    
    Parameters:
        name (str):
            Human-readable name of the command, used for UI/log display.

        log_buffer (collections.deque[str], optional):
            Rolling buffer storing recent log lines (default: maxlen=50).

        lock (threading.Lock, optional):
            Lock to ensure thread-safe log buffer access.

        thread (threading.Thread, internal):
            Background thread for log streaming (initialized on run).
    """
    name: str
    log_buffer: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread = field(init=False)

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

    def log(self, text: str) -> None:
        """
        Add a sanitized log line to the rolling log buffer.

        Args:
            text (str):
                Log message to add.
        """
        clean_text = self._sanitize_log(text.rstrip('\n').rstrip('\r'))
        with self.lock:
            self.log_buffer.append(clean_text)

    def log_error(self, text: str) -> None:
        """
        Add an error message (prefixed) to the log buffer.

        The message is sanitized before being added.

        Args:
            text (str):
                Error message to add.
        """
        self.log(f"ERROR:    {text}")

    def log_warning(self, text: str) -> None:
        """
        Add a warning message (prefixed) to the log buffer.

        The message is sanitized before being added.

        Args:
            text (str):
                Warning message to add.
        """
        self.log(f"WARNING:  {text}")

    def log_info(self, text: str) -> None:
        """
        Add an informational message (prefixed) to the log buffer.

        The message is sanitized before being added.

        Args:
            text (str):
                Informational message to add.
        """
        self.log(f"INFO:     {text}")

    def run_background(self) -> None:
        """
        Launch the background process or command and start streaming its output.

        Calls subclass-specific `_run_background_cmd()`, then starts a background thread
        that streams and logs process output.
        """
        self._run_background_cmd()
        self.thread = threading.Thread(
            target=self.stream_output, name=f"{self.name}_stream"
        )
        self.thread.daemon = True
        self.thread.start()

    @abstractmethod
    def _run_background_cmd(self) -> None:
        """
        Subclass hook: Start the command (e.g., subprocess or Docker exec)
        and store any required handles for output streaming.
        """

    def stream_output(self):
        """
        Entrypoint for the background log streaming thread.

        Calls the subclass-specific `_stream_output()` method and handles any exceptions,
        logging an error message in the log buffer if streaming fails.
        """
        try:
            self._stream_output()
        except Exception:
            self.log_error("Unexpected error in log streaming thread.")

    @abstractmethod
    def _stream_output(self) -> None:
        """
        Subclass hook: Continuously stream log output from the process,
        sanitizing each line before adding to the log buffer.

        Should terminate when process ends or an unrecoverable error occurs.
        """

    def stop_background(self) -> None:
        """
        Initiate shutdown of the background process.

        Calls subclass `_stop_background_cmd()`. Logs the stop request and
        any errors as warnings or errors in the log buffer.
        """
        try:
            self.log_info("Requesting stop.")
            self._stop_background_cmd()
        except RuntimeError:
            self.log_error("stop command failed.")

        # the background thread terminate automatically when the background process stops
        # as it stops streaming

    def _stop_background_cmd(self) -> None:
        """
        Subclass hook: Trigger process shutdown (e.g., send signal or run stop command).

        Should not raise new exceptions; errors must be reported in the log buffer.
        """

    def join_background(self):
        """
        Wait for the process and its log-streaming thread to finish.

        Calls subclass `_join_background_cmd()` before joining the thread.
        Logs a warning message in the log buffer if the thread does not exit cleanly.
        """
        self._join_background_cmd()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            self.log_warning("Thread did not exit cleanly.")

    @abstractmethod
    def _join_background_cmd(self):
        """
        Subclass hook: Wait for process termination (e.g., wait for subprocess).

        Called prior to joining the background log-streaming thread.
        """

@dataclass(kw_only=True)
class CommandLocal(Command):
    """
    Command implementation for running and managing a local subprocess.

    Launches a command in a new process, streams its output to the log buffer,
    and provides methods for graceful and forced shutdown.

    Parameters:
        cmd (list[str]):
            Command-line arguments for the process to execute.

        _popen (subprocess.Popen[str], internal):
            The underlying subprocess handle (initialized on start).
    """

    cmd: list[str]
    _popen: subprocess.Popen[str] = field(init=False, repr=False)

    def _run_background_cmd(self) -> None:
        """
        Start the local subprocess using the specified command-line arguments.

        The process is started by calling `subprocess.Popen()` with stdout and stderr merged,
        and with text mode enabled for real-time log streaming.
        """
        self._popen = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

    def _stream_output(self) -> None:
        """
        Stream output lines from the local subprocess to the log buffer.

        Reads from the process's stdout and appends each line to the log buffer for UI display
        by calling `log`. Each message is sanitized before being added to the log buffer.

        Raises:
            RuntimeError:
                If the process's stdout stream cannot be read.
        """
        if self._popen.stdout is None:
            raise RuntimeError("Cannot read process output")
        for line in self._popen.stdout:
            self.log(line.rstrip('\n').rstrip('\r'))

    def _stop_background_cmd(self) -> None:
        """
        Request graceful shutdown of the subprocess.

        Sends a CTRL_BREAK_EVENT to the process to trigger clean termination.
        """
        self._popen.send_signal(signal.CTRL_BREAK_EVENT)

    def _join_background_cmd(self) -> None:
        """
        Wait for the subprocess to exit, forcing termination if it does not exit in time.

        Waits up to 5 seconds for the process to terminate; if it does not exit,
        sends a kill signal and waits for forced shutdown. Logs a warning if forced.
        """
        try:
            self._popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.log_warning("process did not exit in time. Forcing killing.")
            self._popen.kill()
            self._popen.wait()


@dataclass(kw_only=True)
class CommandDocker(Command):
    """
    Command implementation for running and managing a command inside a Docker container.

    Launches a command within the container using Docker exec (via the Python SDK),
    streams its output to a thread-safe log buffer, and provides methods for
    graceful shutdown and verification of termination.

    Parameters:
        cmd_start (list[str]):
            Command-line arguments to start the background process in the container.

        cmd_stop (list[str]):
            Command-line arguments to request process shutdown (e.g., pkill) in the container.

        cmd_verify_stop (list[str]):
            Command-line arguments to verify process termination (e.g., pgrep) in the container.

        container (docker.models.containers.Container):
            Docker container object in which to run commands.

        working_dir (str | None, optional):
            Working directory inside the container for the start command (default: None).

        _exec_id (str, internal):
            Docker exec instance ID (set after command start).

        _stream (Iterator[bytes], internal):
            Iterator over the output stream from Docker exec (set after command start).
    """
    cmd_start: list[str]
    cmd_stop: list[str]
    cmd_verify_stop: list[str]
    container: Container
    working_dir: str | None = None
    _exec_id: str = field(init=False, repr=False)
    _stream: Iterator[bytes] = field(init=False, repr=False)

    def _run_background_cmd(self) -> None:
        """
        Start the background process inside the Docker container using
        Docker exec (via the Python SDK).

        Initializes the exec instance and begins streaming output as bytes
        for real-time log display. Stores the exec ID and output stream for
        further management.
        """
        client = cast(DockerClient, self.container.client)
        response = cast(
            dict[str, Any],
            # Pylance: Type of "exec_create" is partially unknown
            client.api.exec_create(  # type: ignore[reportUnknownMemberType]
                self.container.id,
                self.cmd_start,
                workdir=self.working_dir
            )
        )
        self._exec_id = response["Id"]
        self._stream = cast(
            Iterator[bytes],
            # Pylance: Type of "exec_start" is partially unknown
            client.api.exec_start(  # type: ignore[reportUnknownMemberType]
                self._exec_id,
                stream=True
            )
        )

    def _stream_output(self) -> None:
        """
        Stream output from the running Docker exec command to the log buffer.

        Decodes each byte chunk, splits on both '\\n' and '\\r', and logs each non-empty line.
        Each line is sanitized by the `log()` function before being added to the buffer.
        Flushes any remaining partial line at the end of the stream.
        """
        buffer = ""
        for chunk in self._stream:
            text = chunk.decode("utf-8")
            buffer += text
            parts = re.split(r'[\r\n]', buffer)
            buffer = parts.pop()
            for line in parts:
                if line.strip():
                    self.log(line)
        if buffer.strip():
            self.log(buffer)

    def _run_once(self, cmd: list[str]) -> int:
        """
        Run a single command inside the Docker container and return its exit code.

        Args:
            cmd (list[str]):
                Command-line arguments for the process to execute.

        Returns:
            int:
                Exit code of the executed command.
        """
        client = cast(DockerClient, self.container.client)

        response = cast(
            dict[str, Any],
            # Pylance: Type of "exec_create" is partially unknown
            client.api.exec_create(  # type: ignore[reportUnknownMemberType]
                self.container.id,
                cmd
            )
        )
        exec_id = response["Id"]

        _output = cast(
            bytes,
            # Pylance: Type of "exec_start" is partially unknown
            client.api.exec_start(  # type: ignore[reportUnknownMemberType]
                exec_id,
                stream=False
            )
        ).decode("utf-8")

        exec_inspect = cast(
            dict[str, Any],
            # Pylance: Type of "exec_inspect" is partially unknown
            client.api.exec_inspect(  # type: ignore[reportUnknownMemberType]
                exec_id
            )
        )

        return exec_inspect["ExitCode"]

    def _stop_background_cmd(self) -> None:
        """
        Request graceful shutdown of the background process inside the container.

        Executes the configured stop command and logs an error if it fails (nonzero exit code).
        """
        exit_code = self._run_once(self.cmd_stop)

        if exit_code != 0:
            self.log_error(f"Stop command exited with error code = {exit_code}")

    def _join_background_cmd(self):
        """
        Verify background process termination by running the verification command.

        Logs an error if the verification command indicates the process is still running.
        """
        exit_code = self._run_once(self.cmd_verify_stop)

        if exit_code == 0:
            self.log_error("unable to stop docker background process.")

class UvicornProcessManager:
    """
    Manages the lifecycle and log streaming for multiple background commands.

    Responsibilities:
        - Starts and tracks multiple background processes (local or Docker).
        - Streams log output from each process to a terminal UI using Rich panels.
        - Handles graceful shutdown of all processes on Ctrl-C.

    Attributes:
        _procs (dict[str, Command]):
            Dictionary mapping process names to Command instances (processes).

        _stop_requested_time (float | None):
            Epoch time when shutdown was requested, or None if still running.

        _stop_requested (bool):
            Flag indicating whether shutdown has already been triggered.
    """
    _procs: dict[str, Command]
    _stop_requested_time: float | None
    _stop_requested: bool

    def __init__(self) -> None:
        self._procs = {}
        self._stop_requested_time = None
        self._stop_requested = False

    def start_process(self, cmd: Command):
        """
        Launches a new process and begins streaming its logs.
        """
        cmd.run_background()
        self._procs[cmd.name] = cmd

    def stop_all(self):
        """
        Attempts to gracefully stop all managed processes and joins their log threads.
        """
        if self._stop_requested:
            return  # Already requested stop

        for _, cmd in self._procs.items():
            cmd.stop_background()

        self._stop_requested_time = time.time()
        self._stop_requested = True

    def run_forever(self):
        """
        Run the terminal UI loop to display log output for all managed processes.

        Handles shutdown on Ctrl-C. Waits for all processes and their threads
        to exit cleanly, then clears all process resources.
        """
        signal.signal(signal.SIGINT, lambda s, f: self.stop_all())

        layout = Layout()
        layout.split_column(*[Layout(name=name, ratio=1) for name in self._procs])

        with Live(layout, refresh_per_second=10, screen=True):
            join_requested = False
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

                if (    self._stop_requested_time 
                    and time.time() > self._stop_requested_time + 3
                    and join_requested is False
                ):
                    join_requested = True

                    # Wait for local processes to exit (Docker exec just closes stream)
                    for name, cmd in self._procs.items():
                        cmd.join_background()

                time.sleep(0.05)

        self._procs.clear()  # Clean up after log display is fully finished


# Docker container name (defined in the README as 'phi')
CONTAINER_NAME = "phi"


def _check_requirements_exit_on_failure() -> Container:

    if platform.system() != "Windows":
        sys.exit("🛑 This program can only run on Windows.")

    client = docker.from_env()

    print("🔍 Checking container status:")

    try:
        container = client.containers.get(CONTAINER_NAME)
        status = container.status  # e.g., "running", "paused", "exited", "created"
    except NotFound:
        sys.exit(f"🛑 Container '{CONTAINER_NAME}' not found. Aborting.")
    except APIError:
        sys.exit("🛑 Docker API error occurred. Aborting.")

    if status == "paused":
        print(f"⏸️  Container '{CONTAINER_NAME}' is paused. Unpausing...")
        try:
            # Pylance: Type of "unpause" is partially unknown
            container.unpause()  # type: ignore[reportUnknownMemberType]
        except APIError:
            sys.exit(f"🛑 Failed to unpause container '{CONTAINER_NAME}'. Aborting.")

    elif status == "exited":
        print(f"🚀 Container '{CONTAINER_NAME}' is exited. Starting...")
        try:
            container.start()
        except APIError:
            sys.exit(f"🛑 Failed to start container '{CONTAINER_NAME}'. Aborting.")

    elif status == "running":
        print(f"✅ Container '{CONTAINER_NAME}' is already running.")

    else:
        sys.exit(f"🛑 Container '{CONTAINER_NAME}' is in unexpected state '{status}'. Aborting.")

    networks = container.attrs['NetworkSettings']['Networks']
    if networks:
        sys.exit(
            f"🛑 Container '{CONTAINER_NAME}' should not be attached to any Docker network, "
            f"but found: {', '.join(networks.keys())}\n"
            "   Please disconnect all networks using:\n"
            f"     docker network disconnect <network> {CONTAINER_NAME}"
        )

    return container

def main():
    """
    Main entry point.
    """
    container = _check_requirements_exit_on_failure()

    time.sleep(1)

    manager = UvicornProcessManager()

    cmd_server = CommandDocker(
            name="SERVER",
            container=container,
            working_dir="/home/fifodev/fifo-tool-airlock-model-env",
            cmd_start=[
                "uvicorn", "fifo_tool_airlock_model_env.server.fastapi_server:app",
                "--host", "127.0.0.1",
                "--port", "8000"
            ],
            cmd_stop=[
                "pkill", "-SIGINT", "-f", "fifo_tool_airlock_model_env.server.fastapi_server:app"
            ],
            cmd_verify_stop=[
                "pgrep", "-f", "fifo_tool_airlock_model_env.server.fastapi_server:app"
            ]
    )

    cmd_bridge = CommandLocal(
            name="BRIDGE",
            cmd=[
                "uvicorn", "fifo_tool_airlock_model_env.bridge.fastapi_server:app",
                "--host", "127.0.0.1",
                "--port", "8000"
            ]
    )
    manager.start_process(cmd_server)
    manager.start_process(cmd_bridge)
    manager.run_forever()

if __name__ == "__main__":
    main()
