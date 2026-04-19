"""Single-instance guard using QLocalServer / QLocalSocket.

Analogous to telecode's implicit port-bind collision (a second
`python main.py` tries to bind 1235 and crashes). VoxType doesn't
bind any fixed external ports of its own — the sidecars do — so we
need an explicit mechanism.

Flow:
  1. New process calls `is_already_running()`.
  2. If a previous instance's QLocalServer is listening, we send a
     b"show" command so it raises its settings window, then exit.
  3. Otherwise we install the server ourselves; subsequent invocations
     will connect to us.

Server key is a per-user, per-machine constant so multiple users on
the same Windows machine can each run their own VoxType.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Callable

from PySide6.QtCore import QObject
from PySide6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger("voxtype.single_instance")


def _server_name() -> str:
    """Stable per-user key. USERPROFILE on Windows, HOME elsewhere."""
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME") or ""
    digest = hashlib.sha1(home.encode("utf-8")).hexdigest()[:10]
    return f"voxtype-single-instance-{digest}"


def is_already_running(timeout_ms: int = 500) -> bool:
    """Return True if a previous instance answered. Side-effect: sends
    b'show' so the running instance surfaces its settings window."""
    name = _server_name()
    sock = QLocalSocket()
    sock.connectToServer(name)
    if not sock.waitForConnected(timeout_ms):
        return False
    try:
        sock.write(b"show\n")
        sock.flush()
        sock.waitForBytesWritten(timeout_ms)
    finally:
        sock.disconnectFromServer()
    log.info("another VoxType instance is running — asked it to show + exiting")
    return True


class InstanceServer(QObject):
    """Listen for activation commands from subsequent invocations."""
    def __init__(self, on_show: Callable[[], None]) -> None:
        super().__init__()
        self._on_show = on_show
        self._server = QLocalServer(self)
        # Clean up any stale socket file (previous instance crashed before
        # QLocalServer could close it). removeServer() is a no-op if fine.
        QLocalServer.removeServer(_server_name())
        self._server.newConnection.connect(self._on_new_connection)
        if not self._server.listen(_server_name()):
            log.warning("single-instance server failed to bind: %s",
                         self._server.errorString())
        else:
            log.info("single-instance server listening on %s", _server_name())

    def _on_new_connection(self) -> None:
        sock = self._server.nextPendingConnection()
        if sock is None:
            return
        def _read():
            data = bytes(sock.readAll()).strip()
            if data == b"show":
                try:
                    self._on_show()
                except Exception as exc:
                    log.warning("on_show failed: %s", exc)
            sock.disconnectFromServer()
        sock.readyRead.connect(_read)
