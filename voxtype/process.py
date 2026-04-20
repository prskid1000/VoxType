"""Subprocess lifecycle — everything process-related for VoxType.

Layers (top → bottom of file):
1. **Generic primitives** — port probe, tree-kill, command-line-aware
   orphan sweep, Windows Job Object (kill-on-close), readiness probe
   with poll-death guard.
2. **Sidecar supervisors** — Whisper + Kokoro child-process lifecycle
   (spawn env, stdout drain, auto-restart with exponential backoff,
   idle-unload, GPU-broken → CPU fallback, graceful stop).

The Job Object binds every spawned sidecar to the Python interpreter's
lifetime: if VoxType dies for ANY reason (Ctrl+C, Task Manager End
Process, pythonw crash, logoff) Windows kills the whole tree. This is
what prevents the orphan-holding-our-port restart loop that motivated
the consolidation.

Windows-specific: CREATE_NO_WINDOW so children don't pop a console."""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import aiohttp

from voxtype import config as _config

log = logging.getLogger("voxtype.process")

ServiceName = Literal["whisper", "kokoro"]
DeviceMode = Literal["gpu", "cpu"]


def _service_log_path(name: ServiceName) -> Path:
    return _config.data_dir() / f"{name}.log"


def _rotate_service_log(name: ServiceName) -> Path:
    """Move <name>.log → <name>.log.prev and return the path to write into.
    Best-effort: if rotation fails (file held by another process), append."""
    cur = _service_log_path(name)
    prev = cur.with_suffix(".log.prev")
    try:
        cur.parent.mkdir(parents=True, exist_ok=True)
        if cur.exists():
            if prev.exists():
                prev.unlink()
            cur.rename(prev)
    except Exception:
        pass
    return cur


# Strings the Whisper subprocess prints when CUDA / cuBLAS isn't usable. When
# we see one we mark the service as GPU-broken; the next start_whisper() call
# will silently switch to CPU mode regardless of what the user has configured.
_GPU_BROKEN_MARKERS = (
    "cublas64_",                # cublas64_12.dll / cublas64_11.dll missing
    "cudnn",                    # cudnn DLL load failure
    "Library cublas",
    "CUDA driver version",
    "no kernel image is available for execution on the device",
)
_gpu_broken: dict[ServiceName, bool] = {}


INSTALL_DIR = Path(os.path.expanduser("~")) / ".voxtype"
STT_VENV    = INSTALL_DIR / "stt-venv"
TTS_VENV    = INSTALL_DIR / "tts-venv"
KOKORO_REPO = INSTALL_DIR / "Kokoro-FastAPI"


@dataclass
class WhisperConfig:
    model: str
    port: int
    device: DeviceMode = "gpu"


@dataclass
class KokoroConfig:
    port: int
    device: DeviceMode = "gpu"


@dataclass
class ServiceStatus:
    name: ServiceName
    pid: int | None
    running: bool
    ready: bool
    last_error: str = ""


@dataclass
class _Managed:
    name: ServiceName
    proc: subprocess.Popen | None = None
    ready: bool = False
    last_error: str = ""
    config: WhisperConfig | KokoroConfig | None = None
    stopping: bool = False
    restart_count: int = 0
    restart_task: asyncio.Task | None = None
    # Idle-unload bookkeeping
    last_used: float = 0.0          # time.monotonic() of last mark_used()
    idle_unload_sec: int = 0        # 0 = never unload


_services: dict[ServiceName, _Managed] = {}
_status_listeners: list[Callable[[ServiceStatus], None]] = []
_lock = threading.Lock()

# Single watcher thread started by start_idle_watcher(); checks every
# IDLE_WATCH_INTERVAL seconds. No pressure on the event loop.
_idle_watcher_started = False
IDLE_WATCH_INTERVAL = 30.0


def mark_used(name: ServiceName) -> None:
    """Call before every real request so the idle watcher doesn't stop
    the service in the middle of active usage. Cheap (monotonic())."""
    m = _services.get(name)
    if m is not None:
        m.last_used = time.monotonic()


def set_idle_unload(name: ServiceName, seconds: int) -> None:
    """Set per-service idle-unload threshold. 0 = never unload."""
    with _lock:
        m = _services.setdefault(name, _Managed(name=name))
        m.idle_unload_sec = max(0, int(seconds))


def start_idle_watcher() -> None:
    """Spawn the shared watcher thread (idempotent)."""
    global _idle_watcher_started
    if _idle_watcher_started:
        return
    _idle_watcher_started = True

    def _loop() -> None:
        while True:
            try:
                time.sleep(IDLE_WATCH_INTERVAL)
                now = time.monotonic()
                for name, m in list(_services.items()):
                    if m.idle_unload_sec <= 0:
                        continue
                    if not m.proc or m.proc.poll() is not None:
                        continue
                    if m.last_used <= 0:
                        # never used yet — start the clock from spawn time
                        m.last_used = now
                        continue
                    idle = now - m.last_used
                    if idle >= m.idle_unload_sec:
                        log.info("%s idle for %.0fs ≥ %ds — unloading",
                                 name, idle, m.idle_unload_sec)
                        # Schedule stop on a short-lived thread so we don't
                        # block the watcher loop.
                        threading.Thread(
                            target=lambda n=name: asyncio.run(stop_service(n)),
                            daemon=True,
                        ).start()
            except Exception as exc:
                log.debug("idle watcher tick failed: %s", exc)

    threading.Thread(target=_loop, daemon=True,
                     name="voxtype-idle-watcher").start()


def on_status_change(fn: Callable[[ServiceStatus], None]) -> None:
    _status_listeners.append(fn)


def _health_url(m: _Managed) -> str:
    port = m.config.port if m.config else 0  # type: ignore[union-attr]
    return f"http://127.0.0.1:{port}/health"


def _notify(name: ServiceName) -> None:
    m = _services.get(name)
    if m is None:
        return
    s = ServiceStatus(
        name=name,
        pid=(m.proc.pid if m.proc and m.proc.poll() is None else None),
        running=bool(m.proc and m.proc.poll() is None),
        ready=m.ready,
        last_error=m.last_error,
    )
    for fn in _status_listeners:
        try:
            fn(s)
        except Exception:
            pass


# ── Binary paths ─────────────────────────────────────────────────────

def _whisper_exe() -> Path:
    return STT_VENV / "Scripts" / "faster-whisper-server.exe"


def _uvicorn_exe() -> Path:
    return TTS_VENV / "Scripts" / "uvicorn.exe"


# ── Kill-on-close Job Object (Windows crash-safety) ──────────────────
#
# Every sidecar we spawn is assigned to a single process-wide Job Object
# flagged KILL_ON_JOB_CLOSE. The Job handle lives in `_JOB_HANDLE` for
# the lifetime of the interpreter — when Python exits for ANY reason
# (clean quit, Ctrl+C, pythonw crash, Task Manager End Process, OS
# logoff) the Job closes and Windows terminates every assigned
# process. This is what prevents the stale-sidecar-holds-our-port
# restart loop that motivated consolidating everything into this file.
#
# atexit fallback handles the narrow case where Job Object creation
# failed (pywin32 missing): on clean interpreter exit we proc.kill()
# every tracked child. It can't help against hard kills — that's what
# the Job Object is for.

_JOB_HANDLE = None
_TRACKED_PIDS: set[int] = set()
_TRACKED_PROCS: list[subprocess.Popen] = []


def _create_kill_on_close_job():
    """Idempotent: create the process-wide Job Object on first call."""
    global _JOB_HANDLE
    if os.name != "nt" or _JOB_HANDLE is not None:
        return _JOB_HANDLE
    try:
        import win32job
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation
        )
        info["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        win32job.SetInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation, info
        )
        _JOB_HANDLE = job
        log.info("created kill-on-close Job Object")
    except Exception as exc:
        log.warning("could not create Job Object (orphans possible "
                    "on hard-kill of voxtype): %s", exc)
    return _JOB_HANDLE


def _bind_to_lifetime_job(proc: subprocess.Popen) -> bool:
    """Bind `proc` to the kill-on-close Job Object so Windows reaps it
    if this interpreter dies unexpectedly. Tracks the Popen for the
    atexit fallback regardless of whether the Job bind succeeds."""
    if proc not in _TRACKED_PROCS:
        _TRACKED_PROCS.append(proc)
    if os.name != "nt" or proc.pid <= 0 or proc.pid in _TRACKED_PIDS:
        return True
    job = _create_kill_on_close_job()
    if job is None:
        return False
    try:
        import win32api
        import win32con
        import win32job
        ph = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, proc.pid)
        try:
            win32job.AssignProcessToJobObject(job, ph)
        finally:
            win32api.CloseHandle(ph)
        _TRACKED_PIDS.add(proc.pid)
        return True
    except Exception as exc:
        log.warning("could not assign PID %d to Job Object: %s", proc.pid, exc)
        return False


def _atexit_kill_tracked() -> None:
    """Clean-exit fallback when the Job Object path isn't available.
    Hard-kill paths (SIGKILL, OS shutdown) are covered by the Job itself."""
    for proc in list(_TRACKED_PROCS):
        try:
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
        except Exception:
            pass


atexit.register(_atexit_kill_tracked)


# ── Orphan sweep ─────────────────────────────────────────────────────

def _port_in_use(port: int) -> bool:
    """Return True if something is listening on 127.0.0.1:<port>."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def _pids_listening_on(port: int) -> list[int]:
    """PIDs of processes holding 127.0.0.1:<port> in LISTEN state.
    Uses Get-NetTCPConnection (no admin needed)."""
    if os.name != "nt":
        return []
    ps = (
        "$c = Get-NetTCPConnection -LocalAddress 127.0.0.1 "
        f"-LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue; "
        "if ($c) { $c.OwningProcess | Sort-Object -Unique }"
    )
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=5.0, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _process_image(pid: int) -> tuple[str, str]:
    """Return (exe_path, command_line) for a PID. Both are "" if unknown.

    Checking the command line matters because a .exe under our venv is
    often launched by the pyenv python.exe — so the owning process `Path`
    points to pyenv, while the real sidecar identity is in argv[0]."""
    if os.name != "nt":
        return "", ""
    ps = (
        f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}' "
        "-ErrorAction SilentlyContinue; "
        "if ($p) { $p.ExecutablePath; '---'; $p.CommandLine }"
    )
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=5.0, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
    except Exception:
        return "", ""
    parts = out.split("---", 1)
    exe = parts[0].strip() if parts else ""
    cmd = parts[1].strip() if len(parts) > 1 else ""
    return exe, cmd


def _sweep_port(name: ServiceName, port: int) -> None:
    """If an orphan from a previous VoxType run is holding `port`, kill it.
    Only targets processes whose exe OR command line points inside our
    sidecar venv, so a foreign app happening to use the same port doesn't
    get murdered."""
    if not _port_in_use(port):
        return
    venv_root = STT_VENV if name == "whisper" else TTS_VENV
    venv_str = str(venv_root).lower()
    pids = _pids_listening_on(port)
    for pid in pids:
        exe, cmd = _process_image(pid)
        exe_lc, cmd_lc = exe.lower(), cmd.lower()
        is_ours = (exe_lc.startswith(venv_str) or venv_str in cmd_lc)
        if is_ours:
            log.warning("[%s] port %d held by orphan PID %d (%s) — killing",
                        name, port, pid, exe or cmd or "unknown")
            _kill_tree(pid, force=True)
        else:
            log.error("[%s] port %d in use by foreign PID %d (exe=%s cmd=%s) — "
                      "not killing; service will fail to bind",
                      name, port, pid, exe or "?", cmd or "?")
    # Give the OS a moment to release the socket.
    for _ in range(20):
        if not _port_in_use(port):
            return
        time.sleep(0.1)


# ── Spawn helpers ────────────────────────────────────────────────────

def _nvidia_dll_dirs() -> list[str]:
    """Directories inside stt-venv that hold the CUDA 12 runtime DLLs the
    `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip wheels drop.

    setup.ps1 pip-installs both into stt-venv, which puts
    `cublas64_12.dll`, `cudnn*64_9.dll`, etc. under
    `stt-venv\\Lib\\site-packages\\nvidia\\*\\bin`. Windows'
    `LoadLibrary` doesn't search there by default — we have to prepend
    them to the child's PATH, otherwise ctranslate2 fails its first
    CUDA call with `Library cublas64_12.dll is not found`.

    Returns paths that exist; empty list on Linux or fresh install.
    """
    nvidia_root = STT_VENV / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.exists():
        return []
    dirs: list[str] = []
    for pkg in nvidia_root.iterdir():
        bin_dir = pkg / "bin"
        if bin_dir.is_dir():
            dirs.append(str(bin_dir))
    return dirs


def _spawn_whisper(cfg: WhisperConfig) -> subprocess.Popen:
    env = os.environ.copy()
    if cfg.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # Prepend NVIDIA DLL dirs so ctranslate2 can LoadLibrary the
        # CUDA 12 runtime. Without this the child crashes on first
        # inference (not at spawn) with a cublas-not-found error.
        dll_dirs = _nvidia_dll_dirs()
        if dll_dirs:
            env["PATH"] = os.pathsep.join(dll_dirs + [env.get("PATH", "")])
    # Force line-buffered stdout so POST /v1/audio/transcriptions lines
    # appear in whisper.log immediately instead of after a 4-8 KB block
    # (which can be never, on a fresh install that only serves a few
    # requests). Without this the log looked empty while real requests
    # were arriving and timing out.
    env["PYTHONUNBUFFERED"] = "1"
    args = [str(_whisper_exe()), cfg.model, "--host", "127.0.0.1",
            "--port", str(cfg.port)]
    proc = subprocess.Popen(
        args, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    _bind_to_lifetime_job(proc)
    return proc


def _spawn_kokoro(cfg: KokoroConfig) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8":      "1",
        "USE_GPU":         "true" if cfg.device == "gpu" else "false",
        "USE_ONNX":        "false",
        "PROJECT_ROOT":    str(KOKORO_REPO),
        "PYTHONPATH":      f"{KOKORO_REPO};{KOKORO_REPO / 'api'}",
        "MODEL_DIR":       "src/models",
        "VOICES_DIR":      "src/voices/v1_0",
        "WEB_PLAYER_PATH": str(KOKORO_REPO / "web"),
    })
    args = [str(_uvicorn_exe()), "api.src.main:app",
            "--host", "127.0.0.1", "--port", str(cfg.port)]
    proc = subprocess.Popen(
        args, env=env, cwd=str(KOKORO_REPO),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    _bind_to_lifetime_job(proc)
    return proc


# ── Health probe ─────────────────────────────────────────────────────

async def _ping_once(url: str, timeout: float = 1.5) -> bool:
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as session:
            async with session.get(url) as resp:
                return resp.status < 500
    except Exception:
        return False


async def _wait_ready(m: _Managed, url: str,
                      total_timeout: float = 600.0) -> bool:
    """Poll /health until the service answers or we hit total_timeout.

    Default is 10 minutes because faster-whisper-server downloads the
    model BEFORE starting uvicorn — so `_ping_once` returns False
    (connection refused) for the entire download. On a slow connection
    downloading faster-whisper-small (~244 MB) can easily take 3-5
    minutes. The previous 60s cap aborted mid-download and left
    whisper in an inconsistent state.

    Also bails out early if the child process exits. Without this check
    an orphan on the same port would answer /health and we'd falsely
    report our dead child as "ready after 0.0s".

    A periodic log line confirms we're still waiting rather than dead."""
    name = m.name
    start = time.monotonic()
    attempts = 0
    next_heartbeat = 30.0
    while (time.monotonic() - start) < total_timeout:
        attempts += 1
        if m.proc is None or m.proc.poll() is not None:
            log.warning("%s exited before becoming ready", name)
            return False
        if await _ping_once(url):
            # Ping success is only meaningful if OUR child is what's
            # serving the port. Re-poll after a short wait: if the child
            # died within that window, the ping was actually hitting an
            # orphan on the same port and we mustn't report "ready".
            await asyncio.sleep(1.0)
            if m.proc is None or m.proc.poll() is not None:
                log.warning("%s died right after /health probe — "
                            "another process on port", name)
                return False
            log.info("%s ready after %.1fs (%d attempts)",
                     name, time.monotonic() - start, attempts)
            return True
        elapsed = time.monotonic() - start
        if elapsed >= next_heartbeat:
            log.info("%s still not ready after %.0fs — likely downloading / "
                     "loading model (check %s.log)", name, elapsed, name)
            next_heartbeat += 30.0
        await asyncio.sleep(0.5)
    log.warning("%s did not become ready in %.0fs", name, total_timeout)
    return False


# ── stdout/stderr drain ──────────────────────────────────────────────

def _drain(name: ServiceName, proc: subprocess.Popen) -> None:
    """Pump the child's combined stdout into a dedicated log file.

    Writes go to <data_dir>/<name>.log (rotated to .prev each spawn). Only
    a one-line breadcrumb lands in the main voxtype.log so the merged
    timeline still shows that the service was alive, but the noisy
    per-request lines stay out of the main log.

    Also sniffs each line for `_GPU_BROKEN_MARKERS`. The first match flips
    `_gpu_broken[name]`, fires a one-time warning into the main log, and
    schedules a stop+restart with `device='cpu'`. That recovers the
    user's next dictation without manual intervention when CUDA is
    misconfigured (cublas DLLs missing, etc.).
    """
    log_path = _rotate_service_log(name)
    try:
        sink = open(log_path, "ab", buffering=0)
    except Exception as exc:
        log.warning("could not open %s for writing: %s", log_path, exc)
        sink = None
    log.info("[%s] writing subprocess log to %s", name, log_path)

    def _reader():
        try:
            if proc.stdout is None:
                return
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                if sink is not None:
                    try:
                        sink.write(line if line.endswith(b"\n") else line + b"\n")
                    except Exception:
                        pass
                if not _gpu_broken.get(name) and any(
                    m.lower() in text.lower() for m in _GPU_BROKEN_MARKERS
                ):
                    _gpu_broken[name] = True
                    log.warning(
                        "[%s] GPU appears broken (saw %r in subprocess output) "
                        "— scheduling restart on CPU. Install matching CUDA / "
                        "cuBLAS DLLs to keep using GPU.",
                        name, text[:160],
                    )
                    threading.Thread(
                        target=_force_cpu_restart, args=(name,),
                        daemon=True, name=f"voxtype-{name}-cpu-restart",
                    ).start()
        except Exception:
            pass
        finally:
            if sink is not None:
                try:
                    sink.close()
                except Exception:
                    pass

    t = threading.Thread(target=_reader, daemon=True,
                         name=f"voxtype-{name}-reader")
    t.start()


def _force_cpu_restart(name: ServiceName) -> None:
    """Stop the service and respawn with device='cpu'. Called from the
    log reader thread when a CUDA / cuBLAS error is detected. Idempotent
    — `_gpu_broken[name]` stays True so future `start_*` calls also
    coerce CPU mode for the rest of the session."""
    m = _services.get(name)
    if m is None or m.config is None:
        return
    new_cfg = m.config
    try:
        # Mutate in place so future restart_* calls keep CPU mode.
        new_cfg.device = "cpu"  # type: ignore[union-attr]
    except Exception:
        pass
    log.info("[%s] restarting on CPU due to GPU failure", name)
    try:
        asyncio.run(restart_service(name, new_cfg))
    except Exception as exc:
        log.error("[%s] CPU restart failed: %s", name, exc)


# ── Lifecycle ────────────────────────────────────────────────────────

async def _start_internal(m: _Managed) -> None:
    assert m.config is not None
    exe = _whisper_exe() if m.name == "whisper" else _uvicorn_exe()
    if not exe.exists():
        m.last_error = f"executable missing: {exe}"
        log.error("%s not installed (%s missing) — skipping start", m.name, exe)
        _notify(m.name)
        return
    if m.name == "kokoro" and not KOKORO_REPO.exists():
        m.last_error = f"Kokoro repo missing: {KOKORO_REPO}"
        log.error("kokoro repo missing (%s) — skipping start", KOKORO_REPO)
        _notify(m.name)
        return

    log.info("starting %s...", m.name)
    # If a previous VoxType died without cleaning up its sidecar, the port
    # is still held by the orphan — the new child will exit with bind error
    # 10048. Kill the orphan first so we can bind.
    port = getattr(m.config, "port", 0)
    if port:
        _sweep_port(m.name, int(port))
    m.proc = (_spawn_whisper(m.config)  # type: ignore[arg-type]
              if m.name == "whisper"
              else _spawn_kokoro(m.config))  # type: ignore[arg-type]
    m.ready = False
    m.last_error = ""
    _drain(m.name, m.proc)
    _notify(m.name)
    log.info("%s spawned (PID %d)", m.name, m.proc.pid)

    # Watch for exit on a thread so we can trigger auto-restart.
    threading.Thread(
        target=_watch_exit, args=(m,), daemon=True,
        name=f"voxtype-{m.name}-watcher",
    ).start()

    ready = await _wait_ready(m, _health_url(m))
    m.ready = ready
    if not ready:
        m.last_error = "service did not become ready"
    _notify(m.name)


def _watch_exit(m: _Managed) -> None:
    """Block on proc.wait() and schedule auto-restart on unexpected exit."""
    if m.proc is None:
        return
    try:
        code = m.proc.wait()
    except Exception:
        code = -1
    log.info("%s exited (code=%s)", m.name, code)
    m.ready = False
    m.proc = None
    _notify(m.name)
    if m.stopping:
        m.stopping = False
        return

    m.restart_count += 1
    delay = min(30.0, 1.0 * (2 ** min(m.restart_count, 5)))
    log.info("%s crashed — restart #%d in %.1fs", m.name, m.restart_count, delay)

    def _later():
        time.sleep(delay)
        # If the lazy `_ensure_*_running()` path already brought the service
        # back up while we were waiting, don't spawn a duplicate — it would
        # lose the port-bind race and trigger another crash-restart cycle.
        if m.proc is not None and m.proc.poll() is None:
            log.info("%s already running (PID %d) — skipping scheduled restart",
                     m.name, m.proc.pid)
            return
        if m.stopping:
            return
        try:
            asyncio.run(_start_internal(m))
        except Exception as exc:
            log.error("%s restart failed: %s", m.name, exc)

    threading.Thread(target=_later, daemon=True,
                     name=f"voxtype-{m.name}-restart").start()


def _coerce_cpu_if_broken(name: ServiceName, cfg) -> None:
    """If a previous launch saw CUDA/cuBLAS load errors, force device='cpu'
    on this config so the service can actually serve requests."""
    if not _gpu_broken.get(name):
        return
    if getattr(cfg, "device", None) == "gpu":
        log.info("[%s] coercing device → cpu (previous GPU failure)", name)
        try:
            cfg.device = "cpu"
        except Exception:
            pass


async def start_whisper(cfg: WhisperConfig) -> None:
    _coerce_cpu_if_broken("whisper", cfg)
    with _lock:
        m = _services.setdefault("whisper", _Managed(name="whisper"))
    if m.proc and m.proc.poll() is None:
        log.info("whisper already running")
        return
    m.config = cfg
    await _start_internal(m)


async def start_kokoro(cfg: KokoroConfig) -> None:
    _coerce_cpu_if_broken("kokoro", cfg)
    with _lock:
        m = _services.setdefault("kokoro", _Managed(name="kokoro"))
    if m.proc and m.proc.poll() is None:
        log.info("kokoro already running")
        return
    m.config = cfg
    await _start_internal(m)


def _kill_tree(pid: int, force: bool) -> None:
    args = ["taskkill.exe", "/PID", str(pid), "/T"]
    if force:
        args.append("/F")
    try:
        # CREATE_NO_WINDOW prevents a console flash when telecode is run
        # under pythonw.exe — taskkill is a console-subsystem exe, so
        # Windows would otherwise allocate a fresh console for it.
        subprocess.run(
            args, capture_output=True, timeout=5.0, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception:
        pass


async def _wait_exit(proc: subprocess.Popen, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            return True
        await asyncio.sleep(0.1)
    return proc.poll() is not None


async def stop_service(name: ServiceName) -> None:
    m = _services.get(name)
    if m is None:
        return
    m.stopping = True
    m.restart_count = 0
    # Capture proc locally: the _watch_exit thread can clear m.proc to None
    # the moment the child dies, and our subsequent awaits would then call
    # .poll() on None. Hold the Popen for the lifetime of this coroutine.
    proc = m.proc
    if proc is None or proc.poll() is not None:
        m.proc = None
        m.ready = False
        _notify(name)
        return
    pid = proc.pid
    log.info("stopping %s (PID %d)...", name, pid)
    _kill_tree(pid, force=False)
    if not await _wait_exit(proc, 3.0):
        log.info("%s did not exit gracefully — forceful kill", name)
        _kill_tree(pid, force=True)
        await _wait_exit(proc, 2.0)
    m.proc = None
    m.ready = False
    _notify(name)


async def restart_service(name: ServiceName,
                           new_cfg: WhisperConfig | KokoroConfig | None = None) -> None:
    m = _services.get(name)
    if new_cfg is not None and m is not None:
        m.config = new_cfg
    await stop_service(name)
    cfg = (m.config if m is not None else new_cfg)
    if cfg is None:
        log.warning("restart_service(%s): no config available", name)
        return
    if name == "whisper":
        await start_whisper(cfg)  # type: ignore[arg-type]
    else:
        await start_kokoro(cfg)  # type: ignore[arg-type]


async def stop_all() -> None:
    await asyncio.gather(*(stop_service(n) for n in list(_services.keys())))


def get_status(name: ServiceName) -> ServiceStatus:
    m = _services.get(name)
    return ServiceStatus(
        name=name,
        pid=(m.proc.pid if m and m.proc and m.proc.poll() is None else None),
        running=bool(m and m.proc and m.proc.poll() is None),
        ready=bool(m and m.ready),
        last_error=(m.last_error if m else ""),
    )


def is_running(name: ServiceName) -> bool:
    m = _services.get(name)
    return bool(m and m.proc and m.proc.poll() is None)
