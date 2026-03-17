# VoiceMode Windows

Local voice input/output for [Claude Code](https://claude.ai/claude-code) on Windows. Fully offline STT (Whisper) + TTS (Kokoro) with GPU acceleration.

## What it does

- **Speech-to-Text**: Local [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) with OpenAI-compatible API
- **Text-to-Speech**: Local [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) with GPU support
- **MCP Integration**: Patched [VoiceMode](https://github.com/mbailey/voicemode) MCP server for Windows
- **No cloud APIs**: Everything runs locally, full privacy
- **Auto-start**: Task Scheduler integration for boot-time startup (hidden, no console window)

## Prerequisites

- Windows 10/11
- Python 3.10+ (3.12 recommended)
- Git
- ffmpeg (in PATH)
- NVIDIA GPU (optional, for Kokoro TTS acceleration)
- [Claude Code](https://claude.ai/claude-code) installed

## Quick Start

```powershell
# Clone this repo (any drive/location works)
git clone https://github.com/YOUR_USERNAME/voicemode-windows.git
cd voicemode-windows

# Run setup (PowerShell)
.\setup.ps1

# Or with custom ports
.\setup.ps1 -WhisperPort 6600 -KokoroPort 6500

# CPU-only (no GPU)
.\setup.ps1 -GpuSupport $false

# Custom install directory (default: %USERPROFILE%\.voicemode-windows)
.\setup.ps1 -InstallDir "D:\voicemode"
```

The setup script can be cloned to any drive/directory — it uses `$env:USERPROFILE` for the install location by default and resolves all paths dynamically.

## Manual Start

```powershell
# Start services (run each in a separate terminal)
%USERPROFILE%\.voicemode-windows\start-whisper-stt.bat
%USERPROFILE%\.voicemode-windows\start-kokoro-tts.bat

# Restart Claude Code, then use voice
```

## Auto-Start (Task Scheduler)

Services run hidden in the background with no console window. No password is saved — tasks run only when the user is logged on.

### Option 1: PowerShell script

```powershell
.\create-scheduled-tasks.ps1
```

### Option 2: PowerShell commands (copy-paste)

```powershell
$installDir = "$env:USERPROFILE\.voicemode-windows"
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

# --- Whisper STT ---
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$installDir\start-whisper-stt.bat`"" `
    -WorkingDirectory $installDir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -Hidden
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "VoiceMode-Whisper-STT" `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force

# --- Kokoro TTS ---
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$installDir\start-kokoro-tts.bat`"" `
    -WorkingDirectory $installDir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -Hidden
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "VoiceMode-Kokoro-TTS" `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force
```

### Start/stop tasks manually

```powershell
# Start
schtasks /run /tn VoiceMode-Whisper-STT
schtasks /run /tn VoiceMode-Kokoro-TTS

# Stop
schtasks /end /tn VoiceMode-Whisper-STT
schtasks /end /tn VoiceMode-Kokoro-TTS

# Remove
schtasks /delete /tn VoiceMode-Whisper-STT /f
schtasks /delete /tn VoiceMode-Kokoro-TTS /f
```

### Task settings explained

| Setting | Value | Why |
|---------|-------|-----|
| `-Hidden` | true | No console window visible |
| `-LogonType Interactive` | Run only when logged on | No password saved |
| `-ExecutionTimeLimit 0` | No time limit | Services run indefinitely |
| `-AllowStartIfOnBatteries` | true | Works on laptop battery |
| `-DontStopIfGoingOnBatteries` | true | Doesn't kill on unplug |
| `-RestartCount 3` | 3 retries | Auto-restart on crash |
| `-RestartInterval 1m` | 1 minute apart | Delay between retries |

## Usage in Claude Code

After setup and restarting Claude Code, use the `/mcp__voicemode__converse` command or invoke the `converse` tool:

```
# Start a voice conversation
/mcp__voicemode__converse
```

The tool will:
1. Speak the message via Kokoro TTS
2. Listen via your microphone (VAD auto-stops on silence)
3. Transcribe via local Whisper STT
4. Return the transcribed text

**STT only** (no TTS, just listen):
```
converse("listening", skip_tts=true, wait_for_response=true)
```

## Windows Patches

VoiceMode is built for Linux/macOS. This project applies these patches for Windows:

| File | Issue | Fix |
|------|-------|-----|
| `conch.py` | Uses `fcntl` (Unix-only) | Replaced with `msvcrt` for Windows file locking |
| `migration_helpers.py` | Uses `os.uname()` | Replaced with `platform.system()` |
| `model_install.py` | Uses `os.uname()` | Replaced with `platform.machine()` |
| `simple_failover.py` | Sends `response_format=text` | Changed to `json` (faster-whisper-server compat) |
| `simple_failover.py` | Sends `language=auto` | Removed (causes 500 on faster-whisper-server) |
| `converse.py` | Slow `scipy.signal.resample` in VAD loop | Replaced with fast numpy decimation |
| `faster_whisper_server/api.py` | Missing `pyproject.toml` in pip install | Added fallback version |

Patches are applied automatically during setup via `patches/apply-patches.py`.

## Configuration

Environment variables (set in `~/.claude.json` under `mcpServers.voicemode.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICEMODE_STT_BASE_URLS` | `http://127.0.0.1:6600/v1` | Whisper STT endpoint |
| `VOICEMODE_TTS_BASE_URLS` | `http://127.0.0.1:6500/v1` | Kokoro TTS endpoint |
| `VOICEMODE_DISABLE_SILENCE_DETECTION` | `false` | Disable VAD silence detection |
| `VOICEMODE_DEFAULT_LISTEN_DURATION` | `30` | Max recording duration (seconds) |
| `VOICEMODE_WHISPER_PORT` | `6600` | Whisper server port |
| `VOICEMODE_KOKORO_PORT` | `6500` | Kokoro server port |

## Architecture

```
Claude Code
    |
    v
VoiceMode MCP (patched for Windows)
    |
    +---> Kokoro TTS (GPU) --> Speaker
    |     localhost:6500
    |
    +---> Microphone --> Whisper STT
          localhost:6600
```

## Troubleshooting

### Services not starting
Check if ports are already in use:
```powershell
netstat -ano | findstr "6500 6600"
```

### Kokoro keeps stopping
Make sure you have **two separate** scheduled tasks (not one task with two actions). One combined task may kill both when either exits.

### No audio output
Check Windows sound settings and ensure the correct output device is selected.

### Microphone not working
Ensure microphone permissions are granted in Windows Settings > Privacy > Microphone.

### Recording freezes
Make sure the VAD resampling patch was applied:
```powershell
python patches\apply-patches.py "$env:USERPROFILE\.voicemode-windows\mcp-venv"
```

### STT returns empty
Try a larger Whisper model:
```powershell
.\setup.ps1 -WhisperModel "Systran/faster-whisper-medium"
```

### Re-apply patches after voice-mode update
```powershell
python patches\apply-patches.py "$env:USERPROFILE\.voicemode-windows\mcp-venv"
```

## Uninstall

```powershell
.\uninstall.ps1
```

## Credits

- [VoiceMode](https://github.com/mbailey/voicemode) by Mike Bailey
- [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) by fedirz
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) by remsky
- [Claude Code](https://claude.ai/claude-code) by Anthropic

## License

MIT
