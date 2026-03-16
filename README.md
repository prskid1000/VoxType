# VoiceMode Windows

Local voice input/output for [Claude Code](https://claude.ai/claude-code) on Windows. Fully offline STT (Whisper) + TTS (Kokoro) with GPU acceleration.

## What it does

- **Speech-to-Text**: Local [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) with OpenAI-compatible API
- **Text-to-Speech**: Local [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) with GPU support
- **MCP Integration**: Patched [VoiceMode](https://github.com/mbailey/voicemode) MCP server for Windows
- **No cloud APIs**: Everything runs locally, full privacy
- **Auto-start**: Task Scheduler integration for boot-time startup

## Prerequisites

- Windows 10/11
- Python 3.10+ (3.12 recommended)
- Git
- ffmpeg (in PATH)
- NVIDIA GPU (optional, for Kokoro TTS acceleration)
- [Claude Code](https://claude.ai/claude-code) installed

## Quick Start

```powershell
# Clone this repo
git clone https://github.com/YOUR_USERNAME/voicemode-windows.git
cd voicemode-windows

# Run setup (PowerShell)
.\setup.ps1

# Or with custom ports
.\setup.ps1 -WhisperPort 6600 -KokoroPort 6500

# CPU-only (no GPU)
.\setup.ps1 -GpuSupport $false
```

## Manual Start

```powershell
# Start services (run each in a separate terminal)
~\.voicemode-windows\start-whisper-stt.bat
~\.voicemode-windows\start-kokoro-tts.bat

# Restart Claude Code, then use voice
```

## Auto-Start (Task Scheduler)

```powershell
# Run as Administrator
.\create-scheduled-tasks.ps1
```

## Usage in Claude Code

After setup and restarting Claude Code, use the `/mcp__voicemode__converse` command or invoke the `converse` tool:

```
# Start a voice conversation
/mcp__voicemode__converse
```

The tool will:
1. Speak the message via Kokoro TTS
2. Listen via your microphone
3. Transcribe via Whisper STT
4. Return the transcribed text

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

### No audio output
Check Windows sound settings and ensure the correct output device is selected.

### Microphone not working
Ensure microphone permissions are granted in Windows Settings > Privacy > Microphone.

### STT returns empty
Try a larger Whisper model:
```powershell
.\setup.ps1 -WhisperModel "Systran/faster-whisper-medium"
```

### Re-apply patches after voice-mode update
```powershell
.\patches\apply-patches.ps1 -VenvPath "$env:USERPROFILE\.voicemode-windows\mcp-venv"
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
