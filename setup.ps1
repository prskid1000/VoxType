#Requires -Version 5.1
<#
.SYNOPSIS
    VoiceMode Windows Setup - Local voice (STT + TTS) for Claude Code
.DESCRIPTION
    Sets up VoiceMode MCP with local Whisper STT and Kokoro TTS on Windows.
    All processing is local - no cloud APIs, full privacy.
.PARAMETER InstallDir
    Installation directory (default: ~/.voicemode-windows)
.PARAMETER WhisperPort
    Port for Whisper STT server (default: 6600)
.PARAMETER KokoroPort
    Port for Kokoro TTS server (default: 6500)
.PARAMETER WhisperModel
    Whisper model to use (default: Systran/faster-whisper-small)
.PARAMETER GpuSupport
    Install GPU support for Kokoro TTS (default: true)
.PARAMETER SkipKokoro
    Skip Kokoro TTS installation (default: false)
.PARAMETER SkipWhisper
    Skip Whisper STT installation (default: false)
#>
param(
    [string]$InstallDir = "$env:USERPROFILE\.voicemode-windows",
    [int]$WhisperPort = 6600,
    [int]$KokoroPort = 6500,
    [string]$WhisperModel = "Systran/faster-whisper-small",
    [bool]$GpuSupport = $true,
    [switch]$SkipKokoro,
    [switch]$SkipWhisper
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "    [FAIL] $msg" -ForegroundColor Red }

Write-Host @"

  VoiceMode Windows Setup
  Local STT (Whisper) + TTS (Kokoro) for Claude Code
  ====================================================

"@ -ForegroundColor Magenta

# --- Check prerequisites ---
Write-Step "Checking prerequisites"

# Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Fail "Python not found. Install Python 3.10+ from https://python.org"
    exit 1
}
$pyVer = python --version 2>&1
Write-Ok "Python: $pyVer"

# pip
python -m pip --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Fail "pip not found"
    exit 1
}
Write-Ok "pip available"

# ffmpeg
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Warn "ffmpeg not found - some audio features may not work"
} else {
    Write-Ok "ffmpeg available"
}

# GPU check
if ($GpuSupport) {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $gpuInfo = nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1
        Write-Ok "GPU: $gpuInfo"
    } else {
        Write-Warn "nvidia-smi not found - falling back to CPU"
        $GpuSupport = $false
    }
}

# --- Create installation directory ---
Write-Step "Creating installation directory: $InstallDir"
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Write-Ok "Directory ready"

# --- Create venv for VoiceMode MCP ---
Write-Step "Setting up VoiceMode MCP virtual environment"
$mcpVenv = Join-Path $InstallDir "mcp-venv"
if (-not (Test-Path "$mcpVenv\Scripts\python.exe")) {
    python -m venv $mcpVenv
    Write-Ok "Created venv: $mcpVenv"
} else {
    Write-Ok "Venv already exists: $mcpVenv"
}

& "$mcpVenv\Scripts\python.exe" -m pip install --upgrade pip --quiet 2>&1 | Out-Null
& "$mcpVenv\Scripts\pip.exe" install "setuptools<71" webrtcvad voice-mode --quiet 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to install voice-mode"
    exit 1
}
Write-Ok "VoiceMode MCP installed"

# --- Apply Windows patches ---
Write-Step "Applying Windows compatibility patches"
$patchScript = Join-Path $PSScriptRoot "patches\apply-patches.ps1"
& $patchScript -VenvPath $mcpVenv
Write-Ok "Patches applied"

# --- Setup Whisper STT ---
if (-not $SkipWhisper) {
    Write-Step "Setting up Whisper STT service"
    $sttVenv = Join-Path $InstallDir "stt-venv"
    if (-not (Test-Path "$sttVenv\Scripts\python.exe")) {
        python -m venv $sttVenv
    }
    & "$sttVenv\Scripts\python.exe" -m pip install --upgrade pip --quiet 2>&1 | Out-Null
    & "$sttVenv\Scripts\pip.exe" install faster-whisper-server --quiet 2>&1

    # Patch faster-whisper-server version lookup bug
    $apiFile = Join-Path $sttVenv "Lib\site-packages\faster_whisper_server\api.py"
    if (Test-Path $apiFile) {
        $content = Get-Content $apiFile -Raw
        if ($content -notmatch 'except FileNotFoundError') {
            $content = $content -replace '(with pyproject_path\.open\("rb"\) as f:\s+data = tomllib\.load\(f\)\s+return data\["project"\]\["version"\])',
                @"
try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except FileNotFoundError:
        return "0.0.2"
"@
            Set-Content $apiFile -Value $content -NoNewline
            Write-Ok "Patched faster-whisper-server version lookup"
        }
    }

    # Create startup script
    $whisperBat = Join-Path $InstallDir "start-whisper-stt.bat"
    @"
@echo off
title Whisper STT Server (port $WhisperPort)
"$sttVenv\Scripts\faster-whisper-server.exe" $WhisperModel --host 127.0.0.1 --port $WhisperPort
"@ | Set-Content $whisperBat
    Write-Ok "Whisper STT ready on port $WhisperPort"
}

# --- Setup Kokoro TTS ---
if (-not $SkipKokoro) {
    Write-Step "Setting up Kokoro TTS service"

    # Clone Kokoro-FastAPI
    $kokoroDir = Join-Path $InstallDir "Kokoro-FastAPI"
    if (-not (Test-Path "$kokoroDir\pyproject.toml")) {
        git clone --depth 1 https://github.com/remsky/Kokoro-FastAPI.git $kokoroDir 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Failed to clone Kokoro-FastAPI"
            exit 1
        }
    }
    Write-Ok "Kokoro-FastAPI cloned"

    # Create venv and install
    $ttsVenv = Join-Path $InstallDir "tts-venv"
    if (-not (Test-Path "$ttsVenv\Scripts\python.exe")) {
        python -m venv $ttsVenv
    }
    & "$ttsVenv\Scripts\python.exe" -m pip install --upgrade pip --quiet 2>&1 | Out-Null

    $extra = if ($GpuSupport) { "gpu" } else { "cpu" }
    Push-Location $kokoroDir
    & "$ttsVenv\Scripts\pip.exe" install -e ".[$extra]" --quiet 2>&1
    Pop-Location

    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to install Kokoro-FastAPI"
        exit 1
    }
    Write-Ok "Kokoro TTS installed ($extra mode)"

    # Download model
    $modelPath = Join-Path $kokoroDir "api\src\models\v1_0\kokoro-v1_0.pth"
    if (-not (Test-Path $modelPath)) {
        Write-Step "Downloading Kokoro model (313MB)..."
        & "$ttsVenv\Scripts\python.exe" "$kokoroDir\docker\scripts\download_model.py" --output "$kokoroDir\api\src\models\v1_0" 2>&1
        Write-Ok "Model downloaded"
    } else {
        Write-Ok "Model already downloaded"
    }

    # Create startup script
    $kokoroBat = Join-Path $InstallDir "start-kokoro-tts.bat"
    $gpuFlag = if ($GpuSupport) { "true" } else { "false" }
    @"
@echo off
title Kokoro TTS Server (port $KokoroPort)
set PYTHONUTF8=1
set USE_GPU=$gpuFlag
set USE_ONNX=false
set PROJECT_ROOT=$kokoroDir
set PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%\api
set MODEL_DIR=src\models
set VOICES_DIR=src\voices\v1_0
set WEB_PLAYER_PATH=%PROJECT_ROOT%\web
cd /d %PROJECT_ROOT%
"$ttsVenv\Scripts\uvicorn.exe" api.src.main:app --host 127.0.0.1 --port $KokoroPort
"@ | Set-Content $kokoroBat
    Write-Ok "Kokoro TTS ready on port $KokoroPort"
}

# --- Configure Claude Code MCP ---
Write-Step "Configuring Claude Code MCP server"
$configScript = Join-Path $PSScriptRoot "configure-claude.ps1"
& $configScript -InstallDir $InstallDir -WhisperPort $WhisperPort -KokoroPort $KokoroPort
Write-Ok "Claude Code configured"

# --- Summary ---
Write-Host @"

  ====================================================
  Setup Complete!
  ====================================================

  Services:
    Whisper STT : 127.0.0.1:$WhisperPort
    Kokoro TTS  : 127.0.0.1:$KokoroPort

  Start services:
    $InstallDir\start-whisper-stt.bat
    $InstallDir\start-kokoro-tts.bat

  Or use Task Scheduler (recommended):
    $InstallDir\create-scheduled-tasks.ps1

  Then restart Claude Code and use voice!

"@ -ForegroundColor Green
