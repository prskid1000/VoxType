#Requires -Version 5.1
<#
.SYNOPSIS
    VoxType Setup — local voice dictation overlay for Windows.
.DESCRIPTION
    Installs Whisper STT and Kokoro TTS as Python venvs inside the install
    directory, builds the VoxType Electron app, and registers a single
    scheduled task (VoxType-Dictation) that auto-starts at logon. VoxType
    itself owns the Whisper and Kokoro child processes — no separate
    scheduled tasks, no .bat/.vbs wrapper scripts.
.PARAMETER InstallDir
    Where everything lives. Defaults to ~/.voicemode-windows (the repo dir
    when this script is run from a clone).
.PARAMETER WhisperModel
    Initial Whisper model. VoxType can switch later from the tray.
.PARAMETER GpuSupport
    Install PyTorch with CUDA. Set to $false for CPU-only Kokoro.
.PARAMETER SkipKokoro
    Skip the Kokoro install (~3 GB of PyTorch + model). VoxType still works
    for dictation; Kokoro is optional.
#>
param(
    [string]$InstallDir   = "$env:USERPROFILE\.voicemode-windows",
    [string]$WhisperModel = "Systran/faster-whisper-small",
    [bool]  $GpuSupport   = $true,
    [switch]$SkipKokoro
)

$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    [OK] $msg"   -ForegroundColor Green }
function Warn($msg) { Write-Host "    [WARN] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "    [FAIL] $msg" -ForegroundColor Red; exit 1 }

Write-Host @"

  VoxType Setup
  Local voice dictation for Windows
  =========================================

"@ -ForegroundColor Magenta

# ─── Prerequisites ──────────────────────────────────────────────────

Step "Checking prerequisites"

# Find a working Python 3.10+
$pythonExe = $null
$candidates = @()
foreach ($name in @("python3.exe", "python.exe")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { $candidates += $cmd.Source }
}
if (Get-Command py.exe -ErrorAction SilentlyContinue) { $candidates += "py.exe" }
$pyenvRoot = "$env:USERPROFILE\.pyenv\pyenv-win\versions"
if (Test-Path $pyenvRoot) {
    Get-ChildItem $pyenvRoot -Directory | Sort-Object Name -Descending | ForEach-Object {
        $p = Join-Path $_.FullName "python.exe"
        if (Test-Path $p) { $candidates += $p }
    }
}
foreach ($p in @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
)) { if (Test-Path $p) { $candidates += $p } }

foreach ($c in $candidates) {
    try {
        $ver = if ($c -eq "py.exe") { py -3 --version 2>&1 } else { & $c --version 2>&1 }
        if ($ver -match 'Python 3\.(1[0-9]|[2-9][0-9])') { $pythonExe = $c; break }
    } catch {}
}
if (-not $pythonExe) { Fail "Python 3.10+ not found. Install from https://python.org" }
Ok "Python: $(& $pythonExe --version 2>&1) ($pythonExe)"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Fail "Node.js not found. Install from https://nodejs.org (18+)"
}
Ok "Node: $(node --version)"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git not found. Install from https://git-scm.com"
}
Ok "git available"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Warn "ffmpeg not found — some audio features may degrade"
} else {
    Ok "ffmpeg available"
}

if ($GpuSupport) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | Select-Object -First 1
        Ok "GPU: $gpuInfo"
    } else {
        Warn "nvidia-smi not found — falling back to CPU"
        $GpuSupport = $false
    }
}

# ─── Install dir ─────────────────────────────────────────────────────

Step "Install directory: $InstallDir"
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Ok "Ready"

# ─── Whisper STT venv ────────────────────────────────────────────────

Step "Installing Whisper STT"
$sttVenv    = Join-Path $InstallDir "stt-venv"
$whisperExe = Join-Path $sttVenv "Scripts\faster-whisper-server.exe"
$apiFile    = Join-Path $sttVenv "Lib\site-packages\faster_whisper_server\api.py"

if (-not (Test-Path "$sttVenv\Scripts\python.exe")) {
    & $pythonExe -m venv $sttVenv
}

# pip install is idempotent — it prints "Requirement already satisfied" and
# exits in seconds when nothing needs downloading. Always run so version
# updates get picked up.
Write-Host "    pip install (skips download if up-to-date)..." -ForegroundColor DarkGray
& "$sttVenv\Scripts\python.exe" -m pip install --upgrade pip --quiet 2>&1 | Out-Null
& "$sttVenv\Scripts\pip.exe" install faster-whisper-server --quiet 2>&1 | Out-Null

# Patch faster-whisper-server's tomllib lookup (PyPI packaging quirk).
# This is a code edit, not pip — needs its own idempotency check.
if (Test-Path $apiFile) {
    $content = Get-Content $apiFile -Raw
    if ($content -notmatch 'except FileNotFoundError') {
        $patched = $content -replace `
            '(with pyproject_path\.open\("rb"\) as f:\s+data = tomllib\.load\(f\)\s+return data\["project"\]\["version"\])', `
@"
try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except FileNotFoundError:
        return "0.0.2"
"@
        Set-Content $apiFile -Value $patched -NoNewline
        Ok "Patched faster-whisper-server version lookup"
    }
}

if (-not (Test-Path $whisperExe)) { Fail "Whisper install failed" }
Ok "Whisper ready (model downloads to ~/.cache/huggingface on first dictation)"

# ─── Kokoro TTS venv (optional) ──────────────────────────────────────

if (-not $SkipKokoro) {
    Step "Installing Kokoro TTS"
    $kokoroDir  = Join-Path $InstallDir "Kokoro-FastAPI"
    $ttsVenv    = Join-Path $InstallDir "tts-venv"
    $uvicornExe = Join-Path $ttsVenv "Scripts\uvicorn.exe"
    $modelPath  = Join-Path $kokoroDir "api\src\models\v1_0\kokoro-v1_0.pth"

    # 1. Clone Kokoro-FastAPI (skip if already there)
    if (-not (Test-Path "$kokoroDir\pyproject.toml")) {
        if (Test-Path $kokoroDir) { Remove-Item -Recurse -Force $kokoroDir }
        git clone --depth 1 https://github.com/remsky/Kokoro-FastAPI.git $kokoroDir 2>&1 | Out-Null
        if (-not (Test-Path "$kokoroDir\pyproject.toml")) { Fail "Failed to clone Kokoro-FastAPI" }
    }

    # 2. Create venv (skip if exists)
    if (-not (Test-Path "$ttsVenv\Scripts\python.exe")) {
        & $pythonExe -m venv $ttsVenv
    }

    # 3. pip install — idempotent. Fast on rerun (no download if installed),
    # slow only on first install or when a new torch/Kokoro version exists.
    Write-Host "    pip install (skips download if up-to-date — first run is multi-GB)..." -ForegroundColor DarkGray
    & "$ttsVenv\Scripts\python.exe" -m pip install --upgrade pip --quiet 2>&1 | Out-Null

    if ($GpuSupport) {
        & "$ttsVenv\Scripts\pip.exe" install torch --index-url https://download.pytorch.org/whl/cu129 --quiet 2>&1 | Out-Null
    } else {
        & "$ttsVenv\Scripts\pip.exe" install torch --index-url https://download.pytorch.org/whl/cpu --quiet 2>&1 | Out-Null
    }

    Push-Location $kokoroDir
    & "$ttsVenv\Scripts\pip.exe" install -e . --quiet 2>&1 | Out-Null
    Pop-Location

    if (-not (Test-Path $uvicornExe)) { Fail "Kokoro install failed" }

    # 4. Model download — file existence check (not a pip dep)
    if (-not (Test-Path $modelPath)) {
        Write-Host "    Downloading Kokoro model (313 MB)..." -ForegroundColor DarkGray
        & "$ttsVenv\Scripts\python.exe" "$kokoroDir\docker\scripts\download_model.py" `
            --output "$kokoroDir\api\src\models\v1_0" 2>&1 | Out-Null
        if (-not (Test-Path $modelPath)) { Fail "Failed to download Kokoro model" }
    }
    Ok "Kokoro ready (off by default — enable from VoxType tray)"
} else {
    Warn "Skipping Kokoro install (per -SkipKokoro)"
}

# ─── VoxType build (in-place — repo IS the install dir) ──────────────

Step "Building VoxType"
$voxTypeDir = Join-Path $InstallDir "voxtype"
if (-not (Test-Path "$voxTypeDir\package.json")) {
    Fail "voxtype/ not found at $voxTypeDir — run setup.ps1 from the repo root"
}

Push-Location $voxTypeDir
Write-Host "    npm install..." -ForegroundColor DarkGray
npm install --silent 2>&1 | Out-Null
if (-not (Test-Path "node_modules\.bin\electron.cmd")) { Pop-Location; Fail "npm install failed" }

Write-Host "    Compiling..." -ForegroundColor DarkGray
npx tsc -p tsconfig.node.json 2>&1 | Out-Null
npx vite build 2>&1 | Out-Null
if (-not (Test-Path "dist\main\main\index.js")) { Pop-Location; Fail "VoxType build failed" }
Pop-Location
Ok "VoxType built in place ($voxTypeDir)"

# ─── Single scheduled task ───────────────────────────────────────────

Step "Registering scheduled task: VoxType-Dictation"

# electron.exe is a GUI binary — no console window appears, so we don't
# need a .vbs/.bat wrapper. The task runs at logon, hidden, with restart
# on crash. VoxType then spawns Whisper/Kokoro as child processes itself.
$electronExe = Join-Path $voxTypeDir "node_modules\electron\dist\electron.exe"
$entryPoint  = Join-Path $voxTypeDir "dist\main\main\index.js"
if (-not (Test-Path $electronExe)) { Fail "electron.exe missing at $electronExe" }
if (-not (Test-Path $entryPoint))  { Fail "VoxType entry missing at $entryPoint" }

$taskName = 'VoxType-Dictation'
$username = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

# Tear down any existing task (idempotent install)
Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | ForEach-Object {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# Tear down legacy tasks left over from the old multi-task layout
foreach ($legacy in @('VoiceMode-Whisper-STT', 'VoiceMode-Kokoro-TTS')) {
    if (Get-ScheduledTask -TaskName $legacy -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $legacy -Confirm:$false
        Ok "Removed legacy task: $legacy"
    }
}

$action    = New-ScheduledTaskAction -Execute $electronExe -Argument "`"$entryPoint`"" -WorkingDirectory $voxTypeDir
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $username
$settings  = New-ScheduledTaskSettingsSet `
                -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries `
                -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
                -RestartCount 3 `
                -RestartInterval (New-TimeSpan -Minutes 1) `
                -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId $username -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null
Ok "Scheduled task registered (auto-start at logon)"

# ─── Start now ───────────────────────────────────────────────────────

Step "Starting VoxType"
Start-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
Ok "Running"

# ─── Done ────────────────────────────────────────────────────────────

Write-Host @"

  =========================================
  Setup complete!
  =========================================

  VoxType is running. Look for the tray icon (bottom-right).
  Press Ctrl+Win to dictate into any app.

  Whisper auto-starts with VoxType. Kokoro is OFF by default —
  enable it from tray > Services > Kokoro if you want TTS.

  Logs:
    %USERPROFILE%\.voxtype\debug.log     (cleared on each start)
    %USERPROFILE%\.voxtype\sessions.jsonl
    %USERPROFILE%\.voxtype\history.json  (user-visible history)

"@ -ForegroundColor Green
