#Requires -Version 5.1
<#
.SYNOPSIS
    Apply Windows compatibility patches to VoiceMode
.DESCRIPTION
    Patches voice-mode package files to work on Windows:
    1. conch.py - Replace fcntl with msvcrt for file locking
    2. migration_helpers.py - Replace os.uname() with platform.system()
    3. model_install.py - Replace os.uname() with platform.machine()
    4. simple_failover.py - Fix response_format and language params for faster-whisper-server
    5. converse.py - Fix VAD resampling performance
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$VenvPath
)

$ErrorActionPreference = "Stop"
$sitePackages = Join-Path $VenvPath "Lib\site-packages\voice_mode"

function Patch-File($path, $patches) {
    if (-not (Test-Path $path)) {
        Write-Host "    [SKIP] File not found: $path" -ForegroundColor Yellow
        return
    }
    $content = Get-Content $path -Raw
    $modified = $false
    foreach ($patch in $patches) {
        if ($content -match [regex]::Escape($patch.Find)) {
            $content = $content.Replace($patch.Find, $patch.Replace)
            $modified = $true
        }
    }
    if ($modified) {
        Set-Content $path -Value $content -NoNewline
        Write-Host "    [OK] Patched: $(Split-Path $path -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "    [OK] Already patched: $(Split-Path $path -Leaf)" -ForegroundColor DarkGreen
    }
}

Write-Host "`n    Applying Windows patches to: $sitePackages" -ForegroundColor Cyan

# --- Patch 1: conch.py - Replace fcntl with msvcrt ---
$conchPath = Join-Path $sitePackages "conch.py"
if (Test-Path $conchPath) {
    $content = Get-Content $conchPath -Raw

    # Only patch if not already patched
    if ($content -match 'import fcntl' -and $content -notmatch 'import msvcrt') {
        # Replace the import
        $content = $content.Replace(
            "import fcntl`nimport json`nimport os",
            "import json`nimport os`nimport sys`n`nif sys.platform == `"win32`":`n    import msvcrt`nelse:`n    import fcntl"
        )

        # Replace acquire lock
        $content = $content.Replace(
            "            # Try to get exclusive lock (non-blocking)`n            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)",
            @"
            # Try to get exclusive lock (non-blocking)
            if sys.platform == "win32":
                os.lseek(self._fd, 0, os.SEEK_SET)
                os.write(self._fd, b'\0')
                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
"@
        )

        # Replace release lock
        $content = $content.Replace(
            "                fcntl.flock(self._fd, fcntl.LOCK_UN)`n                os.close(self._fd)",
            @"
                if sys.platform == "win32":
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
"@
        )

        Set-Content $conchPath -Value $content -NoNewline
        Write-Host "    [OK] Patched: conch.py (fcntl -> msvcrt)" -ForegroundColor Green
    } else {
        Write-Host "    [OK] Already patched: conch.py" -ForegroundColor DarkGreen
    }
}

# --- Patch 2: migration_helpers.py - os.uname() -> platform.system() ---
$migrationPath = Join-Path $sitePackages "utils\migration_helpers.py"
if (Test-Path $migrationPath) {
    $content = Get-Content $migrationPath -Raw
    if ($content -match 'os\.uname\(\)') {
        $content = $content.Replace("import os`nimport subprocess", "import os`nimport platform`nimport subprocess")
        $content = $content.Replace('os.uname().sysname', 'platform.system()')
        Set-Content $migrationPath -Value $content -NoNewline
        Write-Host "    [OK] Patched: migration_helpers.py (os.uname -> platform)" -ForegroundColor Green
    } else {
        Write-Host "    [OK] Already patched: migration_helpers.py" -ForegroundColor DarkGreen
    }
}

# --- Patch 3: model_install.py - os.uname() -> platform.machine() ---
$modelInstallPath = Join-Path $sitePackages "tools\whisper\model_install.py"
if (Test-Path $modelInstallPath) {
    $content = Get-Content $modelInstallPath -Raw
    if ($content -match 'os\.uname\(\)') {
        $content = $content.Replace("import os`nimport sys", "import os`nimport platform`nimport sys")
        $content = $content.Replace('os.uname().machine', 'platform.machine()')
        Set-Content $modelInstallPath -Value $content -NoNewline
        Write-Host "    [OK] Patched: model_install.py (os.uname -> platform)" -ForegroundColor Green
    } else {
        Write-Host "    [OK] Already patched: model_install.py" -ForegroundColor DarkGreen
    }
}

# --- Patch 4: simple_failover.py - Fix STT params for faster-whisper-server ---
$failoverPath = Join-Path $sitePackages "simple_failover.py"
if (Test-Path $failoverPath) {
    $content = Get-Content $failoverPath -Raw
    $modified = $false

    # Fix response_format: "text" -> "json"
    if ($content -match '"response_format": "text"') {
        $content = $content.Replace('"response_format": "text"', '"response_format": "json"')
        $modified = $true
    }

    # Remove language=auto (causes 500 on faster-whisper-server)
    $autoLangBlock = @"
            elif is_local_provider(base_url):
                # Local whisper.cpp with auto mode - must pass "auto" explicitly
                transcription_kwargs["language"] = "auto"
            # For OpenAI with "auto" - don't pass parameter (auto-detect by default)
"@
    $fixedLangBlock = @"
            # Omit language param for auto-detect (works for both OpenAI and faster-whisper-server)
"@
    if ($content.Contains('transcription_kwargs["language"] = "auto"')) {
        $content = $content.Replace($autoLangBlock, $fixedLangBlock)
        $modified = $true
    }

    if ($modified) {
        Set-Content $failoverPath -Value $content -NoNewline
        Write-Host "    [OK] Patched: simple_failover.py (STT params)" -ForegroundColor Green
    } else {
        Write-Host "    [OK] Already patched: simple_failover.py" -ForegroundColor DarkGreen
    }
}

# --- Patch 5: converse.py - Fix VAD resampling (scipy -> numpy decimation) ---
$conversePath = Join-Path $sitePackages "tools\converse.py"
if (Test-Path $conversePath) {
    $content = Get-Content $conversePath -Raw
    if ($content -match 'signal\.resample\(chunk_flat') {
        $oldResample = @"
                        # For VAD, we need to downsample from 24kHz to 16kHz
                        # Use scipy's resample for proper downsampling
                        from scipy import signal
                        # Calculate the number of samples we need after resampling
                        resampled_length = int(len(chunk_flat) * vad_sample_rate / SAMPLE_RATE)
                        vad_chunk = signal.resample(chunk_flat, resampled_length)
                        # Take exactly the number of samples VAD expects
                        vad_chunk = vad_chunk[:vad_chunk_samples].astype(np.int16)
                        chunk_bytes = vad_chunk.tobytes()
"@
        $newResample = @"
                        # For VAD, we need to downsample from 24kHz to 16kHz
                        # Use simple decimation (take every Nth sample) for speed
                        ratio = SAMPLE_RATE / vad_sample_rate  # 1.5 for 24k->16k
                        indices = np.round(np.arange(vad_chunk_samples) * ratio).astype(int)
                        indices = np.clip(indices, 0, len(chunk_flat) - 1)
                        vad_chunk = chunk_flat[indices].astype(np.int16)
                        chunk_bytes = vad_chunk.tobytes()
"@
        $content = $content.Replace($oldResample, $newResample)
        Set-Content $conversePath -Value $content -NoNewline
        Write-Host "    [OK] Patched: converse.py (VAD resampling)" -ForegroundColor Green
    } else {
        Write-Host "    [OK] Already patched: converse.py" -ForegroundColor DarkGreen
    }
}

Write-Host "`n    All patches applied successfully!" -ForegroundColor Green
