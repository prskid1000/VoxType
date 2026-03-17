#Requires -Version 5.1
<#
.SYNOPSIS
    Create Windows Task Scheduler entries for Whisper STT and Kokoro TTS
.DESCRIPTION
    Creates two separate scheduled tasks that start voice services on user login.
    - Runs hidden (no console window)
    - No password required (runs only when user is logged on)
    - Restarts on failure (3 retries, 1 minute apart)
    - Runs on battery power
    - No execution time limit
#>
param(
    [string]$InstallDir = "$env:USERPROFILE\.voicemode-windows"
)

$ErrorActionPreference = "SilentlyContinue"
$username = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

# Whisper STT
$whisperVbs = Join-Path $InstallDir "start-whisper-stt.vbs"
if (Test-Path $whisperVbs) {
    $action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$whisperVbs`"" -WorkingDirectory $InstallDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $username
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -StartWhenAvailable `
        -Hidden
    $principal = New-ScheduledTaskPrincipal -UserId $username -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskName "VoiceMode-Whisper-STT" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    if ($?) {
        Write-Host "[OK] Created task: VoiceMode-Whisper-STT" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Could not create VoiceMode-Whisper-STT (try running as Admin)" -ForegroundColor Red
    }
}

# Kokoro TTS
$kokoroVbs = Join-Path $InstallDir "start-kokoro-tts.vbs"
if (Test-Path $kokoroVbs) {
    $action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$kokoroVbs`"" -WorkingDirectory $InstallDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $username
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -StartWhenAvailable `
        -Hidden
    $principal = New-ScheduledTaskPrincipal -UserId $username -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskName "VoiceMode-Kokoro-TTS" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    if ($?) {
        Write-Host "[OK] Created task: VoiceMode-Kokoro-TTS" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Could not create VoiceMode-Kokoro-TTS (try running as Admin)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Tasks created. Services will auto-start on next login (hidden)." -ForegroundColor Cyan
Write-Host "To start now:" -ForegroundColor Yellow
Write-Host "  schtasks /run /tn VoiceMode-Whisper-STT" -ForegroundColor White
Write-Host "  schtasks /run /tn VoiceMode-Kokoro-TTS" -ForegroundColor White
