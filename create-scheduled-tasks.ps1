#Requires -Version 5.1
#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Create Windows Task Scheduler entries for Whisper STT and Kokoro TTS
.DESCRIPTION
    Creates scheduled tasks that start the voice services on user login.
    Must be run as Administrator.
#>
param(
    [string]$InstallDir = "$env:USERPROFILE\.voicemode-windows"
)

$ErrorActionPreference = "Stop"
$username = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

# Whisper STT
$whisperBat = Join-Path $InstallDir "start-whisper-stt.bat"
if (Test-Path $whisperBat) {
    $action = New-ScheduledTaskAction -Execute $whisperBat
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $username
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 0)
    $principal = New-ScheduledTaskPrincipal -UserId $username -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskName "VoiceMode-Whisper-STT" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "[OK] Created task: VoiceMode-Whisper-STT" -ForegroundColor Green
}

# Kokoro TTS
$kokoroBat = Join-Path $InstallDir "start-kokoro-tts.bat"
if (Test-Path $kokoroBat) {
    $action = New-ScheduledTaskAction -Execute $kokoroBat
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $username
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 0)
    $principal = New-ScheduledTaskPrincipal -UserId $username -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskName "VoiceMode-Kokoro-TTS" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "[OK] Created task: VoiceMode-Kokoro-TTS" -ForegroundColor Green
}

Write-Host "`nScheduled tasks created. Services will start on next login." -ForegroundColor Cyan
Write-Host "To start now, run the .bat files in: $InstallDir" -ForegroundColor Yellow
