#Requires -Version 5.1
<#
.SYNOPSIS
    Uninstall VoiceMode Windows setup
#>
param(
    [string]$InstallDir = "$env:USERPROFILE\.voicemode-windows"
)

Write-Host "`n  VoiceMode Windows Uninstaller" -ForegroundColor Cyan

# Remove scheduled tasks
try {
    Unregister-ScheduledTask -TaskName "VoiceMode-Whisper-STT" -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName "VoiceMode-Kokoro-TTS" -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "  [OK] Removed scheduled tasks" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] Could not remove scheduled tasks (may need admin)" -ForegroundColor Yellow
}

# Remove Claude Code MCP config
$claudeConfig = Join-Path $env:USERPROFILE ".claude.json"
if (Test-Path $claudeConfig) {
    $config = Get-Content $claudeConfig -Raw | ConvertFrom-Json
    if ($config.mcpServers.voicemode) {
        $config.mcpServers.PSObject.Properties.Remove("voicemode")
        $config | ConvertTo-Json -Depth 10 | Set-Content $claudeConfig -Encoding UTF8
        Write-Host "  [OK] Removed VoiceMode from Claude Code config" -ForegroundColor Green
    }
}

# Remove installation directory
if (Test-Path $InstallDir) {
    $confirm = Read-Host "  Delete $InstallDir ? (y/N)"
    if ($confirm -eq 'y') {
        Remove-Item -Recurse -Force $InstallDir
        Write-Host "  [OK] Removed $InstallDir" -ForegroundColor Green
    }
}

Write-Host "`n  Uninstall complete. Restart Claude Code." -ForegroundColor Green
