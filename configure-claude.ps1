#Requires -Version 5.1
<#
.SYNOPSIS
    Configure Claude Code to use local VoiceMode MCP server
#>
param(
    [string]$InstallDir = "$env:USERPROFILE\.voicemode-windows",
    [int]$WhisperPort = 6600,
    [int]$KokoroPort = 6500
)

$ErrorActionPreference = "Stop"
$claudeConfig = Join-Path $env:USERPROFILE ".claude.json"
$voiceModeExe = Join-Path $InstallDir "mcp-venv\Scripts\voice-mode.exe"

if (-not (Test-Path $claudeConfig)) {
    Write-Host "Claude Code config not found at $claudeConfig" -ForegroundColor Red
    Write-Host "Make sure Claude Code is installed and has been run at least once." -ForegroundColor Yellow
    exit 1
}

$config = Get-Content $claudeConfig -Raw | ConvertFrom-Json

# Ensure mcpServers exists
if (-not $config.mcpServers) {
    $config | Add-Member -NotePropertyName "mcpServers" -NotePropertyValue @{} -Force
}

# Add voicemode server
$voicemodeConfig = @{
    type = "stdio"
    command = $voiceModeExe.Replace('\', '\\')
    args = @()
    env = @{
        PYTHONIOENCODING = "utf-8"
        OPENAI_API_KEY = "sk-local-dummy"
        VOICEMODE_STT_BASE_URLS = "http://127.0.0.1:$WhisperPort/v1"
        VOICEMODE_TTS_BASE_URLS = "http://127.0.0.1:$KokoroPort/v1"
        VOICEMODE_KOKORO_PORT = "$KokoroPort"
        VOICEMODE_WHISPER_PORT = "$WhisperPort"
        VOICEMODE_DISABLE_SILENCE_DETECTION = "false"
        VOICEMODE_DEFAULT_LISTEN_DURATION = "30"
    }
}

$config.mcpServers | Add-Member -NotePropertyName "voicemode" -NotePropertyValue $voicemodeConfig -Force
$config | ConvertTo-Json -Depth 10 | Set-Content $claudeConfig -Encoding UTF8

Write-Host "    Claude Code configured with VoiceMode MCP" -ForegroundColor Green
Write-Host "    Restart Claude Code to activate." -ForegroundColor Yellow
