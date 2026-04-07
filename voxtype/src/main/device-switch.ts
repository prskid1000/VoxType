import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import os from 'os';

const execFileAsync = promisify(execFile);

const INSTALL_DIR = path.join(os.homedir(), '.voicemode-windows');
const KOKORO_BAT = path.join(INSTALL_DIR, 'start-kokoro-tts.bat');
const WHISPER_BAT = path.join(INSTALL_DIR, 'start-whisper-stt.bat');
const KOKORO_TASK = 'VoiceMode-Kokoro-TTS';
const WHISPER_TASK = 'VoiceMode-Whisper-STT';

export type DeviceMode = 'gpu' | 'cpu';

// ─── Kokoro TTS ─────────────────────────────────────────────────────

export function getKokoroDevice(): DeviceMode {
  try {
    const content = fs.readFileSync(KOKORO_BAT, 'utf-8');
    const match = content.match(/set USE_GPU=(\w+)/);
    return match?.[1] === 'true' ? 'gpu' : 'cpu';
  } catch {
    return 'gpu';
  }
}

export async function setKokoroDevice(device: DeviceMode): Promise<void> {
  const content = fs.readFileSync(KOKORO_BAT, 'utf-8');
  const updated = content.replace(
    /set USE_GPU=\w+/,
    `set USE_GPU=${device === 'gpu' ? 'true' : 'false'}`,
  );
  fs.writeFileSync(KOKORO_BAT, updated, 'utf-8');
  console.log(`[VoxType] Kokoro device set to: ${device}`);
  await restartTask(KOKORO_TASK, 'uvicorn');
}

// ─── Whisper STT ────────────────────────────────────────────────────

export function getWhisperDevice(): DeviceMode {
  try {
    const content = fs.readFileSync(WHISPER_BAT, 'utf-8');
    // CPU mode uses CUDA_VISIBLE_DEVICES=-1 to hide GPU
    return content.includes('CUDA_VISIBLE_DEVICES=-1') ? 'cpu' : 'gpu';
  } catch {
    return 'gpu';
  }
}

export async function setWhisperDevice(device: DeviceMode): Promise<void> {
  let content = fs.readFileSync(WHISPER_BAT, 'utf-8');

  // Remove existing CUDA_VISIBLE_DEVICES line if present
  content = content.replace(/set CUDA_VISIBLE_DEVICES=-1\r?\n/g, '');

  if (device === 'cpu') {
    // Insert CUDA_VISIBLE_DEVICES=-1 after @echo off line to force CPU
    content = content.replace(
      /(@echo off\r?\n)/,
      `$1set CUDA_VISIBLE_DEVICES=-1\n`,
    );
  }

  fs.writeFileSync(WHISPER_BAT, content, 'utf-8');
  console.log(`[VoxType] Whisper device set to: ${device}`);
  await restartTask(WHISPER_TASK, 'faster-whisper-server');
}

// ─── Shared: restart a scheduled task ───────────────────────────────

async function restartTask(taskName: string, processName: string): Promise<void> {
  try {
    await execFileAsync('powershell.exe', [
      '-NoProfile', '-NonInteractive', '-Command',
      [
        `Stop-ScheduledTask -TaskName '${taskName}' -ErrorAction SilentlyContinue`,
        `Get-Process -Name '${processName}' -ErrorAction SilentlyContinue | Stop-Process -Force`,
        `Start-Sleep -Seconds 2`,
        `Start-ScheduledTask -TaskName '${taskName}'`,
      ].join('; '),
    ], { timeout: 15000 });
    console.log(`[VoxType] ${taskName} restarted`);
  } catch (err) {
    console.error(`[VoxType] Failed to restart ${taskName}:`, err);
  }
}
