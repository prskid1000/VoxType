import { app, Menu, Tray, BrowserWindow, nativeImage, NativeImage, clipboard, dialog, screen } from 'electron';
import path from 'path';
import type { AppSettings } from '../shared/types';
import { setHotkeyMode, setHotkeyCombo, captureHotkey } from './hotkey';
import { getEntries, clearHistory } from './history';
import { WHISPER_MODELS, getCurrentModel, switchModel } from './whisper-model';
import { FEATURED_VOICES, getCurrentVoice, setVoice } from './kokoro-voice';
import { getAvailableModels, getCurrentLLMModel, setLLMModel, fetchModels, preloadCurrentModel, resetAutoUnloadTimer, stopAutoUnloadTimer } from './llm';
import { getKokoroDevice, setKokoroDevice, getWhisperDevice, setWhisperDevice, type DeviceMode } from './device-switch';

let tray: Tray | null = null;

export function createTray(
  mainWindow: BrowserWindow,
  getSettings: () => AppSettings,
  updateSettings: (s: Partial<AppSettings>) => void,
) {
  const iconPath = path.join(__dirname, '../../../resources/icon.png');

  let icon: NativeImage;
  try {
    icon = nativeImage.createFromPath(iconPath);
    if (icon.isEmpty()) throw new Error('empty');
  } catch {
    icon = nativeImage.createFromBuffer(Buffer.alloc(16 * 16 * 4, 0), { width: 16, height: 16 });
  }

  tray = new Tray(icon.resize({ width: 16, height: 16 }));
  tray.setToolTip('VoxType');

  function rebuildMenu() {
    const s = getSettings();
    const history = getEntries();

    const historyItems: Electron.MenuItemConstructorOptions[] = history.slice(0, 10).map((entry) => {
      const label = entry.enhanced.length > 50
        ? entry.enhanced.substring(0, 50) + '...'
        : entry.enhanced;
      const time = new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      return {
        label: `${time}  ${label}`,
        click: () => clipboard.writeText(entry.enhanced),
      };
    });

    if (historyItems.length === 0) {
      historyItems.push({ label: 'No history yet', enabled: false });
    } else {
      historyItems.push({ type: 'separator' });
      historyItems.push({ label: 'Clear history', click: () => { clearHistory(); rebuildMenu(); } });
    }

    const menu = Menu.buildFromTemplate([
      {
        label: 'Hold to talk',
        type: 'radio',
        checked: s.hotkeyMode === 'hold',
        click: () => { updateSettings({ hotkeyMode: 'hold' }); setHotkeyMode('hold'); rebuildMenu(); },
      },
      {
        label: 'Toggle on / off',
        type: 'radio',
        checked: s.hotkeyMode === 'toggle',
        click: () => { updateSettings({ hotkeyMode: 'toggle' }); setHotkeyMode('toggle'); rebuildMenu(); },
      },
      { type: 'separator' },
      {
        label: `Hotkey: ${s.hotkey.label}`,
        click: async () => {
          // Show a small notification that we're listening
          tray?.setToolTip('Press two keys together...');
          const newCombo = await captureHotkey();
          updateSettings({ hotkey: newCombo });
          setHotkeyCombo(newCombo);
          tray?.setToolTip('VoxType');
          rebuildMenu();
        },
      },
      { type: 'separator' },
      {
        label: 'Whisper model',
        submenu: (() => {
          const current = getCurrentModel();
          return WHISPER_MODELS.map((m) => ({
            label: m.label,
            type: 'radio' as const,
            checked: current === m.id,
            click: async () => {
              await switchModel(m.id);
              rebuildMenu();
            },
          }));
        })(),
      },
      {
        label: 'Kokoro voice',
        submenu: (() => {
          const current = getCurrentVoice();
          return FEATURED_VOICES.map((v) => ({
            label: v.label,
            type: 'radio' as const,
            checked: current === v.id,
            click: () => {
              setVoice(v.id);
              rebuildMenu();
            },
          }));
        })(),
      },
      {
        label: 'Device (GPU/CPU)',
        submenu: (() => {
          const kokoroDev = getKokoroDevice();
          const whisperDev = getWhisperDevice();
          return [
            { label: 'Kokoro TTS', enabled: false },
            {
              label: '  GPU',
              type: 'radio' as const,
              checked: kokoroDev === 'gpu',
              click: async () => { await setKokoroDevice('gpu'); updateSettings({ kokoroDevice: 'gpu' }); rebuildMenu(); },
            },
            {
              label: '  CPU',
              type: 'radio' as const,
              checked: kokoroDev === 'cpu',
              click: async () => { await setKokoroDevice('cpu'); updateSettings({ kokoroDevice: 'cpu' }); rebuildMenu(); },
            },
            { type: 'separator' as const },
            { label: 'Whisper STT', enabled: false },
            {
              label: '  GPU',
              type: 'radio' as const,
              checked: whisperDev === 'gpu',
              click: async () => { await setWhisperDevice('gpu'); updateSettings({ whisperDevice: 'gpu' }); rebuildMenu(); },
            },
            {
              label: '  CPU',
              type: 'radio' as const,
              checked: whisperDev === 'cpu',
              click: async () => { await setWhisperDevice('cpu'); updateSettings({ whisperDevice: 'cpu' }); rebuildMenu(); },
            },
          ];
        })(),
      },
      { type: 'separator' },
      {
        label: 'LLM enhance',
        type: 'checkbox',
        checked: s.enhanceEnabled,
        click: (item: any) => { updateSettings({ enhanceEnabled: item.checked }); rebuildMenu(); },
      },
      {
        label: 'LLM model',
        submenu: (() => {
          const models = getAvailableModels();
          const current = getCurrentLLMModel();
          const unloadOptions = [0, 5, 10, 15, 30, 60];
          const unloadSubmenu: Electron.MenuItemConstructorOptions[] = unloadOptions.map((mins) => ({
            label: mins === 0 ? 'Disabled' : `${mins} min`,
            type: 'radio' as const,
            checked: s.autoUnloadMinutes === mins,
            click: () => {
              updateSettings({ autoUnloadMinutes: mins });
              if (mins > 0) {
                resetAutoUnloadTimer(mins, s.lmStudioUrl, s.whisperUrl);
              } else {
                stopAutoUnloadTimer();
              }
              rebuildMenu();
            },
          }));
          if (models.length === 0) {
            return [
              { label: 'No models found', enabled: false },
              { label: 'Auto-unload after', submenu: unloadSubmenu },
              { label: 'Preload on startup', type: 'checkbox' as const, checked: s.preloadModel, click: (item: any) => { updateSettings({ preloadModel: item.checked }); rebuildMenu(); } },
              { label: 'Refresh', click: async () => { await fetchModels(s.lmStudioUrl); rebuildMenu(); } },
            ];
          }
          const items: Electron.MenuItemConstructorOptions[] = models.map((m) => ({
            label: `${m.id}${m.state === 'loaded' ? ' (loaded)' : ''}`,
            type: 'radio' as const,
            checked: current === m.id,
            click: () => { setLLMModel(m.id); updateSettings({ llmModel: m.id }); rebuildMenu(); },
          }));
          items.push({ type: 'separator' });
          items.push({ label: 'Auto-unload after', submenu: unloadSubmenu });
          items.push({ label: 'Preload on startup', type: 'checkbox' as const, checked: s.preloadModel, click: (item: any) => { updateSettings({ preloadModel: item.checked }); rebuildMenu(); } });
          items.push({ label: 'Refresh models', click: async () => { await fetchModels(s.lmStudioUrl); rebuildMenu(); } });
          return items;
        })(),
      },
      {
        label: 'Append mode',
        type: 'checkbox',
        checked: s.appendMode,
        click: (item) => { updateSettings({ appendMode: item.checked }); rebuildMenu(); },
      },
      {
        label: 'Auto-stop on silence',
        type: 'checkbox',
        checked: s.autoStopOnSilence,
        click: (item) => { updateSettings({ autoStopOnSilence: item.checked }); rebuildMenu(); },
      },
      {
        label: 'Skip silence (VAD)',
        type: 'checkbox',
        checked: s.vadEnabled,
        click: (item) => { updateSettings({ vadEnabled: item.checked }); rebuildMenu(); },
      },
      {
        label: 'Save history',
        type: 'checkbox',
        checked: s.saveHistory,
        click: (item) => { updateSettings({ saveHistory: item.checked }); rebuildMenu(); },
      },
      { type: 'separator' },
      { label: 'History', submenu: historyItems },
      { type: 'separator' },
      {
        label: mainWindow.isVisible() ? 'Hide pill' : 'Show pill',
        click: () => { mainWindow.isVisible() ? mainWindow.hide() : mainWindow.show(); rebuildMenu(); },
      },
      {
        label: 'Reset pill position',
        click: () => {
          updateSettings({ pillX: -1, pillY: -1 });
          const { width, height } = screen.getPrimaryDisplay().workAreaSize;
          mainWindow.setPosition(Math.round(width / 2 - 100), height - 100);
          rebuildMenu();
        },
      },
      { label: 'Quit', click: () => app.quit() },
    ]);
    tray?.setContextMenu(menu);
  }

  rebuildMenu();
  tray.on('click', () => { mainWindow.isVisible() ? mainWindow.hide() : mainWindow.show(); rebuildMenu(); });

  return { rebuildMenu };
}
