export type PillState = 'idle' | 'recording' | 'processing' | 'enhancing' | 'typing' | 'error';

export type HotkeyMode = 'hold' | 'toggle';

// One or two keycodes that must be held together to activate
export interface HotkeyCombo {
  key1: number; // uiohook keycode
  key2?: number; // uiohook keycode (optional — omit for single-key hotkey)
  label: string; // human-readable, e.g. "Ctrl + Win" or "F9"
}

export type DeviceMode = 'gpu' | 'cpu';

export interface AppSettings {
  hotkeyMode: HotkeyMode;
  enhanceEnabled: boolean;
  autoStopOnSilence: boolean;
  saveHistory: boolean;
  vadEnabled: boolean;
  appendMode: boolean;
  hotkey: HotkeyCombo;
  pillX: number;
  pillY: number;
  whisperUrl: string;
  lmStudioUrl: string;
  llmModel: string;
  preloadModel: boolean;
  autoUnloadMinutes: number;
  kokoroDevice: DeviceMode;
  whisperDevice: DeviceMode;
}

export const DEFAULT_SETTINGS: AppSettings = {
  hotkeyMode: 'hold',
  enhanceEnabled: true,
  autoStopOnSilence: true,
  saveHistory: true,
  vadEnabled: true,
  appendMode: false,
  hotkey: { key1: 29, key2: 3675, label: 'Ctrl + Win' },
  pillX: -1,
  pillY: -1,
  whisperUrl: 'http://127.0.0.1:6600',
  lmStudioUrl: 'http://127.0.0.1:1234',
  llmModel: '',
  preloadModel: true,
  autoUnloadMinutes: 0,
  kokoroDevice: 'gpu',
  whisperDevice: 'gpu',
};

// IPC channel names
export const IPC = {
  // Main → Renderer
  START_RECORDING: 'start-recording',
  STOP_RECORDING: 'stop-recording',
  STATE_CHANGE: 'state-change',
  ERROR: 'error',

  // Renderer → Main
  AUDIO_DATA: 'audio-data',
  GET_SETTINGS: 'get-settings',
  SET_SETTINGS: 'set-settings',
  CANCEL: 'cancel',
} as const;
