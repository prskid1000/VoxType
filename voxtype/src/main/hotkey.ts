import { uIOhook, UiohookKey } from 'uiohook-napi';
import type { HotkeyMode, HotkeyCombo } from '../shared/types';

type HotkeyCallbacks = {
  onActivate: () => void;
  onDeactivate: () => void;
};

let mode: HotkeyMode = 'hold';
let combo: HotkeyCombo = { key1: 29, key2: 3675, label: 'Ctrl + Win' };
let callbacks: HotkeyCallbacks | null = null;
let isActive = false;

// Track all currently held keys
const heldKeys = new Set<number>();

// Capture mode: next two keys become the new hotkey
let captureResolve: ((combo: HotkeyCombo) => void) | null = null;
let captureReady = false; // becomes true once all pre-capture keys are released
const captureKeys: number[] = [];

export function setHotkeyMode(m: HotkeyMode) {
  mode = m;
}

export function setHotkeyCombo(c: HotkeyCombo) {
  combo = c;
}

/**
 * Enter capture mode: returns a promise that resolves with the next
 * two-key combo the user presses. During capture, normal hotkey
 * activation is disabled.
 */
export function captureHotkey(): Promise<HotkeyCombo> {
  return new Promise((resolve) => {
    captureKeys.length = 0;
    // If the user triggered capture via keyboard (menu accelerator, Enter),
    // some keys may still be held. Defer recording until everything is up.
    captureReady = heldKeys.size === 0;
    captureResolve = resolve;
  });
}

// Keycode → human label
const KEY_NAMES: Record<number, string> = {
  [UiohookKey.Ctrl]: 'Ctrl',
  [UiohookKey.CtrlRight]: 'RCtrl',
  [UiohookKey.Shift]: 'Shift',
  [UiohookKey.ShiftRight]: 'RShift',
  [UiohookKey.Alt]: 'Alt',
  [UiohookKey.AltRight]: 'RAlt',
  [UiohookKey.Meta]: 'Win',
  [UiohookKey.MetaRight]: 'RWin',
  [UiohookKey.Space]: 'Space',
  [UiohookKey.Enter]: 'Enter',
  [UiohookKey.Tab]: 'Tab',
  [UiohookKey.CapsLock]: 'CapsLock',
  [UiohookKey.Escape]: 'Esc',
  [UiohookKey.Backspace]: 'Backspace',
  [UiohookKey.Delete]: 'Delete',
  [UiohookKey.Insert]: 'Insert',
  [UiohookKey.Home]: 'Home',
  [UiohookKey.End]: 'End',
  [UiohookKey.PageUp]: 'PageUp',
  [UiohookKey.PageDown]: 'PageDown',
  [UiohookKey.ArrowUp]: 'Up',
  [UiohookKey.ArrowDown]: 'Down',
  [UiohookKey.ArrowLeft]: 'Left',
  [UiohookKey.ArrowRight]: 'Right',
  [UiohookKey.F1]: 'F1', [UiohookKey.F2]: 'F2', [UiohookKey.F3]: 'F3',
  [UiohookKey.F4]: 'F4', [UiohookKey.F5]: 'F5', [UiohookKey.F6]: 'F6',
  [UiohookKey.F7]: 'F7', [UiohookKey.F8]: 'F8', [UiohookKey.F9]: 'F9',
  [UiohookKey.F10]: 'F10', [UiohookKey.F11]: 'F11', [UiohookKey.F12]: 'F12',
  // Letters
  30: 'A', 48: 'B', 46: 'C', 32: 'D', 18: 'E', 33: 'F', 34: 'G',
  35: 'H', 23: 'I', 36: 'J', 37: 'K', 38: 'L', 50: 'M', 49: 'N',
  24: 'O', 25: 'P', 16: 'Q', 19: 'R', 31: 'S', 20: 'T', 22: 'U',
  47: 'V', 17: 'W', 45: 'X', 21: 'Y', 44: 'Z',
  // Numbers
  2: '1', 3: '2', 4: '3', 5: '4', 6: '5',
  7: '6', 8: '7', 9: '8', 10: '9', 11: '0',
  // Punctuation
  12: '-', 13: '=', 26: '[', 27: ']', 43: '\\',
  39: ';', 40: "'", 41: '`', 51: ',', 52: '.', 53: '/',
};

function keyName(code: number): string {
  return KEY_NAMES[code] || `Key${code}`;
}

// Normalize: treat left/right variants of modifiers as the same
function normalize(code: number): number {
  if (code === UiohookKey.CtrlRight) return UiohookKey.Ctrl;
  if (code === UiohookKey.ShiftRight) return UiohookKey.Shift;
  if (code === UiohookKey.AltRight) return UiohookKey.Alt;
  if (code === UiohookKey.MetaRight) return UiohookKey.Meta;
  return code;
}

export function startHotkeyListener(cbs: HotkeyCallbacks) {
  callbacks = cbs;

  uIOhook.on('keydown', (e) => {
    const key = normalize(e.keycode);
    heldKeys.add(key);

    // Capture mode
    if (captureResolve) {
      // Ignore keydowns until all pre-capture keys have been released
      if (!captureReady) return;
      if (!captureKeys.includes(key)) {
        captureKeys.push(key);
      }
      if (captureKeys.length >= 2) {
        const [k1, k2] = captureKeys;
        const label = `${keyName(k1)} + ${keyName(k2)}`;
        const result: HotkeyCombo = { key1: k1, key2: k2, label };
        captureKeys.length = 0;
        const resolve = captureResolve;
        captureResolve = null;
        resolve(result);
      }
      return;
    }

    // Normal mode: check if all combo keys are held
    const key2Held = combo.key2 === undefined || heldKeys.has(normalize(combo.key2));
    if (heldKeys.has(normalize(combo.key1)) && key2Held) {
      if (mode === 'hold') {
        if (!isActive) {
          isActive = true;
          callbacks?.onActivate();
        }
      } else {
        if (!isActive) {
          isActive = true;
          callbacks?.onActivate();
        } else {
          isActive = false;
          callbacks?.onDeactivate();
        }
      }
    }
  });

  uIOhook.on('keyup', (e) => {
    const key = normalize(e.keycode);
    heldKeys.delete(key);

    // Capture mode: once all pre-capture keys have been released, arm capture.
    if (captureResolve && !captureReady && heldKeys.size === 0) {
      captureReady = true;
      return;
    }

    // Capture mode: if user released all keys with exactly one captured,
    // finalize as a single-key hotkey.
    if (captureResolve && captureReady && captureKeys.length === 1 && heldKeys.size === 0) {
      const [k1] = captureKeys;
      const result: HotkeyCombo = { key1: k1, label: keyName(k1) };
      captureKeys.length = 0;
      const resolve = captureResolve;
      captureResolve = null;
      resolve(result);
      return;
    }

    // Hold mode: deactivate when any combo key is released
    if (mode === 'hold' && isActive) {
      const k2 = combo.key2 === undefined ? -1 : normalize(combo.key2);
      if (key === normalize(combo.key1) || key === k2) {
        isActive = false;
        callbacks?.onDeactivate();
      }
    }
  });

  uIOhook.start();
}

export function stopHotkeyListener() {
  uIOhook.stop();
  callbacks = null;
  isActive = false;
  heldKeys.clear();
}
