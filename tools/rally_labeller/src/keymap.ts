/**
 * Keyboard shortcuts, all on the left half of a QWERTY keyboard.
 *
 * Keys match by physical position (`KeyboardEvent.code`), so they stay put
 * whatever layout the operating system uses. The busiest actions sit under
 * the strong index finger: V/B mark the top/bottom player, F/G step
 * back/forward, R/T move between marks, and ] marks a break. Rarer actions
 * take the outer fingers: Q/W slow down/speed up, A/S move between suggested
 * contacts and Shift+A/S between suggested rallies. Every action also has an on-screen button showing its key.
 *
 * Holding F, G, R or T (with or without Shift on F and G) keeps moving. Every
 * other key acts once per press, so a held key can't add, delete or undo a
 * burst of marks.
 */

export type ActionName =
  | "playPause"
  | "markTop"
  | "markBottom"
  | "markBreak"
  | "markStart"
  | "markEnd"
  | "markSectionStart"
  | "markSectionEnd"
  | "stepBack"
  | "stepForward"
  | "jumpBack"
  | "jumpForward"
  | "faster"
  | "slower"
  | "selectPrevious"
  | "selectNext"
  | "deleteSelected"
  | "toggleUnsure"
  | "undo"
  | "redo"
  | "clearSelection"
  | "previousSuggestedContact"
  | "nextSuggestedContact"
  | "previousSuggestedRally"
  | "nextSuggestedRally"
  | "toggleSuggestionDone";

export interface KeyBinding {
  code: string;
  shift: boolean;
  action: ActionName;
  /** Key name shown on buttons and in the help panel. */
  keyLabel: string;
  /** Milliseconds between repeats while the key is held, or null when holding does nothing more. */
  repeatEveryMs: number | null;
}

/** How long a key is held before it starts repeating, so a quick tap moves once. */
export const REPEAT_DELAY_MS = 350;
// Slow enough to follow the picture: about 20 frame steps a second, 10 jumps a
// second, and 4 marks a second so you can read each one as it goes past.
const STEP_REPEAT_MS = 50;
const JUMP_REPEAT_MS = 100;
const MARK_REPEAT_MS = 250;

// One button to get back to normal speed; Slower and Faster halve and double.
export const SPEED_PRESETS = [1] as const;

export const ACTION_DESCRIPTIONS: Record<ActionName, string> = {
  playPause: "Play / pause",
  markTop: "Contact by the top (far) player. On a bottom contact: switch it to top",
  markBottom: "Contact by the bottom (near) player. On a top contact: switch it to bottom",
  markBreak: "Rally break: the next contact starts a new rally",
  markStart: "Rally start",
  markEnd: "Rally end (shuttle dead)",
  markSectionStart: "Start of a labelled section",
  markSectionEnd: "End of a labelled section",
  stepBack: "Back 1 frame",
  stepForward: "Forward 1 frame",
  jumpBack: "Jump back (frames per jump in Settings)",
  jumpForward: "Jump forward (frames per jump in Settings)",
  faster: "Faster: double the speed",
  slower: "Slower: halve the speed",
  selectPrevious: "Select the previous mark and go to it",
  selectNext: "Select the next mark and go to it",
  deleteSelected: "Delete the mark on this frame (the selected one, if several)",
  toggleUnsure: "Flag the mark on this frame, or else the last one placed, as unsure",
  undo: "Undo",
  redo: "Redo",
  clearSelection: "Clear the selection",
  previousSuggestedContact: "Go to the previous suggested contact",
  nextSuggestedContact: "Go to the next suggested contact",
  previousSuggestedRally: "Go to the previous suggested rally",
  nextSuggestedRally: "Go to the next suggested rally",
  toggleSuggestionDone: "Mark the suggested contact on this frame done (press again to undo)",
};

function bind(
  code: string,
  keyLabel: string,
  action: ActionName,
  options: { shift?: boolean; repeatEveryMs?: number } = {},
): KeyBinding {
  return { code, keyLabel, action, shift: options.shift ?? false, repeatEveryMs: options.repeatEveryMs ?? null };
}

/** Rally start/end and section marks are buttons only: they are rare, and fewer keys are easier to remember. */
export const KEY_BINDINGS: readonly KeyBinding[] = [
  bind("Space", "Space", "playPause"),
  bind("KeyV", "V", "markTop"),
  bind("KeyB", "B", "markBottom"),
  bind("KeyF", "F", "stepBack", { repeatEveryMs: STEP_REPEAT_MS }),
  bind("KeyG", "G", "stepForward", { repeatEveryMs: STEP_REPEAT_MS }),
  bind("KeyF", "⇧F", "jumpBack", { shift: true, repeatEveryMs: JUMP_REPEAT_MS }),
  bind("KeyG", "⇧G", "jumpForward", { shift: true, repeatEveryMs: JUMP_REPEAT_MS }),
  bind("KeyZ", "Z", "undo"),
  bind("KeyZ", "⇧Z", "redo", { shift: true }),
  bind("KeyQ", "Q", "slower"),
  bind("KeyW", "W", "faster"),
  bind("KeyR", "R", "selectPrevious", { repeatEveryMs: MARK_REPEAT_MS }),
  bind("KeyT", "T", "selectNext", { repeatEveryMs: MARK_REPEAT_MS }),
  bind("KeyX", "X", "deleteSelected"),
  bind("BracketRight", "]", "markBreak"),
  bind("KeyA", "A", "previousSuggestedContact"),
  bind("KeyS", "S", "nextSuggestedContact"),
  bind("KeyA", "⇧A", "previousSuggestedRally", { shift: true }),
  bind("KeyS", "⇧S", "nextSuggestedRally", { shift: true }),
  bind("KeyD", "D", "toggleSuggestionDone"),
  bind("Tab", "Tab", "toggleUnsure"),
  bind("Escape", "Esc", "clearSelection"),
];

export function keyLabelFor(action: ActionName): string | undefined {
  return KEY_BINDINGS.find((binding) => binding.action === action)?.keyLabel;
}

/**
 * The binding for a key press. Ctrl+Z and Ctrl+Shift+Z also undo and redo;
 * any other press with Ctrl, Alt or Meta held is left to the browser.
 */
export function bindingFor(event: Pick<KeyboardEvent, "code" | "shiftKey" | "ctrlKey" | "altKey" | "metaKey">) {
  const commandHeld = event.ctrlKey || event.metaKey;
  const commandUndo = commandHeld && !event.altKey && event.code === "KeyZ";
  if ((commandHeld || event.altKey) && !commandUndo) return undefined;
  return KEY_BINDINGS.find((binding) => binding.code === event.code && binding.shift === event.shiftKey);
}
