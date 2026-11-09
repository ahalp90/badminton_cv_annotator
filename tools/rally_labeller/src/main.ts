/** Page wiring: one labelling session per open video. */

import "./style.css";

import { FramePlayer } from "./framePlayer.ts";
import {
  LOUPE_DEFAULTS,
  changeLoupeZoom,
  hideLoupe,
  loupeShown,
  moveLoupe,
  redrawLoupe,
  setLoupeOptions,
  showLoupe,
} from "./loupe.ts";
import {
  ACTION_DESCRIPTIONS,
  KEY_BINDINGS,
  REPEAT_DELAY_MS,
  SPEED_PRESETS,
  bindingFor,
  keyLabelFor,
  type ActionName,
} from "./keymap.ts";
import {
  fpsValue,
  videoTitle,
  type LabelFile,
  type SaveRequest,
  type SourceSummary,
  type SuggestionFile,
  type VideoInfo,
  type VideoSummary,
} from "./labelFile.ts";
import {
  KIND_NAMES,
  KIND_SYMBOLS,
  countText,
  formatClock,
  formatTime,
  highlightRally,
  shortTitle,
  highlightSuggestions,
  renderMarkList,
  renderSuggestionList,
  suggestionsInPart,
  type PartSuggestions,
} from "./markList.ts";
import {
  History,
  deleteMark,
  isContact,
  layOutMarks,
  mergeUnsavedMarks,
  placeMark,
  rallyIndexAtFrame,
  sortMarks,
  toggleUnsure,
  type Mark,
  type MarkKind,
  type MarkLayout,
} from "./marks.ts";
import { detailHit, drawDetail, drawOverview, overviewFrameAt, type TimelineView } from "./timeline.ts";

const DEFAULT_RALLY_GAP_SECONDS = 4;
const MIN_SPEED = 0.0625;
const MAX_SPEED = 16;
const SETTINGS_KEY = "rally-labeller:settings";
// Land a little before a suggested rally so the serve is in view.
const SUGGESTION_LEAD_IN_SECONDS = 1;

interface Settings {
  jumpFrames: number;
  detailSeconds: number;
  pauseAfterMark: boolean;
  speed: number;
  showSuggestions: boolean;
  loupeZoom: number;
  loupeSizePx: number;
  loupeSharpen: boolean;
}

const DEFAULT_SETTINGS: Settings = {
  jumpFrames: 10,
  detailSeconds: 4,
  pauseAfterMark: false,
  speed: 1,
  showSuggestions: true,
  loupeZoom: LOUPE_DEFAULTS.zoom,
  loupeSizePx: LOUPE_DEFAULTS.sizePx,
  loupeSharpen: LOUPE_DEFAULTS.sharpen,
};

/** Everything undo can restore: what the label file stores apart from derived rallies. */
interface Labels {
  marks: Mark[];
  rallyGapSeconds: number;
}

interface Session {
  video: VideoInfo;
  /** Folder name of the open copy within its video. */
  clip: string;
  fps: number;
  player: FramePlayer;
  marks: Mark[];
  layout: MarkLayout;
  rallyGapSeconds: number;
  suggestions: SuggestionFile[];
  /** The suggested rallies and contacts inside this part, worked out once when it opens. */
  suggested: PartSuggestions;
  /** Frames of suggested contacts the labeller has dealt with; shared through the server. */
  doneFrames: Set<number>;
  /** Done toggles run one after another, so a quick second press undoes the first. */
  doneChain: Promise<void>;
  history: History<Labels>;
  /** Only meaningful while the playhead is on the selected mark's frame; see `selectedMark`. */
  selectedId: number | null;
  lastPlacedId: number | null;
  revision: number;
  /** Labels as last saved, so unsaved edits can be re-applied after another page saves. */
  saved: Labels;
  unsaved: boolean;
  saving: boolean;
  saveQueued: boolean;
}

class ServerError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function element<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) throw new Error(`page is missing #${id}`);
  return found as T;
}

const page = {
  picker: element<HTMLSelectElement>("video-picker"),
  readout: element("readout"),
  saveStatus: element("save-status"),
  helpButton: element<HTMLButtonElement>("help-button"),
  videoSlot: element("video-slot"),
  detail: element<HTMLCanvasElement>("detail"),
  overview: element<HTMLCanvasElement>("overview"),
  sectionWarning: element("section-warning"),
  speedPresets: element("speed-presets"),
  speed: element<HTMLInputElement>("speed"),
  jumpFrames: element<HTMLInputElement>("jump-frames"),
  rallyGap: element<HTMLInputElement>("rally-gap"),
  detailSeconds: element<HTMLInputElement>("detail-seconds"),
  pauseAfterMark: element<HTMLInputElement>("pause-after-mark"),
  loupeZoom: element<HTMLInputElement>("loupe-zoom"),
  loupeSize: element<HTMLInputElement>("loupe-size"),
  loupeSharpen: element<HTMLInputElement>("loupe-sharpen"),
  suggestionRow: element("suggestion-row"),
  showSuggestions: element<HTMLInputElement>("show-suggestions"),
  suggestionNames: element("suggestion-names"),
  markList: element("mark-list"),
  suggestionPanel: element("suggestion-panel"),
  suggestionCount: element("suggestion-count"),
  suggestionList: element("suggestion-list"),
  help: element("help"),
  helpKeys: element<HTMLTableElement>("help-keys"),
  helpClose: element<HTMLButtonElement>("help-close"),
  prepare: element("prepare"),
  prepareButton: element<HTMLButtonElement>("prepare-button"),
  prepareSource: element<HTMLSelectElement>("prepare-source"),
  prepareStart: element<HTMLInputElement>("prepare-start"),
  prepareEnd: element<HTMLInputElement>("prepare-end"),
  prepareGo: element<HTMLButtonElement>("prepare-go"),
  prepareClose: element<HTMLButtonElement>("prepare-close"),
  prepareLog: element("prepare-log"),
  listButton: element<HTMLButtonElement>("list-button"),
  moreButton: element<HTMLButtonElement>("more-button"),
  fullscreenButton: element<HTMLButtonElement>("fullscreen-button"),
  note: element("note"),
  sectionBadge: element("section-badge"),
};

interface PrepareJob {
  running: boolean;
  source: string;
  log: string[];
  exitCode: number | null;
  framesDone: number;
  framesWanted: number | null;
}

const PREPARE_POLL_MS = 1000;
const NOTE_MS = 1600;
// Touch gestures on the video.
// Android's own double-tap window.
const DOUBLE_TAP_MS = 300;
// A touch held longer than this is a resting thumb, not a tap.
const LONG_PRESS_MS = 400;
const TAP_MOVE_PX = 12;
const JOG_PX_PER_FRAME = 8;
// Double-taps within this fraction of the picture height either side of its
// middle are ambiguous: the far player often stands right on the middle line.
const MIDDLE_BAND = 0.08;
// A touch on a button is followed by a click; ignore that click.
const TOUCH_CLICK_MS = 800;

// Settings are per device, so they live in the browser rather than the label file.
function loadSettings(): Settings {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function storeSettings(): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Storage can be blocked (private windows); settings then last until reload.
  }
}

function positionKey(video: VideoInfo): string {
  return `rally-labeller:frame:${video.video_id}:${video.first_frame}`;
}

function pickerText(summary: VideoSummary): string {
  const fps = fpsValue(summary.fps);
  const range = `${formatClock(summary.first_frame / fps)}–${formatClock((summary.first_frame + summary.frame_count) / fps)}`;
  const counts = `${countText(summary.rally_count, "rally", "rallies")} · ${countText(summary.mark_count, "mark", "marks")}`;
  return `${shortTitle(summary.title)} · ${range} · ${counts}`;
}

/** Picker value and URL hash for one prepared copy. */
function clipKey(summary: Pick<VideoSummary, "video_id" | "clip">): string {
  return `${summary.video_id}/${summary.clip}`;
}

const settings = loadSettings();
let session: Session | null = null;
/** Above zero while openClip is loading a part; label changes wait for it. */
let partSwitchesUnderway = 0;

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const body = await response.json();
  if (!response.ok) throw new ServerError(response.status, body.error ?? response.statusText);
  return body as T;
}

// ---------------------------------------------------------------------------
// Drawing and readouts
// ---------------------------------------------------------------------------

let drawPending = false;

function timelineView(current: Session): TimelineView {
  return {
    suggestionsDone: current.doneFrames,
    firstFrame: current.video.first_frame,
    frameCount: current.video.frame_count,
    fps: current.fps,
    shownFrame: current.player.currentFrame,
    marks: current.marks,
    layout: current.layout,
    selectedId: current.selectedId,
    suggestions: settings.showSuggestions ? current.suggestions : null,
  };
}

function scheduleDraw(): void {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => {
    drawPending = false;
    if (session === null) return;
    const view = timelineView(session);
    drawDetail(page.detail, view, settings.detailSeconds);
    drawOverview(page.overview, view);
  });
}

function updateReadout(): void {
  if (session === null) return;
  const frame = session.player.currentFrame;
  const rallyIndex = rallyIndexAtFrame(session.layout.rallies, frame);
  const parts = [
    `Frame ${frame}`,
    formatTime(frame / session.fps),
    `${session.player.speed}×`,
    rallyIndex >= 0 ? `Rally ${rallyIndex + 1}` : "Between rallies",
  ];
  const selected = session.marks.find((mark) => mark.id === session?.selectedId);
  if (selected !== undefined) parts.push(`Selected ${KIND_SYMBOLS[selected.kind]} ${KIND_NAMES[selected.kind]} at ${selected.frame}`);
  page.readout.textContent = parts.join(" · ");
  // Any error shown here has just been replaced, so its red goes too.
  page.readout.classList.remove("failed");
  highlightRally(page.markList, rallyIndex);
  if (!page.suggestionPanel.hidden) highlightSuggestions(page.suggestionList, frame);
}

let noteTimer = 0;

/** Say briefly, over the video, what the last action did. */
function showNote(text: string): void {
  page.note.textContent = text;
  page.note.hidden = false;
  clearTimeout(noteTimer);
  noteTimer = window.setTimeout(() => (page.note.hidden = true), NOTE_MS);
}

function describeMark(mark: Mark): string {
  return `${KIND_SYMBOLS[mark.kind]} ${KIND_NAMES[mark.kind]} · ${mark.frame}`;
}

function showSaveStatus(owner: Session, text: string, failed = false): void {
  if (owner !== session) return;
  page.saveStatus.textContent = text;
  page.saveStatus.classList.toggle("failed", failed);
  page.saveStatus.title = failed ? `${text} Press to try again.` : "";
}

function onFrameShown(owner: Session, frame: number): void {
  // A replaced video can still report a frame; it must not touch the open session or its saved position.
  if (owner !== session) return;
  if (session.selectedId !== null && selectedMark(session) === null) {
    // The playhead has left the selected mark, so the selection no longer applies.
    session.selectedId = null;
    refreshMarks();
  }
  updateReadout();
  scheduleDraw();
  redrawLoupe();
  if (!session.player.playing) {
    try {
      localStorage.setItem(positionKey(session.video), String(frame));
    } catch {
      // Remembering the position is only a convenience.
    }
  }
}

/** True when `frame` is one of the open part's frames; one label file covers every part of a video. */
function insidePart(video: VideoInfo, frame: number): boolean {
  return frame >= video.first_frame && frame < video.first_frame + video.frame_count;
}

/** On a phone the More drawer covers the picture, so close it once a go-to button has moved the playhead. */
function closeMoreDrawer(): void {
  document.body.classList.remove("more-open");
}

/** Keep the open part's picker row counting the marks as they change, for this part only. */
function updatePickerRow(current: Session): void {
  const key = clipKey({ video_id: current.video.video_id, clip: current.clip });
  const option = [...page.picker.options].find((candidate) => candidate.value === key);
  if (option === undefined) return;
  const { first_frame: firstFrame, frame_count: frameCount } = current.video;
  option.text = pickerText({
    video_id: current.video.video_id,
    title: videoTitle(current.video),
    clip: current.clip,
    fps: current.video.fps,
    first_frame: firstFrame,
    frame_count: frameCount,
    mark_count: current.marks.filter((mark) => insidePart(current.video, mark.frame)).length,
    rally_count: current.layout.rallies.filter((rally) => insidePart(current.video, rally.startFrame)).length,
  });
}

/** Recompute rallies and redraw everything that depends on the marks. */
function refreshMarks(): void {
  if (session === null) return;
  const current = session;
  current.layout = layOutMarks(current.marks, current.fps, current.rallyGapSeconds);
  if (!current.marks.some((mark) => mark.id === current.selectedId)) current.selectedId = null;
  page.rallyGap.value = String(current.rallyGapSeconds);
  renderMarkList(page.markList, current.marks, current.layout, current.fps, current.selectedId, {
    select: (markId) => {
      document.body.classList.remove("list-open");
      selectMark(markId);
    },
    remove: (markId) => deleteWithNote(current, markId),
  });
  updatePickerRow(current);
  page.sectionWarning.hidden = current.marks.length === 0 || current.layout.sections.length > 0;
  page.sectionBadge.hidden = page.sectionWarning.hidden;
  updateReadout();
  scheduleDraw();
}

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

interface SavedClip {
  video: VideoInfo;
  labels: LabelFile | null;
  suggestions: SuggestionFile[];
  /** Frames of suggested contacts marked done. */
  suggestions_done: number[];
}

function fetchClip(videoId: string, clip: string): Promise<SavedClip> {
  return requestJson<SavedClip>(`/api/clips/${encodeURIComponent(videoId)}/${encodeURIComponent(clip)}`);
}

/**
 * Another page saved this video since this page last saved. Put this page's
 * unsaved edits on top of that copy, so neither page's work is lost.
 */
async function mergeWithLatest(owner: Session): Promise<void> {
  const { labels } = await fetchClip(owner.video.video_id, owner.clip);
  if (labels === null) throw new Error("the label file disappeared from the server");
  const gapChangedHere = owner.rallyGapSeconds !== owner.saved.rallyGapSeconds;
  owner.marks = mergeUnsavedMarks(labels.marks, owner.saved.marks, owner.marks);
  owner.rallyGapSeconds = gapChangedHere ? owner.rallyGapSeconds : labels.rally_gap_seconds;
  owner.saved = { marks: labels.marks, rallyGapSeconds: labels.rally_gap_seconds };
  owner.revision = labels.revision;
  // Undo steps from before the merge would quietly drop the other page's marks.
  owner.history = new History<Labels>();
  if (owner === session) {
    refreshMarks();
    showNote("Merged with labels saved from another page. Undo history cleared.");
  }
}

async function save(owner: Session): Promise<void> {
  if (owner.saving) {
    owner.saveQueued = true;
    return;
  }
  owner.saving = true;
  try {
    do {
      owner.saveQueued = false;
      showSaveStatus(owner, "Saving…");
      const sent: Labels = { marks: owner.marks, rallyGapSeconds: owner.rallyGapSeconds };
      const request: SaveRequest = {
        base_revision: owner.revision,
        rally_gap_seconds: sent.rallyGapSeconds,
        marks: sent.marks,
      };
      try {
        const result = await requestJson<{ revision: number }>(`/api/labels/${encodeURIComponent(owner.video.video_id)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request),
        });
        owner.revision = result.revision;
        owner.saved = sent;
      } catch (error) {
        if (!(error instanceof ServerError && error.status === 409)) throw error;
        await mergeWithLatest(owner);
        owner.saveQueued = true;
      }
    } while (owner.saveQueued);
    owner.unsaved = false;
    showSaveStatus(owner, "Saved");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    showSaveStatus(owner, `Not saved: ${message}.`, true);
  } finally {
    owner.saving = false;
  }
}

/** False, with a note, while another part is loading. */
function labelsCanChange(): boolean {
  if (partSwitchesUnderway === 0) return true;
  showNote("Loading the next part, so that change wasn't made.");
  return false;
}

/** Record an undo step, apply new labels and save them. */
function changeLabels(owner: Session, labels: Labels, description: string): void {
  if (!labelsCanChange()) return;
  owner.history.record({ marks: owner.marks, rallyGapSeconds: owner.rallyGapSeconds }, description);
  owner.marks = labels.marks;
  owner.rallyGapSeconds = labels.rallyGapSeconds;
  owner.unsaved = true;
  refreshMarks();
  void save(owner);
}

function changeMarks(owner: Session, marks: Mark[], description: string): void {
  changeLabels(owner, { marks, rallyGapSeconds: owner.rallyGapSeconds }, description);
}

/**
 * Pull in labels saved from another page while this one sat in the
 * background, so the next save doesn't have to merge.
 */
async function refreshFromServer(owner: Session): Promise<void> {
  const { labels, suggestions_done: doneFrames } = await fetchClip(owner.video.video_id, owner.clip);
  // Done ticks are review progress, not labels, so they refresh even while labels can't.
  owner.doneFrames = new Set(doneFrames);
  if (owner === session) {
    refreshSuggestionList();
    scheduleDraw();
  }
  if (labels === null || labels.revision === owner.revision || owner.unsaved || owner.saving) return;
  owner.marks = labels.marks;
  owner.rallyGapSeconds = labels.rally_gap_seconds;
  owner.saved = { marks: labels.marks, rallyGapSeconds: labels.rally_gap_seconds };
  owner.revision = labels.revision;
  owner.history = new History<Labels>();
  if (owner !== session) return;
  refreshMarks();
  showSaveStatus(owner, "Saved");
  showNote("Loaded newer labels saved from another page.");
}

// ---------------------------------------------------------------------------
// Actions, shared by keys and buttons
// ---------------------------------------------------------------------------

const PLACE_NOTES = {
  added: "",
  switched_side: " (side switched)",
  already_there: " is already here",
} as const;

/** Place a mark on the current frame, or on `frame` when given. */
function markAt(kind: MarkKind, frame?: number): void {
  if (session === null) return;
  frame ??= session.player.currentFrame;
  const placed = placeMark(session.marks, kind, frame);
  showNote(`${describeMark(placed.mark)}${PLACE_NOTES[placed.outcome]}`);
  if (placed.outcome === "already_there") return;
  session.lastPlacedId = placed.mark.id;
  const verb = placed.outcome === "added" ? "added" : "switched side of";
  changeMarks(session, placed.marks, `${verb} ${describeMark(placed.mark)}`);
  if (settings.pauseAfterMark) session.player.pause();
}

/** The selected mark, as long as the playhead is still on its frame. */
function selectedMark(current: Session): Mark | null {
  const mark = current.marks.find((candidate) => candidate.id === current.selectedId);
  return mark !== undefined && mark.frame === current.player.currentFrame ? mark : null;
}

/** The mark an edit key acts on: the selected mark, else a mark on the current frame, contacts first. */
function markHere(current: Session): Mark | null {
  const onFrame = current.marks.filter((mark) => mark.frame === current.player.currentFrame);
  return selectedMark(current) ?? onFrame.find((mark) => isContact(mark.kind)) ?? onFrame[0] ?? null;
}

function deleteWithNote(current: Session, markId: number): void {
  const mark = current.marks.find((candidate) => candidate.id === markId);
  if (mark === undefined) return;
  showNote(`Deleted ${describeMark(mark)}`);
  changeMarks(current, deleteMark(current.marks, markId), `deleted ${describeMark(mark)}`);
}

function selectMark(markId: number): void {
  if (session === null) return;
  const mark = session.marks.find((candidate) => candidate.id === markId);
  if (mark === undefined) return;
  if (!insidePart(session.video, mark.frame)) {
    showNote(`That mark (frame ${mark.frame}) is in another part of this video. Open that part to go to it.`);
    return;
  }
  session.selectedId = markId;
  session.player.pause();
  session.player.seekToFrame(mark.frame);
  refreshMarks();
}

function selectNeighbour(direction: -1 | 1): void {
  if (session === null) return;
  const current = session;
  const sorted = sortMarks(current.marks).filter((mark) => insidePart(current.video, mark.frame));
  if (sorted.length === 0) return;
  closeMoreDrawer();
  const selected = selectedMark(session);
  const selectedIndex = selected === null ? -1 : sorted.findIndex((mark) => mark.id === selected.id);
  let index: number;
  if (selectedIndex >= 0) {
    index = Math.min(Math.max(selectedIndex + direction, 0), sorted.length - 1);
  } else {
    const frame = session.player.currentFrame;
    index = direction < 0 ? sorted.findLastIndex((mark) => mark.frame <= frame) : sorted.findIndex((mark) => mark.frame >= frame);
    if (index < 0) index = direction < 0 ? 0 : sorted.length - 1;
  }
  selectMark(sorted[index].id);
}

function clearSelection(): void {
  page.help.hidden = true;
  page.prepare.hidden = true;
  turnMagnifierOff();
  if (session === null || session.selectedId === null) return;
  session.selectedId = null;
  refreshMarks();
}

function setSpeed(rate: number): void {
  const clamped = Math.min(Math.max(rate, MIN_SPEED), MAX_SPEED);
  settings.speed = clamped;
  storeSettings();
  page.speed.value = String(clamped);
  for (const button of page.speedPresets.querySelectorAll<HTMLButtonElement>("button")) {
    button.classList.toggle("active", Number(button.dataset.speed) === clamped);
  }
  if (session !== null) session.player.speed = clamped;
  updateReadout();
}

function changeSpeed(direction: -1 | 1): void {
  setSpeed(direction > 0 ? settings.speed * 2 : settings.speed / 2);
}

function undoOrRedo(direction: "undo" | "redo"): void {
  if (session === null || !labelsCanChange()) return;
  const current: Labels = { marks: session.marks, rallyGapSeconds: session.rallyGapSeconds };
  const entry = direction === "undo" ? session.history.undo(current) : session.history.redo(current);
  if (entry === null) {
    showNote(direction === "undo" ? "Nothing to undo" : "Nothing to redo");
    return;
  }
  showNote(`${direction === "undo" ? "Undid" : "Redid"}: ${entry.description}`);
  session.marks = entry.state.marks;
  session.rallyGapSeconds = entry.state.rallyGapSeconds;
  session.unsaved = true;
  refreshMarks();
  void save(session);
}

/** Go to the nearest suggested rally start before or after the current frame. */
function goToSuggestedRally(direction: -1 | 1): void {
  if (session === null || !settings.showSuggestions) return;
  const leadIn = Math.round(SUGGESTION_LEAD_IN_SECONDS * session.fps);
  const firstFrame = session.video.first_frame;
  const targets = session.suggested.rallies.map((rally) => Math.max(rally.startFrame - leadIn, firstFrame));
  const frame = session.player.currentFrame;
  const target = direction > 0 ? targets.find((start) => start > frame) : targets.findLast((start) => start < frame);
  if (target === undefined) {
    showNote(direction > 0 ? "No suggested rally after this frame" : "No suggested rally before this frame");
    return;
  }
  closeMoreDrawer();
  session.player.pause();
  session.player.seekToFrame(target);
}

/** Go to the nearest suggested contact before or after the current frame. */
function goToSuggestedContact(direction: -1 | 1): void {
  if (session === null || !settings.showSuggestions) return;
  const frames = session.suggested.contacts.map((contact) => contact.frame);
  const frame = session.player.currentFrame;
  const target = direction > 0 ? frames.find((candidate) => candidate > frame) : frames.findLast((candidate) => candidate < frame);
  if (target === undefined) {
    showNote(direction > 0 ? "No suggested contact after this frame" : "No suggested contact before this frame");
    return;
  }
  closeMoreDrawer();
  session.player.pause();
  session.player.seekToFrame(target);
}

/** Mark a suggested contact done, or not done again, after any toggle still on its way. */
function toggleSuggestionDone(owner: Session, frame: number): void {
  // A note rather than showError: the double-click's seek rewrites the readout straight away.
  const reportFailure = (error: unknown) =>
    showNote(`Done mark not saved: ${error instanceof Error ? error.message : String(error)}`);
  owner.doneChain = owner.doneChain.then(() => sendSuggestionDone(owner, frame)).catch(reportFailure);
}

async function sendSuggestionDone(owner: Session, frame: number): Promise<void> {
  const done = !owner.doneFrames.has(frame);
  const frames = await requestJson<number[]>(`/api/suggestions-done/${encodeURIComponent(owner.video.video_id)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frame, done }),
  });
  owner.doneFrames = new Set(frames);
  if (owner !== session) return;
  showNote(done ? `Suggested contact at ${frame} marked done` : `Suggested contact at ${frame} no longer done`);
  refreshSuggestionList();
  scheduleDraw();
}

/** D: mark the suggested contact on the current frame done, or not done again. */
function toggleSuggestionDoneHere(): void {
  if (session === null || !settings.showSuggestions) return;
  const frame = session.player.currentFrame;
  if (!session.suggested.contacts.some((contact) => contact.frame === frame)) {
    showNote("No suggested contact on this frame. Press A or S to go to one.");
    return;
  }
  toggleSuggestionDone(session, frame);
}

/** Show or hide the suggested-contact list, and rebuild it for the open part. */
function refreshSuggestionList(): void {
  if (session === null) return;
  const current = session;
  page.suggestionPanel.hidden = !settings.showSuggestions || current.suggested.contacts.length === 0;
  if (page.suggestionPanel.hidden) return;
  const doneCount = current.suggested.contacts.filter((contact) => current.doneFrames.has(contact.frame)).length;
  page.suggestionCount.textContent = `${doneCount} of ${current.suggested.contacts.length} done`;
  renderSuggestionList(page.suggestionList, current.suggested, current.fps, current.doneFrames, {
    goTo: (frame) => {
      document.body.classList.remove("list-open");
      current.player.pause();
      current.player.seekToFrame(frame);
    },
    toggleDone: (frame) => toggleSuggestionDone(current, frame),
  });
  highlightSuggestions(page.suggestionList, current.player.currentFrame);
}

const actions: Record<ActionName, () => void> = {
  playPause: () => {
    if (session === null) return;
    if (session.player.playing) {
      session.player.pause();
      return;
    }
    session.selectedId = null;
    refreshMarks();
    session.player.play();
  },
  markTop: () => markAt("top"),
  markBottom: () => markAt("bottom"),
  markBreak: () => markAt("break"),
  markStart: () => markAt("start"),
  markEnd: () => markAt("end"),
  markSectionStart: () => markAt("section_start"),
  markSectionEnd: () => markAt("section_end"),
  stepBack: () => session?.player.stepFrames(-1),
  stepForward: () => session?.player.stepFrames(1),
  jumpBack: () => session?.player.stepFrames(-settings.jumpFrames),
  jumpForward: () => session?.player.stepFrames(settings.jumpFrames),
  faster: () => changeSpeed(1),
  slower: () => changeSpeed(-1),
  selectPrevious: () => selectNeighbour(-1),
  selectNext: () => selectNeighbour(1),
  deleteSelected: () => {
    if (session === null) return;
    const target = markHere(session);
    if (target === null) {
      showNote("No mark on this frame to delete. Select one with R / T or the list.");
      return;
    }
    deleteWithNote(session, target.id);
  },
  toggleUnsure: () => {
    if (session === null) return;
    const current = session;
    // Right after marking during playback the playhead has moved on, so fall back to the last mark placed.
    const target = markHere(current) ?? current.marks.find((mark) => mark.id === current.lastPlacedId) ?? null;
    if (target === null) {
      showNote("No mark on this frame to flag.");
      return;
    }
    const nowUnsure = !target.unsure;
    showNote(`${nowUnsure ? "Unsure" : "No longer unsure"}: ${describeMark(target)}`);
    changeMarks(current, toggleUnsure(current.marks, target.id), `${nowUnsure ? "flagged" : "unflagged"} ${describeMark(target)}`);
  },
  undo: () => undoOrRedo("undo"),
  redo: () => undoOrRedo("redo"),
  clearSelection,
  previousSuggestedContact: () => goToSuggestedContact(-1),
  nextSuggestedContact: () => goToSuggestedContact(1),
  previousSuggestedRally: () => goToSuggestedRally(-1),
  nextSuggestedRally: () => goToSuggestedRally(1),
  toggleSuggestionDone: toggleSuggestionDoneHere,
};

// ---------------------------------------------------------------------------
// Opening a video
// ---------------------------------------------------------------------------

function savedPosition(video: VideoInfo): number {
  try {
    return Number(localStorage.getItem(positionKey(video)) ?? video.first_frame);
  } catch {
    return video.first_frame;
  }
}

function setButtonLabel(action: ActionName, text: string): void {
  for (const label of document.querySelectorAll(`button[data-action="${action}"] .label`)) label.textContent = text;
}

/** A first tap on the video, waiting to see whether a second tap follows. */
interface PendingTap {
  frame: number;
  clientX: number;
  clientY: number;
  timer: number;
  /** The tap paused playback, so a double-tap mark carries on playing. */
  resumeAfterMark: boolean;
}

let pendingTap: PendingTap | null = null;

/** Any other input means the first tap was not the start of a double-tap. */
function cancelPendingTap(): void {
  if (pendingTap === null) return;
  clearTimeout(pendingTap.timer);
  pendingTap = null;
}

/** Top or bottom half of the picture itself, ignoring letterbox bars; null near the middle. */
function sideAt(video: HTMLVideoElement, clientX: number, clientY: number): "top" | "bottom" | "middle" | "outside" {
  const box = video.getBoundingClientRect();
  const scale = Math.min(box.width / video.videoWidth, box.height / video.videoHeight);
  const pictureWidth = video.videoWidth * scale;
  const pictureHeight = video.videoHeight * scale;
  const pictureLeft = box.left + (box.width - pictureWidth) / 2;
  const pictureTop = box.top + (box.height - pictureHeight) / 2;
  const across = (clientX - pictureLeft) / pictureWidth;
  const down = (clientY - pictureTop) / pictureHeight;
  // The black bars around the picture are where a resting thumb lands; they never mark.
  const onPicture = across >= 0 && across <= 1 && down >= 0 && down <= 1;
  if (!onPicture) return "outside";
  const fromMiddle = down - 0.5;
  if (Math.abs(fromMiddle) < MIDDLE_BAND) return "middle";
  return fromMiddle < 0 ? "top" : "bottom";
}

function markFromDoubleTap(video: HTMLVideoElement, firstTap: PendingTap): void {
  if (session === null) return;
  // The first tap is the deliberate one; the second often drifts.
  const place = sideAt(video, firstTap.clientX, firstTap.clientY);
  if (place === "outside") {
    showNote("Double-tap on the picture to mark a contact.");
    return;
  }
  if (place === "middle") {
    showNote("Double-tap clearly above or below the middle of the picture.");
    return;
  }
  markAt(place, firstTap.frame);
  if (firstTap.resumeAfterMark && !settings.pauseAfterMark) session.player.play();
}

/**
 * Touch gestures on the video: tap to play or pause, double-tap the upper or
 * lower half for a top or bottom contact, drag sideways to step through frames.
 *
 * - A tap during playback pauses at once, on the frame touched.
 * - A double-tap marks the frame shown at its first tap.
 * - Only one finger counts. A second finger, a long press or a mostly
 *   vertical drag does nothing, so a thumb resting on the picture is harmless.
 */
/** Right-click turns the desktop magnifier on and off; while on, it follows the pointer over the video. */
let magnifierOn = false;

function turnMagnifierOff(): void {
  magnifierOn = false;
  hideLoupe();
}

function attachVideoGestures(video: HTMLVideoElement): void {
  interface Press {
    pointerId: number;
    startX: number;
    startY: number;
    startedAt: number;
    frame: number;
    wasPlaying: boolean;
    /** Where the finger was when the drag started stepping frames, or null before that. */
    jogFromX: number | null;
    ignored: boolean;
    /** Latest finger position, for a magnifier that starts while the finger is still. */
    lastX: number;
    lastY: number;
    loupeTimer: number;
    magnifying: boolean;
  }
  let press: Press | null = null;
  let lastPointerType = "";

  // Right-click toggles the magnifier, so the browser menu stays away.
  video.addEventListener("contextmenu", (event) => event.preventDefault());
  video.addEventListener(
    "wheel",
    (event) => {
      if (!loupeShown()) return;
      event.preventDefault();
      settings.loupeZoom = changeLoupeZoom(event.deltaY < 0 ? 1 : -1);
      page.loupeZoom.value = String(Math.round(settings.loupeZoom * 10) / 10);
      storeSettings();
    },
    { passive: false },
  );
  video.addEventListener("pointerdown", (event) => {
    lastPointerType = event.pointerType;
    if (event.pointerType === "mouse" && event.button === 2) {
      magnifierOn = !magnifierOn;
      if (magnifierOn) showLoupe(video, event.clientX, event.clientY);
      else hideLoupe();
      return;
    }
    if (event.pointerType !== "touch" || session === null) return;
    if (press !== null) {
      press.ignored = true;
      return;
    }
    video.setPointerCapture(event.pointerId);
    const wasPlaying = session.player.playing;
    const started: Press = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startedAt: event.timeStamp,
      frame: session.player.currentFrame,
      wasPlaying,
      jogFromX: null,
      ignored: false,
      lastX: event.clientX,
      lastY: event.clientY,
      loupeTimer: 0,
      magnifying: false,
    };
    // A finger held still becomes the magnifier rather than a tap or a frame drag.
    started.loupeTimer = window.setTimeout(() => {
      if (press !== started || started.ignored || started.jogFromX !== null) return;
      started.magnifying = true;
      cancelPendingTap();
      showLoupe(video, started.lastX, started.lastY);
    }, LONG_PRESS_MS);
    press = started;
    if (wasPlaying) session.player.pause();
  });
  video.addEventListener("pointermove", (event) => {
    if (event.pointerType === "mouse") {
      if (!magnifierOn) return;
      if (loupeShown()) moveLoupe(event.clientX, event.clientY);
      else showLoupe(video, event.clientX, event.clientY);
      return;
    }
    if (press === null || event.pointerId !== press.pointerId || press.ignored || session === null) return;
    press.lastX = event.clientX;
    press.lastY = event.clientY;
    if (press.magnifying) {
      moveLoupe(event.clientX, event.clientY);
      return;
    }
    if (press.jogFromX === null) {
      const movedX = Math.abs(event.clientX - press.startX);
      const movedY = Math.abs(event.clientY - press.startY);
      if (movedX < TAP_MOVE_PX && movedY < TAP_MOVE_PX) return;
      if (movedY > movedX) {
        press.ignored = true;
        return;
      }
      // Count frames from here, so the first frame step needs a full step of movement.
      press.jogFromX = event.clientX;
      cancelPendingTap();
      session.player.pause();
    }
    session.player.seekToFrame(press.frame + Math.round((event.clientX - press.jogFromX) / JOG_PX_PER_FRAME));
  });
  video.addEventListener("pointerup", (event) => {
    if (event.pointerType === "mouse") return;
    if (press === null || event.pointerId !== press.pointerId) return;
    const released = press;
    press = null;
    clearTimeout(released.loupeTimer);
    if (released.magnifying) {
      hideLoupe();
      return;
    }
    const isTap = !released.ignored && released.jogFromX === null && event.timeStamp - released.startedAt < LONG_PRESS_MS;
    if (!isTap) return;
    if (pendingTap !== null) {
      const firstTap = pendingTap;
      cancelPendingTap();
      markFromDoubleTap(video, firstTap);
      return;
    }
    const timer = window.setTimeout(() => {
      pendingTap = null;
      // A tap during playback already paused; a tap while paused plays.
      if (!released.wasPlaying) actions.playPause();
    }, DOUBLE_TAP_MS);
    pendingTap = {
      frame: released.frame,
      clientX: released.startX,
      clientY: released.startY,
      timer,
      resumeAfterMark: released.wasPlaying,
    };
  });
  // The box hides off the video but the magnifier stays on, so it returns with the pointer.
  video.addEventListener("pointerleave", (event) => {
    if (event.pointerType === "mouse") hideLoupe();
  });
  video.addEventListener("pointercancel", (event) => {
    if (press?.pointerId !== event.pointerId) return;
    clearTimeout(press.loupeTimer);
    press = null;
    hideLoupe();
  });
  video.addEventListener("click", () => {
    if (lastPointerType !== "touch") actions.playPause();
  });
}

/**
 * Run a button's action. A touch acts when the finger lifts over the button,
 * because Chromium sends no click when two touches overlap, such as one
 * thumb still on +1 while the other taps Top. Mouse clicks act on click.
 */
function attachButtonAction(button: HTMLButtonElement, run: () => void): void {
  let touchedAt = -Infinity;
  button.addEventListener("pointerup", (event) => {
    if (event.pointerType !== "touch") return;
    const box = button.getBoundingClientRect();
    const inside = event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom;
    if (!inside) return;
    touchedAt = event.timeStamp;
    run();
  });
  button.addEventListener("click", (event) => {
    // The touch already acted on pointerup. Browsers that don't say what made the click fall back to timing.
    const pointerType = (event as PointerEvent).pointerType;
    const fromTouch = pointerType === undefined ? event.timeStamp - touchedAt < TOUCH_CLICK_MS : pointerType === "touch";
    if (!fromTouch) run();
  });
}

function runAction(action: ActionName): void {
  cancelPendingTap();
  actions[action]();
}

async function toggleFullscreen(): Promise<void> {
  if (document.fullscreenElement !== null) {
    await document.exitFullscreen();
    return;
  }
  await document.documentElement.requestFullscreen();
  // Phones allow locking the orientation only in full screen; desktops refuse, which is fine.
  const orientation = screen.orientation as ScreenOrientation & { lock?: (orientation: string) => Promise<void> };
  await orientation.lock?.("landscape").catch(() => undefined);
}

function updatePlayButton(): void {
  const playing = session?.player.playing ?? false;
  setButtonLabel("playPause", playing ? "Pause" : "Play");
}

/** Wait for the open part's saves; false when its marks are still unsaved, so it must stay open. */
async function settleBeforeLeaving(): Promise<boolean> {
  if (session === null || !(session.unsaved || session.saving)) return true;
  // The new part loads labels from the server, so unsaved marks here would be lost.
  const leaving = session;
  while (leaving.saving) await new Promise((resolve) => setTimeout(resolve, 50));
  if (leaving.unsaved) await save(leaving);
  if (!leaving.unsaved) return true;
  showNote("This part has unsaved marks, so it stays open. Press the save status to try again.");
  page.picker.value = clipKey({ video_id: leaving.video.video_id, clip: leaving.clip });
  return false;
}

async function openClip(key: string): Promise<void> {
  // The old picture stays on screen while the new part loads; a mark made then would
  // be replaced along with the old part, so label changes wait until the switch ends.
  partSwitchesUnderway += 1;
  try {
    if (!(await settleBeforeLeaving())) return;
    const [videoId, clip] = key.split("/");
    showPart(key, await fetchClip(videoId, clip));
  } finally {
    partSwitchesUnderway -= 1;
  }
}

function showPart(key: string, fetched: SavedClip): void {
  const [videoId, clip] = key.split("/");
  const { video: info, labels, suggestions, suggestions_done: doneFrames } = fetched;
  hideLoupe();
  session?.player.video.pause();

  const video = document.createElement("video");
  video.preload = "auto";
  video.playsInline = true;
  video.src = `/media/${encodeURIComponent(videoId)}/${encodeURIComponent(clip)}/video.mp4`;
  page.videoSlot.replaceChildren(video);
  attachVideoGestures(video);
  video.addEventListener("play", updatePlayButton);
  video.addEventListener("pause", updatePlayButton);

  const fps = fpsValue(info.fps);
  const marks = labels?.marks ?? [];
  const rallyGapSeconds = labels === null ? DEFAULT_RALLY_GAP_SECONDS : labels.rally_gap_seconds;
  const opened: Session = {
    video: info,
    clip,
    fps,
    player: new FramePlayer(video, fps, info.first_frame, info.frame_count, (frame) => onFrameShown(opened, frame)),
    marks,
    layout: layOutMarks(marks, fps, rallyGapSeconds),
    rallyGapSeconds,
    suggestions,
    suggested: suggestionsInPart(suggestions, info.first_frame, info.first_frame + info.frame_count),
    doneFrames: new Set(doneFrames),
    doneChain: Promise.resolve(),
    history: new History<Labels>(),
    selectedId: null,
    lastPlacedId: null,
    revision: labels?.revision ?? 0,
    saved: { marks, rallyGapSeconds },
    unsaved: false,
    saving: false,
    saveQueued: false,
  };
  session = opened;
  video.addEventListener(
    "loadedmetadata",
    () => {
      // Loading a new source resets the playback rate.
      opened.player.speed = settings.speed;
      opened.player.seekToFrame(savedPosition(info));
    },
    { once: true },
  );
  page.suggestionRow.hidden = suggestions.length === 0;
  page.suggestionNames.textContent = suggestions.map((file) => file.name).join(", ");
  refreshSuggestionList();
  showSaveStatus(opened, labels === null ? "No labels yet" : `Loaded ${marks.length} marks`);
  history.replaceState(null, "", `#${key}`);
  updatePlayButton();
  refreshMarks();
}

/** Rebuild the picker, keeping the open part selected. */
async function loadVideoList(): Promise<VideoSummary[]> {
  const videos = await requestJson<VideoSummary[]>("/api/videos");
  const placeholder = new Option(videos.length === 0 ? "No prepared videos" : "Choose a video…", "");
  placeholder.disabled = true;
  const options = videos.map((summary) => {
    const option = new Option(pickerText(summary), clipKey(summary));
    option.title = `${summary.title} (${summary.video_id})`;
    return option;
  });
  page.picker.replaceChildren(placeholder, ...options);
  page.picker.value = session === null ? "" : clipKey({ video_id: session.video.video_id, clip: session.clip });
  if (videos.length === 0) page.readout.textContent = "Prepare a video first: press Prepare at the top right.";
  return videos;
}

/** On page load, open the part named in the address, if it's still prepared. */
async function openFromAddress(): Promise<void> {
  const videos = await loadVideoList();
  const wanted = decodeURIComponent(location.hash.slice(1));
  if (!videos.some((summary) => clipKey(summary) === wanted)) return;
  page.picker.value = wanted;
  await openClip(wanted);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function setUpButtons(): void {
  // The settings toggle must not take focus either, or Space would open and close it.
  for (const summary of document.querySelectorAll("summary")) {
    summary.addEventListener("mousedown", (event) => event.preventDefault());
  }
  for (const rate of SPEED_PRESETS) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.speed = String(rate);
    button.textContent = `${rate}×`;
    attachButtonAction(button, () => setSpeed(rate));
    page.speedPresets.append(button);
  }
  for (const button of document.querySelectorAll<HTMLButtonElement>("button")) {
    // Buttons never take keyboard focus, so Space and the letter keys always reach the shortcuts.
    button.addEventListener("mousedown", (event) => {
      event.preventDefault();
      // Focus stays put, so leave a box the user typed in; shortcuts don't reach a focused box.
      if (isTyping(document.activeElement)) (document.activeElement as HTMLElement).blur();
    });
    const action = button.dataset.action as ActionName | undefined;
    if (action === undefined) continue;
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = button.textContent;
    button.replaceChildren(label);
    button.title = ACTION_DESCRIPTIONS[action];
    const keyLabel = keyLabelFor(action);
    if (keyLabel !== undefined) {
      const hint = document.createElement("kbd");
      hint.textContent = keyLabel;
      button.append(hint);
    }
    attachButtonAction(button, () => runAction(action));
  }
}

function updateJumpButtons(): void {
  setButtonLabel("jumpBack", `−${settings.jumpFrames}`);
  setButtonLabel("jumpForward", `+${settings.jumpFrames}`);
}

function setUpSettings(): void {
  page.jumpFrames.value = String(settings.jumpFrames);
  page.detailSeconds.value = String(settings.detailSeconds);
  page.pauseAfterMark.checked = settings.pauseAfterMark;
  page.showSuggestions.checked = settings.showSuggestions;
  page.showSuggestions.addEventListener("change", () => {
    settings.showSuggestions = page.showSuggestions.checked;
    storeSettings();
    page.showSuggestions.blur();
    refreshSuggestionList();
    scheduleDraw();
  });
  page.jumpFrames.addEventListener("change", () => {
    settings.jumpFrames = Math.max(2, Math.round(Number(page.jumpFrames.value)) || DEFAULT_SETTINGS.jumpFrames);
    page.jumpFrames.value = String(settings.jumpFrames);
    storeSettings();
    updateJumpButtons();
  });
  /** Read the magnifier settings, keep them within limits, and show the values in use. */
  const applyLoupeSettings = () => {
    const inUse = setLoupeOptions({ zoom: settings.loupeZoom, sizePx: settings.loupeSizePx, sharpen: settings.loupeSharpen });
    settings.loupeZoom = inUse.zoom;
    settings.loupeSizePx = inUse.sizePx;
    page.loupeZoom.value = String(Math.round(inUse.zoom * 10) / 10);
    page.loupeSize.value = String(inUse.sizePx);
    page.loupeSharpen.checked = inUse.sharpen;
  };
  applyLoupeSettings();
  // The panel opens above the Settings button and grows leftwards; where the button
  // wraps to the left of its row, grow rightwards instead so the panel stays on screen.
  const settingsBox = document.querySelector<HTMLDetailsElement>("details.settings");
  const settingsPanel = document.querySelector<HTMLElement>(".settings-panel");
  if (settingsBox === null || settingsPanel === null) throw new Error("page is missing the settings panel");
  settingsBox.addEventListener("toggle", () => {
    if (!settingsBox.open) return;
    settingsPanel.style.left = "";
    settingsPanel.style.right = "";
    if (settingsPanel.getBoundingClientRect().left < 0) {
      settingsPanel.style.left = "0";
      settingsPanel.style.right = "auto";
    }
  });
  page.loupeZoom.addEventListener("change", () => {
    settings.loupeZoom = Number(page.loupeZoom.value) || LOUPE_DEFAULTS.zoom;
    applyLoupeSettings();
    storeSettings();
  });
  page.loupeSize.addEventListener("change", () => {
    settings.loupeSizePx = Number(page.loupeSize.value) || LOUPE_DEFAULTS.sizePx;
    applyLoupeSettings();
    storeSettings();
  });
  page.loupeSharpen.addEventListener("change", () => {
    settings.loupeSharpen = page.loupeSharpen.checked;
    page.loupeSharpen.blur();
    applyLoupeSettings();
    storeSettings();
  });
  page.detailSeconds.addEventListener("change", () => {
    settings.detailSeconds = Math.max(1, Number(page.detailSeconds.value) || DEFAULT_SETTINGS.detailSeconds);
    page.detailSeconds.value = String(settings.detailSeconds);
    storeSettings();
    scheduleDraw();
  });
  page.pauseAfterMark.addEventListener("change", () => {
    settings.pauseAfterMark = page.pauseAfterMark.checked;
    storeSettings();
    page.pauseAfterMark.blur();
  });
  page.rallyGap.addEventListener("change", () => {
    if (session === null) return;
    const previous = session.rallyGapSeconds;
    const gap = Number(page.rallyGap.value);
    if (page.rallyGap.value.trim() === "" || !(gap > 0)) {
      // A stray key must not change how rallies split, so keep the old value.
      page.rallyGap.value = String(previous);
      showNote(`The pause must be more than 0 seconds; kept ${previous} s.`);
      return;
    }
    if (gap === previous) return;
    showNote(`New rally after a ${gap} s pause (was ${previous} s)`);
    changeLabels(session, { marks: session.marks, rallyGapSeconds: gap }, `rally gap ${previous} s → ${gap} s`);
  });
  // Apply while typing, without rewriting the box mid-entry; tidy the text when done.
  page.speed.addEventListener("input", () => {
    const rate = Number(page.speed.value);
    if (rate >= MIN_SPEED && rate <= MAX_SPEED && session !== null) {
      settings.speed = rate;
      storeSettings();
      session.player.speed = rate;
      updateReadout();
    }
  });
  page.speed.addEventListener("change", () => setSpeed(Number(page.speed.value) || settings.speed));
  updateJumpButtons();
  setSpeed(settings.speed);
}

function setUpHelp(): void {
  const rows = KEY_BINDINGS.map((binding) => {
    const row = document.createElement("tr");
    const key = document.createElement("td");
    key.textContent = binding.keyLabel;
    const description = document.createElement("td");
    const holdNote = binding.repeatEveryMs === null ? "" : ". Hold to keep going";
    description.textContent = ACTION_DESCRIPTIONS[binding.action] + holdNote;
    row.append(key, description);
    return row;
  });
  const zoomRow = document.createElement("tr");
  const zoomHow = document.createElement("td");
  zoomHow.textContent = "Right-click";
  const zoomWhat = document.createElement("td");
  zoomWhat.textContent = "Turn the magnifier on or off; it follows the pointer over the video. Scroll to zoom; Esc also turns it off. On a phone, press and hold the picture.";
  zoomRow.append(zoomHow, zoomWhat);
  page.helpKeys.replaceChildren(...rows, zoomRow);
  page.helpButton.addEventListener("click", () => (page.help.hidden = !page.help.hidden));
  page.helpClose.addEventListener("click", () => (page.help.hidden = true));
  page.help.addEventListener("click", (event) => {
    if (event.target === page.help) page.help.hidden = true;
  });
}

function setUpTimelines(): void {
  const seekOverview = (event: PointerEvent) => {
    if (session === null) return;
    session.player.seekToFrame(overviewFrameAt(page.overview, event.clientX, timelineView(session)));
  };
  page.overview.addEventListener("pointerdown", (event) => {
    cancelPendingTap();
    page.overview.setPointerCapture(event.pointerId);
    seekOverview(event);
  });
  page.overview.addEventListener("pointermove", (event) => {
    if (event.buttons !== 0) seekOverview(event);
  });
  page.detail.addEventListener("click", (event) => {
    cancelPendingTap();
    if (session === null) return;
    const hit = detailHit(page.detail, event.clientX, timelineView(session), settings.detailSeconds);
    if (hit.markId !== null) selectMark(hit.markId);
    else session.player.seekToFrame(hit.frame);
  });
  window.addEventListener("resize", scheduleDraw);
}

/** Clips listed before a preparation started, so the new one can be opened when it finishes. */
let clipsBeforePrepare: Set<string> | null = null;
let sourceSummaries: SourceSummary[] = [];

// Same forms prepare_video.py accepts: 95, 1:35 or 1:01:35, with optional decimals.
const TIME_TEXT = /^\d+(\.\d+)?(:\d{1,2}(\.\d+)?){0,2}$/;
// prepare_video.py prints this when the part already exists.
const ALREADY_PREPARED_LINE = /^(\S+): frames (\d+)-(\d+) already prepared/;

function parseTimeText(text: string): number | null {
  if (!TIME_TEXT.test(text)) return null;
  return text.split(":").reduce((seconds, part) => seconds * 60 + Number(part), 0);
}

/** What's wrong with the times typed into the Prepare panel, in plain words, or null. */
function prepareProblem(): string | null {
  const startText = page.prepareStart.value.trim();
  const endText = page.prepareEnd.value.trim();
  const start = startText === "" ? 0 : parseTimeText(startText);
  const end = endText === "" ? null : parseTimeText(endText);
  if (start === null) return `Start "${startText}" isn't a time. Use 95, 1:35 or 1:01:35.`;
  if (endText !== "" && end === null) return `End "${endText}" isn't a time. Use 95, 1:35 or 1:01:35.`;
  const duration = sourceSummaries.find((source) => source.name === page.prepareSource.value)?.duration_seconds ?? null;
  if (duration !== null && start >= duration) return `This video is ${formatClock(duration)} long, so it can't start at ${formatClock(start)}.`;
  if (end !== null && end <= start) return `The end (${formatClock(end)}) must come after the start (${formatClock(start)}).`;
  if (end !== null && duration !== null && end > duration + 1) {
    return `This video is only ${formatClock(duration)} long. Leave End blank to run to the end.`;
  }
  return null;
}

function showPrepareMessage(text: string, failed: boolean): void {
  page.prepareLog.textContent = text;
  page.prepareLog.classList.toggle("failed", failed);
}

function prepareStatus(job: PrepareJob): { text: string; failed: boolean } {
  if (job.running) {
    if (job.framesWanted === null) return { text: `Preparing ${job.source}…`, failed: false };
    const percent = Math.floor((100 * job.framesDone) / job.framesWanted);
    return { text: `Preparing ${job.source}: ${job.framesDone} of ${job.framesWanted} frames (${percent}%)`, failed: false };
  }
  if (job.exitCode === null) return { text: "", failed: false };
  if (job.exitCode === 0) return { text: "Done.", failed: false };
  return { text: [`Couldn't prepare ${job.source}:`, ...job.log.slice(-4)].join("\n"), failed: true };
}

async function pollPrepare(): Promise<void> {
  const job = await requestJson<PrepareJob>("/api/prepare");
  const status = prepareStatus(job);
  showPrepareMessage(status.text, status.failed);
  page.prepareGo.disabled = job.running;
  if (job.running) {
    setTimeout(() => pollPrepare().catch(showError), PREPARE_POLL_MS);
    return;
  }
  if (clipsBeforePrepare === null || job.exitCode !== 0) return;
  const before = clipsBeforePrepare;
  clipsBeforePrepare = null;
  const videos = await requestJson<VideoSummary[]>("/api/videos");
  const alreadyPrepared = job.log.map((line) => ALREADY_PREPARED_LINE.exec(line)).find((match) => match !== null);
  const existing =
    alreadyPrepared === undefined
      ? undefined
      : `${alreadyPrepared[1]}/${alreadyPrepared[2].padStart(7, "0")}-${alreadyPrepared[3].padStart(7, "0")}`;
  const opened = videos.map(clipKey).find((key) => !before.has(key)) ?? existing;
  await loadVideoList();
  if (opened === undefined) return;
  page.picker.value = opened;
  page.prepare.hidden = true;
  await openClip(opened);
  if (existing !== undefined) showNote("That part was already prepared, so it opened as it was.");
}

function setUpPrepare(): void {
  page.prepareButton.addEventListener("click", () => {
    page.prepare.hidden = false;
    requestJson<SourceSummary[]>("/api/sources")
      .then((sources) => {
        sourceSummaries = sources;
        const options = sources.map((source) => {
          const length = source.duration_seconds === null ? "length unknown" : formatClock(source.duration_seconds);
          return new Option(`${shortTitle(source.title)} · ${length}`, source.name);
        });
        page.prepareSource.replaceChildren(...options);
      })
      .then(() => pollPrepare())
      .catch(showError);
  });
  page.prepareClose.addEventListener("click", () => (page.prepare.hidden = true));
  page.prepareGo.addEventListener("click", async () => {
    const problem = prepareProblem();
    if (problem !== null) {
      showPrepareMessage(problem, true);
      return;
    }
    try {
      const videos = await requestJson<VideoSummary[]>("/api/videos");
      clipsBeforePrepare = new Set(videos.map(clipKey));
      await requestJson<PrepareJob>("/api/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: page.prepareSource.value,
          start: page.prepareStart.value,
          end: page.prepareEnd.value,
        }),
      });
      await pollPrepare();
    } catch (error) {
      clipsBeforePrepare = null;
      showPrepareMessage(error instanceof Error ? error.message : String(error), true);
    }
  });
}

function isTyping(target: EventTarget | null): boolean {
  if (target instanceof HTMLTextAreaElement) return true;
  return target instanceof HTMLInputElement && target.type !== "checkbox";
}

/** Timer for the next repeat of a held key, or null when no key is repeating. */
let heldKeyTimer: number | null = null;
/** How often a held step checks whether the last frame has appeared yet. */
const SEEK_CHECK_MS = 10;

function stopRepeating(): void {
  if (heldKeyTimer !== null) clearTimeout(heldKeyTimer);
  heldKeyTimer = null;
}

/**
 * Repeat `action` while its key stays down. This timer sets the pace rather
 * than the system's key repeat, so it is the same on every computer.
 */
function repeatWhileHeld(action: ActionName, everyMs: number): void {
  const repeat = () => {
    // Wait for the last step's frame to appear, so the picture keeps up with the frame number.
    if (session?.player.video.seeking) {
      heldKeyTimer = window.setTimeout(repeat, SEEK_CHECK_MS);
      return;
    }
    runAction(action);
    heldKeyTimer = window.setTimeout(repeat, everyMs);
  };
  heldKeyTimer = window.setTimeout(repeat, REPEAT_DELAY_MS);
}

function setUpKeys(): void {
  // Releasing any key, Shift included, ends a repeat. A window that loses focus
  // never hears the key come up, so that ends it too.
  document.addEventListener("keyup", stopRepeating);
  window.addEventListener("blur", stopRepeating);
  window.addEventListener("blur", hideLoupe);
  document.addEventListener("keydown", (event) => {
    // A new key press ends any repeat, as the system's own key repeat does.
    if (!event.repeat) stopRepeating();
    if (!page.help.hidden || !page.prepare.hidden) {
      // A dialog is open: Esc closes it, and nothing reaches the video behind it.
      if (event.code === "Escape") clearSelection();
      return;
    }
    if (isTyping(event.target)) {
      // Tab leaves the box rather than walking into the next setting, where a stray key could change it.
      if (event.code === "Escape" || event.code === "Enter" || event.code === "Tab") {
        event.preventDefault();
        (event.target as HTMLElement).blur();
      }
      return;
    }
    const binding = bindingFor(event);
    if (binding === undefined) return;
    event.preventDefault();
    // A focused drop-down would otherwise swallow every key; letters must not pick another video either.
    if (event.target instanceof HTMLSelectElement) event.target.blur();
    // The system's repeats are ignored; repeatWhileHeld paces held keys instead.
    if (event.repeat) return;
    runAction(binding.action);
    if (binding.repeatEveryMs !== null) repeatWhileHeld(binding.action, binding.repeatEveryMs);
  });
}

function showError(error: unknown): void {
  page.readout.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
  page.readout.classList.add("failed");
}

setUpButtons();
setUpSettings();
setUpHelp();
setUpTimelines();
setUpKeys();
setUpPrepare();
page.listButton.addEventListener("click", () => {
  document.body.classList.remove("more-open");
  document.body.classList.toggle("list-open");
});
page.moreButton.addEventListener("click", () => {
  document.body.classList.remove("list-open");
  document.body.classList.toggle("more-open");
});
page.fullscreenButton.addEventListener("click", () => toggleFullscreen().catch(showError));
// Turning the phone upright hides the player behind the rotate prompt, so stop playback too.
matchMedia("(pointer: coarse) and (orientation: portrait)").addEventListener("change", (event) => {
  if (event.matches) session?.player.pause();
});
page.picker.addEventListener("change", () => {
  page.picker.blur();
  openClip(page.picker.value).catch(showError);
});
page.saveStatus.addEventListener("click", () => {
  if (session !== null && session.unsaved) void save(session);
});
// Another device may have saved this video while this page sat in the background.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && session !== null) refreshFromServer(session).catch(showError);
});
window.addEventListener("beforeunload", (event) => {
  if (session?.unsaved) event.preventDefault();
});
window.addEventListener("error", (event) => showError(event.error));
window.addEventListener("unhandledrejection", (event) => showError(event.reason));
openFromAddress().catch(showError);
