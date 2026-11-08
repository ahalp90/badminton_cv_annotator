/**
 * Marks placed by the labeller, and the rallies and labelled sections worked
 * out from them.
 *
 * Only marks are edited. Rallies and sections are recomputed from the marks
 * after every change, so a boundary can never go stale.
 */

export type MarkKind = "top" | "bottom" | "start" | "end" | "break" | "section_start" | "section_end";

export const MARK_KINDS: readonly MarkKind[] = [
  "top",
  "bottom",
  "start",
  "end",
  "break",
  "section_start",
  "section_end",
];

export interface Mark {
  id: number;
  kind: MarkKind;
  frame: number;
  unsure: boolean;
}

/** Why a rally finished where it did. */
export type RallyEnding = "end_mark" | "break_mark" | "next_start_mark" | "long_gap" | "last_mark";

export interface Rally {
  /** Inclusive first frame: the start mark, or the first contact when unmarked. */
  startFrame: number;
  startMarked: boolean;
  /** Inclusive last frame: the end mark, or the last contact when unmarked. */
  endFrame: number;
  endMarked: boolean;
  endedBy: RallyEnding;
  /** Top and bottom contacts in frame order. */
  contacts: Mark[];
  /** Every mark that belongs to this rally, including its start, end or break mark. */
  markIds: number[];
  /** Index of the labelled section that contains the whole rally, or null. */
  sectionIndex: number | null;
}

/**
 * A stretch of video that has been fully labelled. Frames outside every
 * section are unlabelled, not rally-free.
 */
export interface Section {
  startFrame: number;
  /** Inclusive last frame, or null while the section is still open. */
  endFrame: number | null;
}

export interface MarkLayout {
  rallies: Rally[];
  sections: Section[];
  rallyIndexByMarkId: Map<number, number>;
  /** Marks that change nothing: an end or break outside a rally, or a stray section mark. */
  noEffectMarkIds: Set<number>;
  /** Rally and contact marks outside every section. Empty when there are no sections yet. */
  outsideSectionMarkIds: Set<number>;
}

export function isContact(kind: MarkKind): boolean {
  return kind === "top" || kind === "bottom";
}

export function isSectionMark(kind: MarkKind): boolean {
  return kind === "section_start" || kind === "section_end";
}

// Order of marks that share a frame: a section opens first, then a rally
// starts, then contacts, then the rally closes, and a section closes last.
const SAME_FRAME_ORDER: Record<MarkKind, number> = {
  section_start: 0,
  start: 1,
  top: 2,
  bottom: 2,
  end: 3,
  break: 3,
  section_end: 4,
};

export function sortMarks(marks: readonly Mark[]): Mark[] {
  return [...marks].sort(
    (left, right) =>
      left.frame - right.frame ||
      SAME_FRAME_ORDER[left.kind] - SAME_FRAME_ORDER[right.kind] ||
      left.id - right.id,
  );
}

interface OpenRally {
  startMark: Mark | null;
  contacts: Mark[];
  markIds: number[];
}

function finishRally(open: OpenRally, endedBy: RallyEnding, closingMark: Mark | null): Rally {
  // An open rally always begins with a start mark or a contact, so one of these exists.
  const startFrame = open.startMark?.frame ?? open.contacts[0].frame;
  const lastContactFrame = open.contacts.at(-1)?.frame ?? startFrame;
  const endMarked = closingMark?.kind === "end";
  return {
    startFrame,
    startMarked: open.startMark !== null,
    endFrame: endMarked ? closingMark.frame : lastContactFrame,
    endMarked,
    endedBy,
    contacts: open.contacts,
    markIds: closingMark === null ? open.markIds : [...open.markIds, closingMark.id],
    sectionIndex: null,
  };
}

/**
 * Group rally and contact marks into rallies.
 *
 * - A start mark opens a rally, closing any rally already open.
 * - An end mark closes the open rally at its own frame.
 * - A break mark closes the open rally at its last contact.
 * - A contact with no open rally opens one.
 * - A contact more than `gapFrames` after the previous contact opens a new rally.
 */
function layOutRallies(sorted: readonly Mark[], gapFrames: number, noEffectMarkIds: Set<number>): Rally[] {
  const rallies: Rally[] = [];
  let open: OpenRally | null = null;
  for (const mark of sorted) {
    if (isSectionMark(mark.kind)) continue;
    if (mark.kind === "start") {
      if (open !== null) rallies.push(finishRally(open, "next_start_mark", null));
      open = { startMark: mark, contacts: [], markIds: [mark.id] };
      continue;
    }
    if (mark.kind === "end" || mark.kind === "break") {
      if (open === null) {
        noEffectMarkIds.add(mark.id);
        continue;
      }
      rallies.push(finishRally(open, mark.kind === "end" ? "end_mark" : "break_mark", mark));
      open = null;
      continue;
    }
    const previousContact = open?.contacts.at(-1);
    const pauseTooLong = previousContact !== undefined && mark.frame - previousContact.frame > gapFrames;
    if (open !== null && pauseTooLong) {
      rallies.push(finishRally(open, "long_gap", null));
      open = null;
    }
    open ??= { startMark: null, contacts: [], markIds: [] };
    open.contacts.push(mark);
    open.markIds.push(mark.id);
  }
  if (open !== null) rallies.push(finishRally(open, "last_mark", null));
  return rallies;
}

/** Pair section start and end marks. A second start while one is open, or an end with none open, does nothing. */
function layOutSections(sorted: readonly Mark[], noEffectMarkIds: Set<number>): Section[] {
  const sections: Section[] = [];
  let open: Section | null = null;
  for (const mark of sorted) {
    if (mark.kind === "section_start") {
      if (open !== null) {
        noEffectMarkIds.add(mark.id);
        continue;
      }
      open = { startFrame: mark.frame, endFrame: null };
      sections.push(open);
    } else if (mark.kind === "section_end") {
      if (open === null) {
        noEffectMarkIds.add(mark.id);
        continue;
      }
      open.endFrame = mark.frame;
      open = null;
    }
  }
  return sections;
}

export function sectionIndexAtFrame(sections: readonly Section[], frame: number): number {
  return sections.findIndex((section) => section.startFrame <= frame && frame <= (section.endFrame ?? Infinity));
}

/**
 * Work out rallies and labelled sections from marks in any order.
 *
 * @param marks Every mark for one video.
 * @param fps Video frame rate, used to convert the gap to frames.
 * @param gapSeconds Longest pause between contacts within one rally.
 */
export function layOutMarks(marks: readonly Mark[], fps: number, gapSeconds: number): MarkLayout {
  const sorted = sortMarks(marks);
  const noEffectMarkIds = new Set<number>();
  const gapFrames = gapSeconds * fps;
  const rallies = layOutRallies(sorted, gapFrames, noEffectMarkIds);
  const sections = layOutSections(sorted, noEffectMarkIds);

  const rallyIndexByMarkId = new Map<number, number>();
  rallies.forEach((rally, rallyIndex) => {
    const sectionIndex = sectionIndexAtFrame(sections, rally.startFrame);
    const section = sections[sectionIndex];
    const insideSection = section !== undefined && rally.endFrame <= (section.endFrame ?? Infinity);
    rally.sectionIndex = insideSection ? sectionIndex : null;
    for (const markId of rally.markIds) rallyIndexByMarkId.set(markId, rallyIndex);
  });

  const outsideSectionMarkIds = new Set<number>();
  if (sections.length > 0) {
    for (const mark of sorted) {
      const outside = !isSectionMark(mark.kind) && sectionIndexAtFrame(sections, mark.frame) === -1;
      if (outside) outsideSectionMarkIds.add(mark.id);
    }
  }
  return { rallies, sections, rallyIndexByMarkId, noEffectMarkIds, outsideSectionMarkIds };
}

/** Index of the rally whose inclusive span contains `frame`, or -1. */
export function rallyIndexAtFrame(rallies: readonly Rally[], frame: number): number {
  return rallies.findIndex((rally) => rally.startFrame <= frame && frame <= rally.endFrame);
}

export function nextMarkId(marks: readonly Mark[]): number {
  return marks.reduce((largest, mark) => Math.max(largest, mark.id), 0) + 1;
}

export type PlaceOutcome = "added" | "switched_side" | "already_there";

/**
 * Place a mark on a frame.
 *
 * A frame holds at most one contact, since only one player can hit the
 * shuttle at a time. A contact key on a frame that already has the other
 * side's contact switches that contact's side, keeping its unsure flag.
 * Pressing the same key twice on one frame changes nothing.
 */
export function placeMark(
  marks: readonly Mark[],
  kind: MarkKind,
  frame: number,
): { marks: Mark[]; mark: Mark; outcome: PlaceOutcome } {
  const sameKind = marks.find((mark) => mark.kind === kind && mark.frame === frame);
  if (sameKind !== undefined) return { marks: [...marks], mark: sameKind, outcome: "already_there" };
  const otherSide = isContact(kind) ? marks.find((mark) => isContact(mark.kind) && mark.frame === frame) : undefined;
  if (otherSide !== undefined) {
    const switched: Mark = { ...otherSide, kind };
    const switchedMarks = marks.map((mark) => (mark.id === otherSide.id ? switched : mark));
    return { marks: switchedMarks, mark: switched, outcome: "switched_side" };
  }
  const added: Mark = { id: nextMarkId(marks), kind, frame, unsure: false };
  return { marks: [...marks, added], mark: added, outcome: "added" };
}

export function deleteMark(marks: readonly Mark[], id: number): Mark[] {
  return marks.filter((mark) => mark.id !== id);
}

export function toggleUnsure(marks: readonly Mark[], id: number): Mark[] {
  return marks.map((mark) => (mark.id === id ? { ...mark, unsure: !mark.unsure } : mark));
}

function sameMark(left: Mark, right: Mark): boolean {
  return left.kind === right.kind && left.frame === right.frame && left.unsure === right.unsure;
}

/**
 * Re-apply this page's unsaved edits on top of marks another page saved.
 *
 * Both pages started from `saved`. Edits are found by comparing `saved` with
 * `page`: marks added, deleted, or changed (side or unsure flag). A deletion
 * or change applies only while the other page still has that mark unchanged,
 * because ids can be reused after a deletion. Added marks go through
 * `placeMark`, so the one-contact-per-frame rule still holds.
 */
export function mergeUnsavedMarks(theirs: readonly Mark[], saved: readonly Mark[], page: readonly Mark[]): Mark[] {
  const savedById = new Map(saved.map((mark) => [mark.id, mark]));
  const pageById = new Map(page.map((mark) => [mark.id, mark]));
  let merged: Mark[] = [];
  for (const mark of theirs) {
    const before = savedById.get(mark.id);
    const untouchedThere = before !== undefined && sameMark(before, mark);
    if (!untouchedThere) {
      merged.push(mark);
      continue;
    }
    // Deleted on this page: leave it out. Otherwise keep this page's version. Marks
    // never move, so the same id on another frame is a new mark that reused a deleted
    // mark's id; the loop below adds it.
    const mine = pageById.get(mark.id);
    if (mine !== undefined && mine.frame === mark.frame) merged.push(mine);
  }
  for (const mine of page) {
    const before = savedById.get(mine.id);
    const existedBefore = before !== undefined && before.frame === mine.frame;
    if (existedBefore) continue;
    const placed = placeMark(merged, mine.kind, mine.frame);
    merged = placed.marks;
    if (mine.unsure && !placed.mark.unsure) merged = toggleUnsure(merged, placed.mark.id);
  }
  return merged;
}

/** One undo step: the state before a change, and what the change did. */
export interface HistoryEntry<T> {
  state: T;
  description: string;
}

/** Undo and redo over whole snapshots, each with a short description for feedback. */
export class History<T> {
  private readonly undoStack: HistoryEntry<T>[] = [];
  private redoStack: HistoryEntry<T>[] = [];
  private readonly limit: number;

  constructor(limit = 1000) {
    this.limit = limit;
  }

  /** Remember the state from before a change. */
  record(previous: T, description: string): void {
    this.undoStack.push({ state: previous, description });
    if (this.undoStack.length > this.limit) this.undoStack.shift();
    this.redoStack = [];
  }

  /** The state to go back to, and the description of the change it undoes. */
  undo(current: T): HistoryEntry<T> | null {
    const entry = this.undoStack.pop();
    if (entry === undefined) return null;
    this.redoStack.push({ state: current, description: entry.description });
    return entry;
  }

  redo(current: T): HistoryEntry<T> | null {
    const entry = this.redoStack.pop();
    if (entry === undefined) return null;
    this.undoStack.push({ state: current, description: entry.description });
    return entry;
  }
}
