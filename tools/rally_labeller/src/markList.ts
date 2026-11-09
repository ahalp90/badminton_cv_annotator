/**
 * The side panel's two lists: marks grouped under one heading per rally, and
 * below them the suggested contacts grouped under one heading per suggested rally.
 */

import type { SuggestionFile } from "./labelFile.ts";
import { sortMarks, type Mark, type MarkKind, type MarkLayout, type RallyEnding } from "./marks.ts";

// Shapes rather than letters, so they can't be mistaken for key hints.
export const KIND_SYMBOLS: Record<MarkKind, string> = {
  top: "▲",
  bottom: "▼",
  start: "⊢",
  end: "⊣",
  break: "‖",
  section_start: "[",
  section_end: "]",
};
export const KIND_NAMES: Record<MarkKind, string> = {
  top: "Top contact",
  bottom: "Bottom contact",
  start: "Rally start",
  end: "Rally end",
  break: "Rally break",
  section_start: "Section start",
  section_end: "Section end",
};

const SIDE_NAMES = { top: "Top", bottom: "Bottom" } as const;

const TAG_TITLES = {
  unsure: "Flagged as unsure (Tab)",
  "no effect": "Changes nothing: a rally end or break with no rally open, or a section mark with no partner",
  "outside sections": "Outside every labelled section, so nothing here counts as labelled yet",
} as const;

const ENDING_TEXT: Record<RallyEnding, string> = {
  end_mark: "end mark",
  break_mark: "break mark",
  next_start_mark: "next start mark",
  long_gap: "long gap",
  last_mark: "last mark so far",
};

export interface MarkListHandlers {
  select: (markId: number) => void;
  remove: (markId: number) => void;
}

export function formatTime(seconds: number): string {
  // Round once, to hundredths, so 599.999 s shows as 10:00.00 rather than 9:60.00.
  const hundredths = Math.round(seconds * 100);
  const minutes = Math.floor(hundredths / 6000);
  const remainder = ((hundredths % 6000) / 100).toFixed(2).padStart(5, "0");
  return `${minutes}:${remainder}`;
}

const TITLE_CHARS = 60;

/** A YouTube title short enough for a drop-down: hashtags removed, long titles cut. */
export function shortTitle(title: string): string {
  const plain = title.replace(/#\S+/g, "").replace(/\s+/g, " ").trim();
  return plain.length <= TITLE_CHARS ? plain : `${plain.slice(0, TITLE_CHARS - 1).trimEnd()}…`;
}

/** Whole seconds, for times that only need to be recognisable, such as a part's range. */
export function formatClock(seconds: number): string {
  const whole = Math.round(seconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const rest = String(whole % 60).padStart(2, "0");
  return hours > 0 ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}` : `${minutes}:${rest}`;
}

export function countText(count: number, singular: string, plural: string): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function rallyHeading(layout: MarkLayout, rallyIndex: number): HTMLElement {
  const rally = layout.rallies[rallyIndex];
  const heading = document.createElement("div");
  heading.className = "rally-heading";
  heading.dataset.rallyIndex = String(rallyIndex);
  const contactText = countText(rally.contacts.length, "contact", "contacts");
  const title = document.createElement("strong");
  title.textContent = `Rally ${rallyIndex + 1} · ${contactText}`;
  const detail = document.createElement("span");
  detail.className = "muted";
  const startText = rally.startMarked ? "start marked" : "starts at first contact";
  const sectionText = layout.sections.length > 0 && rally.sectionIndex === null ? " · not inside one section" : "";
  detail.textContent = `${startText} · ended by ${ENDING_TEXT[rally.endedBy]}${sectionText}`;
  if (sectionText !== "") detail.title = "The rally crosses a section end, or lies outside every section";
  heading.append(title, detail);
  if (rally.contacts.length === 0 || sectionText !== "") heading.classList.add("warning");
  return heading;
}

function markRow(mark: Mark, fps: number, layout: MarkLayout, selected: boolean, handlers: MarkListHandlers) {
  const row = document.createElement("div");
  row.className = `mark-row kind-${mark.kind}`;
  row.dataset.markId = String(mark.id);
  if (selected) row.classList.add("selected");

  const symbol = document.createElement("span");
  symbol.className = "symbol";
  symbol.textContent = KIND_SYMBOLS[mark.kind];
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = KIND_NAMES[mark.kind];
  const position = document.createElement("span");
  position.className = "position";
  position.textContent = `${mark.frame} · ${formatTime(mark.frame / fps)}`;
  row.append(symbol, name, position);

  for (const [shown, text] of [
    [mark.unsure, "unsure"],
    [layout.noEffectMarkIds.has(mark.id), "no effect"],
    [layout.outsideSectionMarkIds.has(mark.id), "outside sections"],
  ] as const) {
    if (!shown) continue;
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = text;
    tag.title = TAG_TITLES[text];
    row.append(tag);
  }

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove";
  remove.title = "Delete this mark";
  remove.textContent = "×";
  remove.addEventListener("click", (event) => {
    event.stopPropagation();
    handlers.remove(mark.id);
  });
  row.append(remove);
  row.addEventListener("click", () => handlers.select(mark.id));
  return row;
}

/**
 * Rebuild the list. Marks outside every rally (section marks, and end or
 * break marks with no rally to close) appear in frame order between the
 * rallies around them.
 */
export function renderMarkList(
  container: HTMLElement,
  marks: readonly Mark[],
  layout: MarkLayout,
  fps: number,
  selectedId: number | null,
  handlers: MarkListHandlers,
): void {
  const items: HTMLElement[] = [];
  let previousRallyIndex: number | undefined;
  for (const mark of sortMarks(marks)) {
    const rallyIndex = layout.rallyIndexByMarkId.get(mark.id);
    // Marks outside rallies (such as a section end mid-rally) don't restart the heading.
    if (rallyIndex !== undefined && rallyIndex !== previousRallyIndex) {
      items.push(rallyHeading(layout, rallyIndex));
      previousRallyIndex = rallyIndex;
    }
    items.push(markRow(mark, fps, layout, mark.id === selectedId, handlers));
  }
  if (items.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No marks yet. Press V or B at a contact.";
    items.push(empty);
  }
  container.replaceChildren(...items);
  container.querySelector(".selected")?.scrollIntoView({ block: "nearest" });
}

/** Highlight the heading of the rally that contains the current frame. */
export function highlightRally(container: HTMLElement, rallyIndex: number): void {
  for (const heading of container.querySelectorAll<HTMLElement>(".rally-heading")) {
    heading.classList.toggle("current", heading.dataset.rallyIndex === String(rallyIndex));
  }
}

/** A suggested rally inside the open part. Ends are inclusive. */
export interface SuggestedRally {
  startFrame: number;
  endFrame: number;
}

/** A suggested contact inside the open part. */
export interface SuggestedContact {
  frame: number;
  side: "top" | "bottom" | null;
  confidence: number | null;
  /** Index into the part's suggested rallies, or null when the contact falls outside them all. */
  rallyIndex: number | null;
}

export interface PartSuggestions {
  /** Sorted by start frame. */
  rallies: SuggestedRally[];
  /** Sorted by frame; navigation and the list both rely on this order. */
  contacts: SuggestedContact[];
}

/** The suggested rallies and contacts from every file that fall inside frames [firstFrame, endFrame). */
export function suggestionsInPart(files: readonly SuggestionFile[], firstFrame: number, endFrame: number): PartSuggestions {
  const rallies: SuggestedRally[] = [];
  const contacts: SuggestedContact[] = [];
  for (const file of files) {
    for (const rally of file.rallies) {
      const overlapsPart = rally.end_frame >= firstFrame && rally.start_frame < endFrame;
      if (overlapsPart) rallies.push({ startFrame: rally.start_frame, endFrame: rally.end_frame });
    }
  }
  rallies.sort((left, right) => left.startFrame - right.startFrame);
  for (const file of files) {
    for (const contact of file.contacts) {
      const insidePart = contact.frame >= firstFrame && contact.frame < endFrame;
      if (!insidePart) continue;
      const rallyIndex = rallies.findIndex((rally) => rally.startFrame <= contact.frame && contact.frame <= rally.endFrame);
      contacts.push({
        frame: contact.frame,
        side: contact.side,
        confidence: contact.confidence,
        rallyIndex: rallyIndex >= 0 ? rallyIndex : null,
      });
    }
  }
  contacts.sort((left, right) => left.frame - right.frame);
  return { rallies, contacts };
}

function suggestedRallyHeading(suggested: PartSuggestions, rallyIndex: number, fps: number): HTMLElement {
  const rally = suggested.rallies[rallyIndex];
  const contactCount = suggested.contacts.filter((contact) => contact.rallyIndex === rallyIndex).length;
  const heading = document.createElement("div");
  heading.className = "rally-heading";
  heading.dataset.startFrame = String(rally.startFrame);
  heading.dataset.endFrame = String(rally.endFrame);
  const title = document.createElement("strong");
  title.textContent = `Suggested rally ${rallyIndex + 1} · ${countText(contactCount, "contact", "contacts")}`;
  const detail = document.createElement("span");
  detail.className = "muted";
  detail.textContent = `${formatTime(rally.startFrame / fps)} to ${formatTime(rally.endFrame / fps)}`;
  heading.append(title, detail);
  return heading;
}

export interface SuggestionListHandlers {
  goTo: (frame: number) => void;
  toggleDone: (frame: number) => void;
}

function suggestedContactRow(contact: SuggestedContact, fps: number, done: boolean, handlers: SuggestionListHandlers): HTMLElement {
  const row = document.createElement("div");
  row.className = `mark-row kind-${contact.side ?? "unknown"}`;
  if (done) row.classList.add("done");
  row.title = "Double-click to mark done, or not done again";
  row.dataset.frame = String(contact.frame);
  const symbol = document.createElement("span");
  symbol.className = "symbol";
  symbol.textContent = contact.side === null ? "?" : KIND_SYMBOLS[contact.side];
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = contact.side === null ? "Side unknown" : SIDE_NAMES[contact.side];
  const position = document.createElement("span");
  position.className = "position";
  position.textContent = `${contact.frame} · ${formatTime(contact.frame / fps)}`;
  row.append(symbol, name, position);
  if (contact.confidence !== null) {
    const confidence = document.createElement("span");
    confidence.className = "muted";
    confidence.textContent = `confidence ${contact.confidence}`;
    row.append(confidence);
  }
  row.addEventListener("click", () => handlers.goTo(contact.frame));
  row.addEventListener("dblclick", () => handlers.toggleDone(contact.frame));
  return row;
}

/** Rebuild the suggested-contact list. Contacts outside every suggested rally sit between the rallies around them. */
export function renderSuggestionList(
  container: HTMLElement,
  suggested: PartSuggestions,
  fps: number,
  doneFrames: ReadonlySet<number>,
  handlers: SuggestionListHandlers,
): void {
  const items: HTMLElement[] = [];
  let previousRallyIndex: number | null = null;
  for (const contact of suggested.contacts) {
    if (contact.rallyIndex !== null && contact.rallyIndex !== previousRallyIndex) {
      items.push(suggestedRallyHeading(suggested, contact.rallyIndex, fps));
      previousRallyIndex = contact.rallyIndex;
    }
    items.push(suggestedContactRow(contact, fps, doneFrames.has(contact.frame), handlers));
  }
  container.replaceChildren(...items);
}

/** Highlight the suggested contacts on the current frame, and the suggested rally around it. */
export function highlightSuggestions(container: HTMLElement, frame: number): void {
  for (const row of container.querySelectorAll<HTMLElement>(".mark-row")) {
    const onThisFrame = Number(row.dataset.frame) === frame;
    // Scroll only when a row first becomes current, so the list doesn't fight the user's own scrolling.
    if (onThisFrame && !row.classList.contains("current")) row.scrollIntoView({ block: "nearest" });
    row.classList.toggle("current", onThisFrame);
  }
  for (const heading of container.querySelectorAll<HTMLElement>(".rally-heading")) {
    const insideRally = Number(heading.dataset.startFrame) <= frame && frame <= Number(heading.dataset.endFrame);
    heading.classList.toggle("current", insideRally);
  }
}
