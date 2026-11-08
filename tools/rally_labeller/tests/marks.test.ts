import { describe, expect, it } from "vitest";

import { KEY_BINDINGS, bindingFor } from "../src/keymap.ts";
import { formatTime, shortTitle, suggestionsInPart } from "../src/markList.ts";
import { buildLabelFile, checkSaveRequest, checkSuggestionFile, type VideoInfo } from "../src/labelFile.ts";
import { History, layOutMarks, mergeUnsavedMarks, placeMark, sortMarks, type Mark, type MarkKind } from "../src/marks.ts";

const FPS = 50;
const GAP_SECONDS = 4;

function marksFrom(entries: [MarkKind, number][]): Mark[] {
  return entries.map(([kind, frame], index) => ({ id: index + 1, kind, frame, unsure: false }));
}

function rallySpans(entries: [MarkKind, number][], gapSeconds = GAP_SECONDS) {
  const layout = layOutMarks(marksFrom(entries), FPS, gapSeconds);
  return layout.rallies.map((rally) => [rally.startFrame, rally.endFrame, rally.endedBy, rally.contacts.length]);
}

describe("layOutMarks rallies", () => {
  it("runs an unmarked rally from its first to its last contact", () => {
    expect(rallySpans([["top", 100], ["bottom", 150], ["top", 200]])).toEqual([[100, 200, "last_mark", 3]]);
  });

  it("uses start and end marks when they are placed", () => {
    expect(rallySpans([["start", 80], ["top", 100], ["bottom", 150], ["end", 190]])).toEqual([[80, 190, "end_mark", 2]]);
  });

  it("ends a rally at its last contact after a break, and starts the next at the next contact", () => {
    expect(rallySpans([["top", 100], ["bottom", 150], ["break", 160], ["top", 170], ["bottom", 200]])).toEqual([
      [100, 150, "break_mark", 2],
      [170, 200, "last_mark", 2],
    ]);
  });

  it("splits on a pause longer than the gap, and only then", () => {
    const longPause = 150 + GAP_SECONDS * FPS + 1;
    expect(rallySpans([["top", 100], ["bottom", 150], ["top", longPause]])).toEqual([
      [100, 150, "long_gap", 2],
      [longPause, longPause, "last_mark", 1],
    ]);
    expect(rallySpans([["top", 100], ["bottom", 100 + GAP_SECONDS * FPS]])).toEqual([[100, 300, "last_mark", 2]]);
    expect(rallySpans([["top", 100], ["bottom", 600]], 20)).toEqual([[100, 600, "last_mark", 2]]);
  });

  it("closes an open rally when a new start mark appears", () => {
    expect(rallySpans([["top", 100], ["bottom", 150], ["start", 160], ["top", 170]])).toEqual([
      [100, 150, "next_start_mark", 2],
      [160, 170, "last_mark", 1],
    ]);
  });

  it("orders a start before a contact, and an end after one, on the same frame", () => {
    expect(rallySpans([["end", 150], ["bottom", 150], ["top", 100], ["start", 100]])).toEqual([[100, 150, "end_mark", 2]]);
  });

  it("flags end and break marks that have no rally to close", () => {
    const layout = layOutMarks(marksFrom([["end", 50], ["top", 100], ["break", 150], ["break", 160]]), FPS, GAP_SECONDS);
    expect([...layout.noEffectMarkIds]).toEqual([1, 4]);
  });
});

describe("layOutMarks sections", () => {
  it("pairs section marks and leaves the last section open until it is closed", () => {
    const layout = layOutMarks(
      marksFrom([["section_start", 0], ["top", 100], ["section_end", 500], ["section_start", 1000], ["top", 1100]]),
      FPS,
      GAP_SECONDS,
    );
    expect(layout.sections).toEqual([
      { startFrame: 0, endFrame: 500 },
      { startFrame: 1000, endFrame: null },
    ]);
    expect(layout.rallies.map((rally) => rally.sectionIndex)).toEqual([0, 1]);
    expect(layout.outsideSectionMarkIds.size).toBe(0);
  });

  it("flags marks outside every section and rallies that cross a section end", () => {
    const layout = layOutMarks(
      marksFrom([["section_start", 0], ["top", 90], ["bottom", 110], ["section_end", 100], ["top", 2000]]),
      FPS,
      GAP_SECONDS,
    );
    expect(layout.rallies.map((rally) => rally.sectionIndex)).toEqual([null, null]);
    expect([...layout.outsideSectionMarkIds]).toEqual([3, 5]);
  });
});

describe("mark editing", () => {
  const kindsAndFrames = (marks: Mark[]) => marks.map((mark) => [mark.kind, mark.frame]);

  it("ignores a second identical mark on the same frame", () => {
    const first = placeMark([], "top", 10);
    const second = placeMark(first.marks, "top", 10);
    expect(second.outcome).toBe("already_there");
    expect(second.marks).toHaveLength(1);
  });

  it("switches a contact's side instead of adding a second contact on its frame", () => {
    const bottom = placeMark([], "bottom", 10);
    const flagged = [{ ...bottom.marks[0], unsure: true }];
    const top = placeMark(flagged, "top", 10);
    expect(top.outcome).toBe("switched_side");
    expect(top.marks).toEqual([{ id: 1, kind: "top", frame: 10, unsure: true }]);
  });

  it("adds a boundary mark on a frame that already has a contact", () => {
    const contact = placeMark([], "top", 10);
    const withBreak = placeMark(contact.marks, "break", 10);
    expect(withBreak.outcome).toBe("added");
    expect(kindsAndFrames(withBreak.marks)).toEqual([["top", 10], ["break", 10]]);
  });

  it("undoes and redoes in order with descriptions, and a new change clears redo", () => {
    const history = new History<number>();
    history.record(1, "first");
    history.record(2, "second");
    expect(history.undo(3)).toEqual({ state: 2, description: "second" });
    expect(history.undo(2)).toEqual({ state: 1, description: "first" });
    expect(history.redo(1)).toEqual({ state: 2, description: "first" });
    history.record(2, "third");
    expect(history.redo(5)).toBeNull();
  });
});

describe("merging unsaved edits", () => {
  it("keeps a new mark that reused the id of a mark deleted on both pages", () => {
    const saved: Mark[] = [{ id: 1, kind: "top", frame: 12000, unsure: false }];
    const theirs: Mark[] = [];
    // This page deleted the top contact, then added a bottom one, which took the freed id.
    const page: Mark[] = [{ id: 1, kind: "bottom", frame: 12001, unsure: false }];
    const merged = mergeUnsavedMarks(theirs, saved, page);
    expect(merged.map((mark) => `${mark.kind}@${mark.frame}`)).toEqual(["bottom@12001"]);
  });

  const saved = marksFrom([["top", 100], ["bottom", 150], ["top", 200]]);

  it("keeps both pages' additions and this page's deletions", () => {
    const theirs = [...saved, { id: 4, kind: "bottom" as const, frame: 300, unsure: false }];
    const page = [saved[0], saved[2], { id: 4, kind: "top" as const, frame: 400, unsure: false }];
    const merged = mergeUnsavedMarks(theirs, saved, page);
    expect(sortMarks(merged).map((mark) => [mark.kind, mark.frame])).toEqual([
      ["top", 100],
      ["top", 200],
      ["bottom", 300],
      ["top", 400],
    ]);
    expect(new Set(merged.map((mark) => mark.id)).size).toBe(merged.length);
  });

  it("applies this page's side switch and unsure flag to marks the other page left alone", () => {
    const page = [{ ...saved[0], kind: "bottom" as const }, { ...saved[1], unsure: true }, saved[2]];
    expect(mergeUnsavedMarks(saved, saved, page)).toEqual(page);
  });

  it("keeps the other page's change when both pages changed one mark", () => {
    const theirs = [{ ...saved[0], frame: 101 }, saved[1], saved[2]];
    const page = [{ ...saved[0], unsure: true }, saved[1], saved[2]];
    expect(mergeUnsavedMarks(theirs, saved, page)[0]).toEqual({ id: 1, kind: "top", frame: 101, unsure: false });
  });

  it("switches the side of the other page's contact rather than adding a second one on that frame", () => {
    const theirs = [...saved, { id: 4, kind: "top" as const, frame: 300, unsure: false }];
    const page = [...saved, { id: 4, kind: "bottom" as const, frame: 300, unsure: false }];
    const merged = mergeUnsavedMarks(theirs, saved, page);
    expect(merged.filter((mark) => mark.frame === 300)).toEqual([{ id: 4, kind: "bottom", frame: 300, unsure: false }]);
  });
});

describe("label files", () => {
  const video: VideoInfo = {
    schema: "rally-labeller-video/1",
    video_id: "abc",
    fps: "50/1",
    first_frame: 0,
    frame_count: 1000,
    source_frame_count: 1000,
    has_audio: false,
    source: { file_name: "abc.mp4", sha256: "0".repeat(64), width: 1920, height: 1080, youtube: null },
  };

  it("saves derived rallies with snake_case fields", () => {
    const file = buildLabelFile(video, marksFrom([["top", 100], ["bottom", 150]]), GAP_SECONDS, 3);
    expect(file.rallies).toEqual([
      {
        start_frame: 100,
        start_marked: false,
        end_frame: 150,
        end_marked: false,
        ended_by: "last_mark",
        section_index: null,
        contacts: [
          { frame: 100, side: "top", unsure: false },
          { frame: 150, side: "bottom", unsure: false },
        ],
      },
    ]);
  });

  it("rejects suggestions from a longer video", () => {
    const suggestions = {
      schema: "rally-suggestions/1",
      name: "annotator",
      rallies: [{ start_frame: 900, end_frame: 1200 }],
      contacts: [],
    };
    expect(() => checkSuggestionFile(suggestions, video)).toThrow("outside the video");
  });

  it("rejects marks outside the video", () => {
    const request = { base_revision: 0, rally_gap_seconds: 4, marks: marksFrom([["top", 1000]]) };
    expect(() => checkSaveRequest(request, video)).toThrow("outside the video");
  });

  it("rejects two contacts on one frame, and a missing rally gap", () => {
    const twoContacts = { base_revision: 0, rally_gap_seconds: 4, marks: marksFrom([["top", 10], ["bottom", 10]]) };
    expect(() => checkSaveRequest(twoContacts, video)).toThrow("two contact marks on frame 10");
    const noGap = { base_revision: 0, rally_gap_seconds: null, marks: [] };
    expect(() => checkSaveRequest(noGap, video)).toThrow("rally_gap_seconds");
  });
});

describe("formatTime", () => {
  it("carries rounding into the minutes", () => {
    expect(formatTime(599.999)).toBe("10:00.00");
    expect(formatTime(65.5)).toBe("1:05.50");
  });
});

describe("keys", () => {
  const press = (code: string, modifiers: Partial<KeyboardEvent> = {}) =>
    bindingFor({ code, shiftKey: false, ctrlKey: false, altKey: false, metaKey: false, ...modifiers })?.action;

  it("maps plain and shifted keys", () => {
    expect(press("KeyG")).toBe("stepForward");
    expect(press("KeyG", { shiftKey: true })).toBe("jumpForward");
    expect(press("BracketRight")).toBe("markBreak");
    expect(press("KeyS")).toBe("nextSuggestedContact");
    expect(press("KeyA", { shiftKey: true })).toBe("previousSuggestedRally");
    expect(press("KeyD")).toBe("toggleSuggestionDone");
  });

  it("treats Ctrl+Z as undo but leaves other Ctrl keys to the browser", () => {
    expect(press("KeyZ", { ctrlKey: true })).toBe("undo");
    expect(press("KeyZ", { ctrlKey: true, shiftKey: true })).toBe("redo");
    expect(press("KeyV", { ctrlKey: true })).toBeUndefined();
  });

  it("repeats only frame steps, jumps and mark-to-mark moves when held", () => {
    const repeating = KEY_BINDINGS.filter((binding) => binding.repeatEveryMs !== null).map((binding) => binding.keyLabel);
    expect(repeating.sort()).toEqual(["F", "G", "R", "T", "⇧F", "⇧G"].sort());
  });
});

describe("suggestions in a part", () => {
  const file = {
    schema: "rally-suggestions/1" as const,
    name: "test",
    rallies: [
      { start_frame: 500, end_frame: 600 },
      { start_frame: 100, end_frame: 200 },
      { start_frame: 900, end_frame: 950 },
    ],
    contacts: [
      { frame: 550, side: "bottom" as const, confidence: 0.6 },
      { frame: 120, side: "top" as const, confidence: null },
      { frame: 300, side: null, confidence: 0.3 },
      { frame: 920, side: "top" as const, confidence: 0.9 },
    ],
  };

  it("keeps what falls inside the part, sorted, with each contact's rally", () => {
    // The part covers frames 100-899, so the rally at 900 and its contact are left out.
    const suggested = suggestionsInPart([file], 100, 900);
    expect(suggested.rallies).toEqual([
      { startFrame: 100, endFrame: 200 },
      { startFrame: 500, endFrame: 600 },
    ]);
    expect(suggested.contacts.map((contact) => [contact.frame, contact.rallyIndex])).toEqual([
      [120, 0],
      [300, null],
      [550, 1],
    ]);
  });
});

describe("shortTitle", () => {
  it("drops hashtags and cuts long titles", () => {
    expect(shortTitle("Final | Rourkela #badminton #bwf")).toBe("Final | Rourkela");
    const long = shortTitle("A".repeat(80));
    expect(long.length).toBe(60);
    expect(long.endsWith("…")).toBe(true);
  });
});
