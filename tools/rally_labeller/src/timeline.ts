/**
 * Two timeline strips drawn on canvases.
 *
 * The overview strip spans the whole video. It shades each rally and
 * underlines each labelled section. The detail
 * strip shows a few seconds either side of the current frame, with each
 * contact as a circle lettered T (top player) or B (bottom player). Suggestions
 * from other sources show on both strips as a sparse crosshatched block for
 * each suggested rally, set apart from the solid shading of labelled rallies.
 * The detail strip also shows a tick for each suggested contact. Top and
 * bottom also differ in colour (blue and orange, which stay distinct for
 * red-green colour blindness).
 */

import type { SuggestionFile } from "./labelFile.ts";
import type { Mark, MarkLayout, MarkKind } from "./marks.ts";

export interface TimelineView {
  /** Source frame number of the copy's first frame; the overview strip spans the copy. */
  firstFrame: number;
  frameCount: number;
  fps: number;
  shownFrame: number;
  marks: readonly Mark[];
  layout: MarkLayout;
  selectedId: number | null;
  /** Null while suggestions are hidden. */
  suggestions: readonly SuggestionFile[] | null;
  /** Frames of suggested contacts marked done; their ticks are drawn faint. */
  suggestionsDone: ReadonlySet<number>;
}

export const COLOURS = {
  top: "#4ea1ff",
  bottom: "#ffa94d",
  boundary: "#d0d7de",
  break: "#8b949e",
  section: "#a371f7",
  rallyEven: "rgba(255, 255, 255, 0.10)",
  rallyOdd: "rgba(255, 255, 255, 0.18)",
  background: "#161b22",
  playhead: "#ffffff",
  selected: "rgba(255, 255, 255, 0.28)",
  tick: "#6e7681",
  suggestedRally: "rgba(210, 153, 34, 0.45)",
  suggestedRallyFill: "rgba(210, 153, 34, 0.12)",
  suggestedTop: "rgba(78, 161, 255, 0.85)",
  suggestedBottom: "rgba(255, 169, 77, 0.85)",
  suggestedUnknown: "rgba(139, 148, 158, 0.85)",
} as const;

const HATCH_SPACING_PX = 16;
const SUGGESTED_TICK_HALF_PX = 10;
const DONE_TICK_ALPHA = 0.25;
const HATCH_TILE = makeHatchTile();

const BOUNDARY_STYLES: Record<MarkKind, { letter: string; colour: string; dashed: boolean }> = {
  top: { letter: "", colour: COLOURS.top, dashed: false },
  bottom: { letter: "", colour: COLOURS.bottom, dashed: false },
  start: { letter: "⊢", colour: COLOURS.boundary, dashed: false },
  end: { letter: "⊣", colour: COLOURS.boundary, dashed: false },
  break: { letter: "‖", colour: COLOURS.break, dashed: true },
  section_start: { letter: "[", colour: COLOURS.section, dashed: false },
  section_end: { letter: "]", colour: COLOURS.section, dashed: false },
};

const MARK_RADIUS = 8;
const CONTACT_LETTERS = { top: "T", bottom: "B" } as const;
// Clicks this close to a mark select it instead of seeking.
const HIT_DISTANCE_PX = 10;

/** Size the canvas buffer to its displayed size and return a context in CSS pixels. */
function prepareCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * ratio);
  const height = Math.round(canvas.clientHeight * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  if (context === null) throw new Error("canvas 2D drawing is unavailable");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.fillStyle = COLOURS.background;
  context.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  return context;
}

/** One square tile crossed corner to corner both ways; repeated, it makes a crosshatch. */
function makeHatchTile(): HTMLCanvasElement {
  const tile = document.createElement("canvas");
  tile.width = HATCH_SPACING_PX;
  tile.height = HATCH_SPACING_PX;
  const context = tile.getContext("2d");
  if (context === null) throw new Error("canvas 2D drawing is unavailable");
  context.strokeStyle = COLOURS.suggestedRally;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, 0);
  context.lineTo(HATCH_SPACING_PX, HATCH_SPACING_PX);
  context.moveTo(HATCH_SPACING_PX, 0);
  context.lineTo(0, HATCH_SPACING_PX);
  context.stroke();
  return tile;
}

function crosshatch(context: CanvasRenderingContext2D): CanvasPattern {
  const pattern = context.createPattern(HATCH_TILE, "repeat");
  if (pattern === null) throw new Error("canvas patterns are unavailable");
  return pattern;
}

function drawSuggestedRallies(
  context: CanvasRenderingContext2D,
  view: TimelineView,
  xAtFrame: (frame: number) => number,
  height: number,
): void {
  const hatch = crosshatch(context);
  for (const file of view.suggestions ?? []) {
    for (const rally of file.rallies) {
      const left = xAtFrame(rally.start_frame);
      const right = Math.max(xAtFrame(rally.end_frame + 1), left + 1);
      // A faint fill under the hatch keeps a rally visible when it is narrower than the hatch spacing.
      context.fillStyle = COLOURS.suggestedRallyFill;
      context.fillRect(left, 0, right - left, height);
      context.fillStyle = hatch;
      context.fillRect(left, 0, right - left, height);
    }
  }
}

function drawSuggestedContacts(context: CanvasRenderingContext2D, view: TimelineView, geometry: DetailGeometry): void {
  const lastFrame = geometry.firstFrame + geometry.framesShown - 1;
  const centreY = geometry.height / 2;
  context.lineWidth = 3;
  for (const file of view.suggestions ?? []) {
    for (const contact of file.contacts) {
      if (contact.frame < geometry.firstFrame || contact.frame > lastFrame) continue;
      const x = detailX(geometry, contact.frame);
      context.strokeStyle =
        contact.side === "top" ? COLOURS.suggestedTop : contact.side === "bottom" ? COLOURS.suggestedBottom : COLOURS.suggestedUnknown;
      context.globalAlpha = view.suggestionsDone.has(contact.frame) ? DONE_TICK_ALPHA : 1;
      context.beginPath();
      context.moveTo(x, centreY - SUGGESTED_TICK_HALF_PX);
      context.lineTo(x, centreY + SUGGESTED_TICK_HALF_PX);
      context.stroke();
      context.globalAlpha = 1;
    }
  }
}

function drawRallySpans(
  context: CanvasRenderingContext2D,
  view: TimelineView,
  xAtFrame: (frame: number) => number,
  height: number,
): void {
  view.layout.rallies.forEach((rally, rallyIndex) => {
    const left = xAtFrame(rally.startFrame);
    const right = Math.max(xAtFrame(rally.endFrame + 1), left + 1);
    context.fillStyle = rallyIndex % 2 === 0 ? COLOURS.rallyEven : COLOURS.rallyOdd;
    context.fillRect(left, 0, right - left, height);
  });
}

export function drawOverview(canvas: HTMLCanvasElement, view: TimelineView): void {
  const context = prepareCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const xAtFrame = (frame: number) => ((frame - view.firstFrame) / view.frameCount) * width;
  drawRallySpans(context, view, xAtFrame, height);
  drawSuggestedRallies(context, view, xAtFrame, height);
  context.fillStyle = COLOURS.section;
  for (const section of view.layout.sections) {
    const left = xAtFrame(section.startFrame);
    const right = xAtFrame((section.endFrame ?? view.firstFrame + view.frameCount - 1) + 1);
    context.fillRect(left, height - 4, Math.max(right - left, 2), 4);
  }

  const selected = view.marks.find((mark) => mark.id === view.selectedId);
  if (selected !== undefined) {
    context.fillStyle = COLOURS.selected;
    context.fillRect(xAtFrame(selected.frame) - 2, 0, 4, height);
  }
  context.fillStyle = COLOURS.playhead;
  context.fillRect(xAtFrame(view.shownFrame) - 1, 0, 2, height);
}

export function overviewFrameAt(canvas: HTMLCanvasElement, clientX: number, view: TimelineView): number {
  const bounds = canvas.getBoundingClientRect();
  return view.firstFrame + Math.floor(((clientX - bounds.left) / bounds.width) * view.frameCount);
}

interface DetailGeometry {
  firstFrame: number;
  framesShown: number;
  width: number;
  height: number;
}

function detailGeometry(canvas: HTMLCanvasElement, view: TimelineView, halfWindowSeconds: number): DetailGeometry {
  const halfWindowFrames = Math.round(halfWindowSeconds * view.fps);
  return {
    firstFrame: view.shownFrame - halfWindowFrames,
    framesShown: 2 * halfWindowFrames + 1,
    width: canvas.clientWidth,
    height: canvas.clientHeight,
  };
}

function detailX(geometry: DetailGeometry, frame: number): number {
  // Centre each frame in its own slot so the shown frame sits mid-strip.
  return ((frame - geometry.firstFrame + 0.5) / geometry.framesShown) * geometry.width;
}

function drawSecondTicks(context: CanvasRenderingContext2D, view: TimelineView, geometry: DetailGeometry): void {
  context.fillStyle = COLOURS.tick;
  const firstSecond = Math.ceil(geometry.firstFrame / view.fps);
  const lastSecond = Math.floor((geometry.firstFrame + geometry.framesShown) / view.fps);
  for (let second = Math.max(firstSecond, 0); second <= lastSecond; second += 1) {
    const x = detailX(geometry, second * view.fps);
    context.fillRect(x, geometry.height - 6, 1, 6);
  }
}

function drawMark(context: CanvasRenderingContext2D, mark: Mark, x: number, height: number): void {
  if (mark.kind === "top" || mark.kind === "bottom") {
    // Colour and letter both carry the side, so it reads without relying on colour alone.
    const y = height / 2;
    const colour = mark.kind === "top" ? COLOURS.top : COLOURS.bottom;
    context.beginPath();
    context.arc(x, y, MARK_RADIUS, 0, 2 * Math.PI);
    context.lineWidth = 2;
    if (mark.unsure) {
      context.fillStyle = COLOURS.background;
      context.fill();
      context.strokeStyle = colour;
      context.stroke();
    } else {
      context.fillStyle = colour;
      context.fill();
    }
    context.fillStyle = mark.unsure ? colour : COLOURS.background;
    context.font = "bold 11px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(CONTACT_LETTERS[mark.kind], x, y + 0.5);
    context.textBaseline = "alphabetic";
    return;
  }
  const style = BOUNDARY_STYLES[mark.kind];
  const letter = style.letter;
  context.strokeStyle = style.colour;
  context.setLineDash(style.dashed ? [4, 3] : []);
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(x, 12);
  context.lineTo(x, height);
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = context.strokeStyle;
  context.font = "bold 11px system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(mark.unsure ? `${letter}?` : letter, x, 10);
}

export function drawDetail(canvas: HTMLCanvasElement, view: TimelineView, halfWindowSeconds: number): void {
  const context = prepareCanvas(canvas);
  const geometry = detailGeometry(canvas, view, halfWindowSeconds);
  const lastFrame = geometry.firstFrame + geometry.framesShown - 1;
  drawRallySpans(context, view, (frame) => detailX(geometry, frame - 0.5), geometry.height);
  drawSuggestedRallies(context, view, (frame) => detailX(geometry, frame - 0.5), geometry.height);

  drawSecondTicks(context, view, geometry);
  drawSuggestedContacts(context, view, geometry);

  for (const mark of view.marks) {
    if (mark.frame < geometry.firstFrame || mark.frame > lastFrame) continue;
    const x = detailX(geometry, mark.frame);
    if (mark.id === view.selectedId) {
      context.fillStyle = COLOURS.selected;
      context.fillRect(x - MARK_RADIUS - 3, 0, 2 * MARK_RADIUS + 6, geometry.height);
    }
    drawMark(context, mark, x, geometry.height);
  }

  const playheadX = detailX(geometry, view.shownFrame);
  context.fillStyle = COLOURS.playhead;
  context.fillRect(playheadX - 1, 0, 2, geometry.height);
}

/** The mark under a click on the detail strip, or else the frame clicked. */
export function detailHit(
  canvas: HTMLCanvasElement,
  clientX: number,
  view: TimelineView,
  halfWindowSeconds: number,
): { markId: number | null; frame: number } {
  const geometry = detailGeometry(canvas, view, halfWindowSeconds);
  const x = clientX - canvas.getBoundingClientRect().left;
  let nearestId: number | null = null;
  let nearestDistance = HIT_DISTANCE_PX;
  for (const mark of view.marks) {
    const distance = Math.abs(detailX(geometry, mark.frame) - x);
    if (distance <= nearestDistance) {
      nearestId = mark.id;
      nearestDistance = distance;
    }
  }
  const frame = geometry.firstFrame + Math.floor((x / geometry.width) * geometry.framesShown);
  return { markId: nearestId, frame };
}
