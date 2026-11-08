/**
 * Saved file formats shared by the browser app and the dev-server API.
 *
 * Label files use snake_case keys so Python code can read them directly.
 * `marks` is the source of truth. The server rebuilds `rallies` and
 * `sections` from the marks on every save, using the saved
 * `rally_gap_seconds`.
 */

import { MARK_KINDS, isContact, layOutMarks, type Mark, type MarkKind, type RallyEnding } from "./marks.ts";

export const LABEL_SCHEMA = "rally-labels/1";
export const VIDEO_INFO_SCHEMA = "rally-labeller-video/1";
export const SUGGESTION_SCHEMA = "rally-suggestions/1";

/**
 * Rough rallies and contacts from another source, such as an auto-annotator
 * run or a language model's first pass. Shown as hints; never saved as labels.
 * Frames use the same numbering as the labels, and rally ends are inclusive.
 */
export interface SuggestionFile {
  schema: typeof SUGGESTION_SCHEMA;
  /** Short name shown in the labeller, for example "annotator-seed0". */
  name: string;
  rallies: { start_frame: number; end_frame: number }[];
  contacts: { frame: number; side: "top" | "bottom" | null; confidence: number | null }[];
}

export function checkSuggestionFile(value: unknown, video: VideoInfo): SuggestionFile {
  const file = value as SuggestionFile;
  if (file?.schema !== SUGGESTION_SCHEMA) throw new Error(`suggestion schema must be ${SUGGESTION_SCHEMA}`);
  if (typeof file.name !== "string" || file.name === "") throw new Error("suggestions need a name");
  const insideVideo = (frame: unknown) =>
    Number.isInteger(frame) && (frame as number) >= 0 && (frame as number) < video.source_frame_count;
  for (const rally of file.rallies) {
    if (!insideVideo(rally.start_frame) || !insideVideo(rally.end_frame) || rally.end_frame < rally.start_frame) {
      throw new Error(`${file.name}: suggested rally ${rally.start_frame}-${rally.end_frame} is outside the video`);
    }
  }
  for (const contact of file.contacts) {
    if (!insideVideo(contact.frame)) throw new Error(`${file.name}: suggested contact ${contact.frame} is outside the video`);
    if (![null, "top", "bottom"].includes(contact.side)) throw new Error(`${file.name}: unknown side ${contact.side}`);
  }
  return file;
}

export interface YoutubeDetails {
  id: string;
  url: string;
  title: string;
  channel: string;
  /** YYYYMMDD, as YouTube reports it. */
  upload_date: string;
  duration_seconds: number;
}

/** Details of the original file the browser copy was made from. */
export interface SourceDetails {
  file_name: string;
  sha256: string;
  width: number;
  height: number;
  youtube: YoutubeDetails | null;
}

/**
 * Written by prepare_video.py next to each browser copy. A copy holds source
 * frames [first_frame, first_frame + frame_count); a whole video is the copy
 * starting at frame 0.
 */
export interface VideoInfo {
  schema: typeof VIDEO_INFO_SCHEMA;
  video_id: string;
  /** Exact frame rate as "numerator/denominator". */
  fps: string;
  first_frame: number;
  /** Frames in this copy. */
  frame_count: number;
  /** Frames in the whole source video. */
  source_frame_count: number;
  has_audio: boolean;
  source: SourceDetails;
}

export interface SavedRally {
  start_frame: number;
  start_marked: boolean;
  end_frame: number;
  end_marked: boolean;
  ended_by: RallyEnding;
  /** Labelled section containing the whole rally, or null when it falls outside or across sections. */
  section_index: number | null;
  contacts: { frame: number; side: "top" | "bottom"; unsure: boolean }[];
}

export interface SavedSection {
  start_frame: number;
  /** Inclusive, or null while the section is still open. */
  end_frame: number | null;
}

export interface LabelFile {
  schema: typeof LABEL_SCHEMA;
  video_id: string;
  fps: string;
  frame_count: number;
  source: SourceDetails;
  /** Goes up by one on every save; the server rejects a save based on an older revision. */
  revision: number;
  rally_gap_seconds: number;
  marks: Mark[];
  sections: SavedSection[];
  rallies: SavedRally[];
}

/** One row of the video picker: one prepared copy. */
export interface VideoSummary {
  video_id: string;
  /** YouTube title, or the video ID for a video from elsewhere. */
  title: string;
  /** Folder name of the copy, unique within its video. */
  clip: string;
  fps: string;
  first_frame: number;
  frame_count: number;
  mark_count: number;
  rally_count: number;
}

/** One source video offered by the Prepare panel. */
export interface SourceSummary {
  name: string;
  /** YouTube title once any part of the video is prepared; until then the file name. */
  title: string;
  /** Null when ffprobe can't read the file. */
  duration_seconds: number | null;
}

/** What the browser sends on every save. */
export interface SaveRequest {
  /** Revision the browser last loaded or saved. */
  base_revision: number;
  rally_gap_seconds: number;
  marks: Mark[];
}

/** The video's YouTube title, or its ID when it has none. */
export function videoTitle(video: VideoInfo): string {
  return video.source.youtube?.title ?? video.video_id;
}

export function fpsValue(fps: string): number {
  const [numerator, denominator] = fps.split("/").map(Number);
  return numerator / denominator;
}

export function buildLabelFile(
  video: VideoInfo,
  marks: readonly Mark[],
  rallyGapSeconds: number,
  revision: number,
): LabelFile {
  const layout = layOutMarks(marks, fpsValue(video.fps), rallyGapSeconds);
  const rallies: SavedRally[] = layout.rallies.map((rally) => ({
    start_frame: rally.startFrame,
    start_marked: rally.startMarked,
    end_frame: rally.endFrame,
    end_marked: rally.endMarked,
    ended_by: rally.endedBy,
    section_index: rally.sectionIndex,
    contacts: rally.contacts.map((contact) => ({
      frame: contact.frame,
      side: contact.kind as "top" | "bottom",
      unsure: contact.unsure,
    })),
  }));
  const sections: SavedSection[] = layout.sections.map((section) => ({
    start_frame: section.startFrame,
    end_frame: section.endFrame,
  }));
  return {
    schema: LABEL_SCHEMA,
    video_id: video.video_id,
    fps: video.fps,
    frame_count: video.source_frame_count,
    source: video.source,
    revision,
    rally_gap_seconds: rallyGapSeconds,
    marks: [...marks].sort((left, right) => left.frame - right.frame || left.id - right.id),
    sections,
    rallies,
  };
}

/** Throw unless `value` is a label file for `video` with valid marks. */
export function checkLabelFile(value: unknown, video: VideoInfo): LabelFile {
  const file = value as LabelFile;
  if (file?.schema !== LABEL_SCHEMA) throw new Error(`label schema must be ${LABEL_SCHEMA}`);
  if (file.video_id !== video.video_id) throw new Error(`labels are for ${file.video_id}, not ${video.video_id}`);
  if (file.fps !== video.fps || file.frame_count !== video.source_frame_count) {
    throw new Error("label frame rate or frame count differs from the video");
  }
  if (file.source?.sha256 !== video.source.sha256) throw new Error("labels were made from a different source file");
  checkRevision(file.revision);
  checkGap(file.rally_gap_seconds);
  checkMarks(file.marks, video);
  return file;
}

export function checkSaveRequest(value: unknown, video: VideoInfo): SaveRequest {
  const request = value as SaveRequest;
  checkRevision(request?.base_revision);
  checkGap(request.rally_gap_seconds);
  checkMarks(request.marks, video);
  return request;
}

function checkRevision(revision: unknown): void {
  if (!Number.isInteger(revision) || (revision as number) < 0) throw new Error("revision must be a whole number");
}

function checkGap(gap: unknown): void {
  if (!(typeof gap === "number" && gap > 0)) throw new Error("rally_gap_seconds must be a positive number");
}

function checkMarks(marks: unknown, video: VideoInfo): void {
  if (!Array.isArray(marks)) throw new Error("marks must be a list");
  const seenIds = new Set<number>();
  for (const mark of marks as Mark[]) {
    if (!Number.isInteger(mark.id) || seenIds.has(mark.id)) throw new Error(`mark id ${mark.id} is invalid or repeated`);
    seenIds.add(mark.id);
    if (!MARK_KINDS.includes(mark.kind as MarkKind)) throw new Error(`unknown mark kind ${mark.kind}`);
    if (!Number.isInteger(mark.frame) || mark.frame < 0 || mark.frame >= video.source_frame_count) {
      throw new Error(`mark frame ${mark.frame} is outside the video`);
    }
    if (typeof mark.unsure !== "boolean") throw new Error(`mark ${mark.id} needs a true/false unsure flag`);
  }
  // One mark of each kind per frame, and one contact per frame whichever the side.
  const seenPlaces = new Set<string>();
  for (const mark of marks as Mark[]) {
    const what = isContact(mark.kind) ? "contact" : mark.kind;
    const place = `${what}@${mark.frame}`;
    if (seenPlaces.has(place)) throw new Error(`two ${what} marks on frame ${mark.frame}`);
    seenPlaces.add(place);
  }
}
