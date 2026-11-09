/**
 * Dev-server routes for the rally labeller.
 *
 * GET  /api/videos               prepared copies and how far each video is labelled
 * GET  /api/clips/:id/:clip      one copy's details, the video's labels and suggestions
 * PUT  /api/labels/:id           save marks; rallies are rebuilt here before writing
 * GET  /api/sources              source videos that can be prepared, with their lengths
 * POST /api/prepare              start preparing part of a source video
 * GET  /api/prepare              progress of the current or last preparation
 * POST /api/suggestions-done/:id mark a suggested contact's frame done or not done
 * GET  /media/:id/:clip/...      browser copies, with range requests for seeking
 */

import { execFile, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import zlib from "node:zlib";
import type { IncomingMessage, ServerResponse } from "node:http";
import sirv from "sirv";
import type { Plugin } from "vite";

import {
  VIDEO_INFO_SCHEMA,
  buildLabelFile,
  checkLabelFile,
  checkSaveRequest,
  checkSuggestionFile,
  videoTitle,
  type LabelFile,
  type SaveRequest,
  type SourceSummary,
  type SuggestionFile,
  type VideoInfo,
  type VideoSummary,
} from "../src/labelFile.ts";

export interface LabellerFolders {
  /** One folder per video, holding one subfolder per prepared copy. */
  mediaRoot: string;
  /** One `<video_id>.json.gz` label file per video. */
  labelsRoot: string;
  /** Source videos offered for preparation. */
  sourcesRoot: string;
  /** Repository root, for running prepare_video.py. */
  repoRoot: string;
}

export interface PrepareJob {
  running: boolean;
  source: string;
  /** Last lines of the script's output, without ffmpeg's progress lines. */
  log: string[];
  exitCode: number | null;
  /** Frames encoded so far and frames wanted, once the script has said. */
  framesDone: number;
  framesWanted: number | null;
}

// IDs and names become folder and file names, so allow only plain names.
const PLAIN_NAME = /^[A-Za-z0-9_-][A-Za-z0-9_.-]*$/;
const TIME_TEXT = /^\d+(\.\d+)?(:\d{1,2}(\.\d+)?){0,2}$/;
const VIDEO_EXTENSIONS = new Set([".mp4", ".mkv", ".mov", ".webm", ".avi"]);
const LOG_LINES_KEPT = 12;
// prepare_video.py announces "<id>: encoding frames A-B (N frames at ...)"; ffmpeg then reports "frame= 341 ...".
const FRAMES_WANTED_LINE = /\((\d+) frames at /;
const FFMPEG_PROGRESS_LINE = /^frame=\s*(\d+)/;
const execFileAsync = promisify(execFile);

class RequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function readJsonGz(file: string): unknown {
  return JSON.parse(zlib.gunzipSync(fs.readFileSync(file)).toString("utf8"));
}

function writeJsonGz(file: string, value: unknown): void {
  // Write then rename, so a crash mid-write never leaves a half-written label file.
  const partial = `${file}.part`;
  fs.writeFileSync(partial, zlib.gzipSync(JSON.stringify(value, null, 1)));
  fs.renameSync(partial, file);
}

function checkName(name: string | undefined, what: string): string {
  if (name === undefined || !PLAIN_NAME.test(name)) throw new RequestError(400, `invalid ${what}: ${name}`);
  return name;
}

function readClipInfo(folders: LabellerFolders, videoId: string, clip: string): VideoInfo {
  const infoPath = path.join(folders.mediaRoot, videoId, clip, "video.json.gz");
  if (!fs.existsSync(infoPath)) throw new RequestError(404, `no prepared copy ${videoId}/${clip}`);
  const info = readJsonGz(infoPath) as VideoInfo;
  if (info.schema !== VIDEO_INFO_SCHEMA) throw new Error(`${infoPath} is not a ${VIDEO_INFO_SCHEMA} file`);
  return info;
}

function clipNames(folders: LabellerFolders, videoId: string): string[] {
  const videoDir = path.join(folders.mediaRoot, videoId);
  return fs
    .readdirSync(videoDir)
    .filter((name) => fs.existsSync(path.join(videoDir, name, "video.json.gz")))
    .sort();
}

/** Details shared by every copy of a video, taken from its first copy. */
function readAnyClipInfo(folders: LabellerFolders, videoId: string): VideoInfo {
  if (!fs.existsSync(path.join(folders.mediaRoot, videoId))) throw new RequestError(404, `no prepared video ${videoId}`);
  const [firstClip] = clipNames(folders, videoId);
  if (firstClip === undefined) throw new RequestError(404, `no prepared copy of ${videoId}`);
  return readClipInfo(folders, videoId, firstClip);
}

function readLabels(folders: LabellerFolders, video: VideoInfo): LabelFile | null {
  const labelPath = path.join(folders.labelsRoot, `${video.video_id}.json.gz`);
  return fs.existsSync(labelPath) ? checkLabelFile(readJsonGz(labelPath), video) : null;
}

/**
 * Frames of suggested contacts the labeller has dealt with, kept beside the video's
 * suggestions rather than in the label file, because they record review progress, not labels.
 */
function doneFramesPath(folders: LabellerFolders, videoId: string): string {
  return path.join(folders.mediaRoot, videoId, "suggestions_done.json");
}

function readDoneFrames(folders: LabellerFolders, videoId: string): number[] {
  const file = doneFramesPath(folders, videoId);
  if (!fs.existsSync(file)) return [];
  return (JSON.parse(fs.readFileSync(file, "utf8")) as { frames: number[] }).frames;
}

/** Change one frame and return them all; one frame per request, so two devices can't overwrite each other's ticks. */
function setSuggestionDone(folders: LabellerFolders, videoId: string, body: unknown): number[] {
  if (!fs.existsSync(path.join(folders.mediaRoot, videoId))) throw new RequestError(404, `no prepared video ${videoId}`);
  const request = body as { frame?: unknown; done?: unknown };
  if (!Number.isInteger(request.frame) || typeof request.done !== "boolean") {
    throw new RequestError(400, "send a whole-number frame and done: true or false");
  }
  const frames = new Set(readDoneFrames(folders, videoId));
  if (request.done) frames.add(request.frame as number);
  else frames.delete(request.frame as number);
  const sorted = [...frames].sort((left, right) => left - right);
  fs.writeFileSync(doneFramesPath(folders, videoId), `${JSON.stringify({ frames: sorted })}\n`);
  return sorted;
}

function readSuggestions(folders: LabellerFolders, video: VideoInfo): SuggestionFile[] {
  const folder = path.join(folders.mediaRoot, video.video_id, "suggestions");
  if (!fs.existsSync(folder)) return [];
  const names = fs.readdirSync(folder).filter((name) => name.endsWith(".json.gz")).sort();
  return names.map((name) => checkSuggestionFile(readJsonGz(path.join(folder, name)), video));
}

function listVideos(folders: LabellerFolders): VideoSummary[] {
  if (!fs.existsSync(folders.mediaRoot)) return [];
  const summaries: VideoSummary[] = [];
  for (const entry of fs.readdirSync(folders.mediaRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const clips = clipNames(folders, entry.name);
    if (clips.length === 0) continue;
    const labels = readLabels(folders, readClipInfo(folders, entry.name, clips[0]));
    for (const clip of clips) {
      const info = readClipInfo(folders, entry.name, clip);
      // Counts cover this part only, so each row shows how far that part is labelled.
      const inPart = (frame: number) => frame >= info.first_frame && frame < info.first_frame + info.frame_count;
      summaries.push({
        video_id: info.video_id,
        title: videoTitle(info),
        clip,
        fps: info.fps,
        first_frame: info.first_frame,
        frame_count: info.frame_count,
        mark_count: labels?.marks.filter((mark) => inPart(mark.frame)).length ?? 0,
        rally_count: labels?.rallies.filter((rally) => inPart(rally.start_frame)).length ?? 0,
      });
    }
  }
  return summaries.sort((left, right) => left.title.localeCompare(right.title) || left.clip.localeCompare(right.clip));
}

function saveLabels(folders: LabellerFolders, videoId: string, body: unknown): LabelFile {
  const video = readAnyClipInfo(folders, videoId);
  let request: SaveRequest;
  try {
    request = checkSaveRequest(body, video);
  } catch (error) {
    throw new RequestError(400, (error as Error).message);
  }
  const savedRevision = readLabels(folders, video)?.revision ?? 0;
  if (request.base_revision !== savedRevision) {
    throw new RequestError(
      409,
      `these labels were saved from somewhere else (revision ${savedRevision}, this page has ${request.base_revision})`,
    );
  }
  const file = buildLabelFile(video, request.marks, request.rally_gap_seconds, savedRevision + 1);
  fs.mkdirSync(folders.labelsRoot, { recursive: true });
  writeJsonGz(path.join(folders.labelsRoot, `${videoId}.json.gz`), file);
  return file;
}

function listSources(folders: LabellerFolders): string[] {
  if (!fs.existsSync(folders.sourcesRoot)) return [];
  return fs
    .readdirSync(folders.sourcesRoot)
    .filter((name) => VIDEO_EXTENSIONS.has(path.extname(name).toLowerCase()))
    .sort();
}

async function sourceDuration(file: string): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", file]);
    return Number(stdout.trim());
  } catch {
    // A file ffprobe can't read is still listed; preparing it reports the real error.
    return null;
  }
}

async function listSourceSummaries(folders: LabellerFolders): Promise<SourceSummary[]> {
  const names = listSources(folders);
  const durations = await Promise.all(names.map((name) => sourceDuration(path.join(folders.sourcesRoot, name))));
  return names.map((name, index) => ({
    name,
    title: preparedTitle(folders, path.parse(name).name) ?? name,
    duration_seconds: durations[index],
  }));
}

/** The title recorded when a part of this video was prepared, or null before its first part. */
function preparedTitle(folders: LabellerFolders, videoId: string): string | null {
  if (!fs.existsSync(path.join(folders.mediaRoot, videoId))) return null;
  const clips = clipNames(folders, videoId);
  if (clips.length === 0) return null;
  return videoTitle(readClipInfo(folders, videoId, clips[0]));
}

let prepareJob: PrepareJob = { running: false, source: "", log: [], exitCode: null, framesDone: 0, framesWanted: null };

function startPrepare(folders: LabellerFolders, body: unknown): PrepareJob {
  if (prepareJob.running) throw new RequestError(409, `already preparing ${prepareJob.source}`);
  const request = body as { source?: string; start?: string; end?: string };
  const source = request.source ?? "";
  if (!listSources(folders).includes(source)) throw new RequestError(400, `unknown source video: ${source}`);
  const script = path.join(folders.repoRoot, "tools/rally_labeller/prepare_video.py");
  const scriptArguments = [script, path.join(folders.sourcesRoot, source), "--out-root", folders.mediaRoot];
  for (const [flag, value] of [
    ["--start", request.start],
    ["--end", request.end],
  ] as const) {
    if (value === undefined || value.trim() === "") continue;
    if (!TIME_TEXT.test(value.trim())) throw new RequestError(400, `times look like 95, 1:35 or 1:01:35, not ${value}`);
    scriptArguments.push(flag, value.trim());
  }

  const job: PrepareJob = { running: true, source, log: [], exitCode: null, framesDone: 0, framesWanted: null };
  prepareJob = job;
  const child = spawn("python3", scriptArguments, {
    cwd: folders.repoRoot,
    // Unbuffered, so the frame count it prints reaches the progress display straight away.
    env: { ...process.env, PYTHONPATH: path.join(folders.repoRoot, "src"), PYTHONUNBUFFERED: "1" },
  });
  const keepLines = (chunk: Buffer) => {
    // ffmpeg rewrites its progress line with carriage returns.
    const lines = chunk
      .toString("utf8")
      .split(/[\r\n]+/)
      .map((line) => line.trim())
      .filter((line) => line !== "");
    const messages: string[] = [];
    for (const line of lines) {
      const progress = FFMPEG_PROGRESS_LINE.exec(line);
      if (progress !== null) {
        job.framesDone = Number(progress[1]);
        continue;
      }
      const wanted = FRAMES_WANTED_LINE.exec(line);
      if (wanted !== null) job.framesWanted = Number(wanted[1]);
      messages.push(line);
    }
    job.log = [...job.log, ...messages].slice(-LOG_LINES_KEPT);
  };
  child.stdout.on("data", keepLines);
  child.stderr.on("data", keepLines);
  child.on("error", (error) => {
    job.log = [...job.log, `could not start python3: ${error.message}`];
    job.running = false;
  });
  child.on("close", (code) => {
    job.exitCode = code;
    job.running = false;
  });
  return job;
}

async function readBody(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(chunk as Buffer);
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(value));
}

async function handleApi(folders: LabellerFolders, request: IncomingMessage, response: ServerResponse): Promise<void> {
  const [resource, first, second] = (request.url ?? "").split("?")[0].split("/").filter(Boolean).map(decodeURIComponent);
  const route = `${request.method} ${resource}`;
  if (route === "GET videos") {
    sendJson(response, 200, listVideos(folders));
  } else if (route === "GET clips") {
    const video = readClipInfo(folders, checkName(first, "video ID"), checkName(second, "copy name"));
    sendJson(response, 200, {
      video,
      labels: readLabels(folders, video),
      suggestions: readSuggestions(folders, video),
      suggestions_done: readDoneFrames(folders, video.video_id),
    });
  } else if (route === "PUT labels") {
    const file = saveLabels(folders, checkName(first, "video ID"), await readBody(request));
    sendJson(response, 200, { revision: file.revision });
  } else if (route === "GET sources") {
    sendJson(response, 200, await listSourceSummaries(folders));
  } else if (route === "POST prepare") {
    sendJson(response, 202, startPrepare(folders, await readBody(request)));
  } else if (route === "GET prepare") {
    sendJson(response, 200, prepareJob);
  } else if (route === "POST suggestions-done") {
    sendJson(response, 200, setSuggestionDone(folders, checkName(first, "video ID"), await readBody(request)));
  } else {
    throw new RequestError(404, `no route for ${request.method} ${request.url}`);
  }
}

export function labellerApi(folders: LabellerFolders): Plugin {
  return {
    name: "rally-labeller-api",
    configureServer(server) {
      server.middlewares.use("/media", sirv(folders.mediaRoot, { dev: true }));
      server.middlewares.use("/api", (request, response) => {
        handleApi(folders, request, response).catch((error: unknown) => {
          const status = error instanceof RequestError ? error.status : error instanceof SyntaxError ? 400 : 500;
          const message = error instanceof Error ? error.message : String(error);
          sendJson(response, status, { error: message });
        });
      });
    },
  };
}
