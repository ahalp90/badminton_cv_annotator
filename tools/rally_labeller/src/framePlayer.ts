/**
 * Frame-exact control of an HTML video element.
 *
 * prepare_video.py gives copy frame N the timestamp N / fps, so a frame
 * number and a time convert directly. A copy of part of a video starts at
 * source frame `firstFrame`; this class works in source frame numbers. The frame on screen comes from
 * `requestVideoFrameCallback`, which reports the exact timestamp of each
 * frame the browser shows, including while playing fast.
 */

/**
 * The time to seek to for a frame: a quarter of the way into its display time.
 *
 * Chromium shows the frame nearest the requested time, so aiming at the exact
 * middle can land on either neighbour. A quarter in is nearest to the wanted
 * frame and also after its start, which suits browsers that show the latest
 * frame starting at or before the requested time.
 */
export function seekTime(copyFrame: number, fps: number): number {
  return (copyFrame + 0.25) / fps;
}

export class FramePlayer {
  readonly video: HTMLVideoElement;
  readonly fps: number;
  /** Source frame number of the copy's first frame. */
  readonly firstFrame: number;
  /** Frames in the copy. */
  readonly frameCount: number;
  /** The source frame currently on screen. */
  shownFrame = 0;
  /** Set while a seek is under way, so quick repeated steps add up from the target. */
  private targetFrame: number | null = null;
  private readonly onFrameShown: (frame: number) => void;

  constructor(
    video: HTMLVideoElement,
    fps: number,
    firstFrame: number,
    frameCount: number,
    onFrameShown: (frame: number) => void,
  ) {
    if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {
      throw new Error("This browser cannot report video frames; use a current Chrome, Edge or Firefox.");
    }
    this.video = video;
    this.fps = fps;
    this.firstFrame = firstFrame;
    this.frameCount = frameCount;
    this.shownFrame = firstFrame;
    this.onFrameShown = onFrameShown;
    video.requestVideoFrameCallback(this.handleFrame);
    video.addEventListener("seeked", () => {
      // Some browsers skip the frame callback for a seek while paused. The seek
      // aimed just inside the frame, so rounding the time down gives the frame it landed on.
      if (this.targetFrame === null || !video.paused) return;
      this.targetFrame = null;
      this.showFrame(this.firstFrame + Math.floor(video.currentTime * this.fps));
    });
  }

  private readonly handleFrame = (_now: number, metadata: VideoFrameCallbackMetadata): void => {
    // A frame shown mid-seek can still be the old one, so keep the target until the seek ends.
    if (!this.video.seeking) this.targetFrame = null;
    // A shown frame's timestamp sits exactly on frame / fps, so rounding only absorbs float error.
    this.showFrame(this.firstFrame + Math.round(metadata.mediaTime * this.fps));
    this.video.requestVideoFrameCallback(this.handleFrame);
  };

  private showFrame(frame: number): void {
    this.shownFrame = this.clampFrame(frame);
    this.onFrameShown(this.shownFrame);
  }

  /** The frame a mark goes on: the seek target while a step is still loading, else the frame on screen. */
  get currentFrame(): number {
    return this.targetFrame ?? this.shownFrame;
  }

  clampFrame(frame: number): number {
    return Math.min(Math.max(Math.round(frame), this.firstFrame), this.firstFrame + this.frameCount - 1);
  }

  get playing(): boolean {
    return !this.video.paused;
  }

  play(): void {
    this.targetFrame = null;
    void this.video.play();
  }

  pause(): void {
    if (this.video.paused) return;
    this.video.pause();
    // Playback can stop part-way past the last reported frame. Settle on that
    // frame so the picture and the frame number agree.
    this.targetFrame = this.shownFrame;
    this.video.currentTime = seekTime(this.shownFrame - this.firstFrame, this.fps);
  }

  seekToFrame(frame: number): void {
    const target = this.clampFrame(frame);
    if (target === this.currentFrame) return;
    this.targetFrame = target;
    this.video.currentTime = seekTime(target - this.firstFrame, this.fps);
  }

  stepFrames(frames: number): void {
    this.pause();
    this.seekToFrame(this.currentFrame + frames);
  }

  stepSeconds(seconds: number): void {
    this.stepFrames(Math.round(seconds * this.fps));
  }

  get speed(): number {
    return this.video.playbackRate;
  }

  set speed(rate: number) {
    this.video.playbackRate = rate;
  }
}
