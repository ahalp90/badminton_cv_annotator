/**
 * A magnifier over the video: a box beside the pointer showing the area
 * around it enlarged, with a small cross on the exact spot. On a desktop a
 * right-click turns it on and off; on a phone it shows while a finger is held
 * still on the picture and then dragged.
 *
 * The box enlarges pixels from the browser copy of the video, so it adds no
 * detail. The optional sharpening works on those pixels before enlarging, so
 * edges come out crisper rather than the enlarged blur being sharpened.
 */

export interface LoupeOptions {
  zoom: number;
  /** Width and height of the box, in CSS pixels. */
  sizePx: number;
  sharpen: boolean;
}

export const LOUPE_DEFAULTS: LoupeOptions = { zoom: 3, sizePx: 220, sharpen: false };
const MIN_ZOOM = 1.5;
const MAX_ZOOM = 8;
const MIN_SIZE_PX = 120;
const MAX_SIZE_PX = 480;
const GAP_PX = 24;
const CROSS_PX = 6;
// How far each colour value moves away from the average of its four neighbours.
const SHARPEN_AMOUNT = 0.6;

interface LoupeState {
  video: HTMLVideoElement;
  clientX: number;
  clientY: number;
}

let state: LoupeState | null = null;
let options: LoupeOptions = { ...LOUPE_DEFAULTS };
/** The video's own pixels under the pointer, before enlarging. */
let nativeCanvas: HTMLCanvasElement | null = null;

function loupeCanvas(): HTMLCanvasElement {
  const canvas = document.getElementById("loupe");
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error("page is missing the #loupe canvas");
  return canvas;
}

/** Apply new options, kept within their limits; returns the options in use. */
export function setLoupeOptions(wanted: LoupeOptions): LoupeOptions {
  options = {
    zoom: Math.min(Math.max(wanted.zoom, MIN_ZOOM), MAX_ZOOM),
    sizePx: Math.round(Math.min(Math.max(wanted.sizePx, MIN_SIZE_PX), MAX_SIZE_PX)),
    sharpen: wanted.sharpen,
  };
  redrawLoupe();
  return options;
}

export function loupeShown(): boolean {
  return state !== null;
}

export function showLoupe(video: HTMLVideoElement, clientX: number, clientY: number): void {
  state = { video, clientX, clientY };
  loupeCanvas().hidden = false;
  redrawLoupe();
}

export function moveLoupe(clientX: number, clientY: number): void {
  if (state === null) return;
  state.clientX = clientX;
  state.clientY = clientY;
  redrawLoupe();
}

export function hideLoupe(): void {
  state = null;
  loupeCanvas().hidden = true;
}

/** Zoom in (direction 1) or out (-1) by a quarter; returns the new zoom. */
export function changeLoupeZoom(direction: 1 | -1): number {
  return setLoupeOptions({ ...options, zoom: options.zoom * (direction > 0 ? 1.25 : 0.8) }).zoom;
}

function sharpenPixels(context: CanvasRenderingContext2D, width: number, height: number): void {
  const image = context.getImageData(0, 0, width, height);
  const source = image.data;
  const sharpened = new Uint8ClampedArray(source);
  const rowBytes = width * 4;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const pixel = y * rowBytes + x * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const at = pixel + channel;
        const neighbours = source[at - 4] + source[at + 4] + source[at - rowBytes] + source[at + rowBytes];
        sharpened[at] = source[at] + SHARPEN_AMOUNT * (4 * source[at] - neighbours);
      }
    }
  }
  context.putImageData(new ImageData(sharpened, width, height), 0, 0);
}

/** Draw the current frame into the box; call whenever the shown frame or the pointer changes. */
export function redrawLoupe(): void {
  if (state === null) return;
  const { video, clientX, clientY } = state;
  const { zoom, sizePx, sharpen } = options;
  const canvas = loupeCanvas();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(sizePx * ratio);
  canvas.height = Math.round(sizePx * ratio);
  canvas.style.width = `${sizePx}px`;
  canvas.style.height = `${sizePx}px`;
  const context = canvas.getContext("2d");
  if (context === null) throw new Error("canvas 2D drawing is unavailable");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.fillStyle = "#000000";
  context.fillRect(0, 0, sizePx, sizePx);

  // Screen pixels per video pixel, and where the picture sits inside any black bars.
  const box = video.getBoundingClientRect();
  const scale = Math.min(box.width / video.videoWidth, box.height / video.videoHeight);
  const pictureLeft = box.left + (box.width - video.videoWidth * scale) / 2;
  const pictureTop = box.top + (box.height - video.videoHeight * scale) / 2;
  const videoX = (clientX - pictureLeft) / scale;
  const videoY = (clientY - pictureTop) / scale;
  // The square of video pixels that fills the box at this zoom, centred on the pointer.
  const side = Math.max(1, Math.round(sizePx / (scale * zoom)));
  nativeCanvas ??= document.createElement("canvas");
  nativeCanvas.width = side;
  nativeCanvas.height = side;
  const native = nativeCanvas.getContext("2d", { willReadFrequently: true });
  if (native === null) throw new Error("canvas 2D drawing is unavailable");
  const sourceLeft = Math.round(videoX - side / 2);
  const sourceTop = Math.round(videoY - side / 2);
  native.drawImage(video, sourceLeft, sourceTop, side, side, 0, 0, side, side);
  if (sharpen) sharpenPixels(native, side, side);
  context.drawImage(nativeCanvas, 0, 0, side, side, 0, 0, sizePx, sizePx);

  const centre = sizePx / 2;
  context.strokeStyle = "rgba(255, 255, 255, 0.8)";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(centre - CROSS_PX, centre);
  context.lineTo(centre + CROSS_PX, centre);
  context.moveTo(centre, centre - CROSS_PX);
  context.lineTo(centre, centre + CROSS_PX);
  context.stroke();
  context.fillStyle = "rgba(255, 255, 255, 0.8)";
  context.font = "12px system-ui, sans-serif";
  context.fillText(`${zoom.toFixed(1)}×${sharpen ? " · sharpened" : ""}`, 6, sizePx - 6);

  // Beside the pointer so a finger or cursor doesn't cover the box: above, then below,
  // then level with it, on whichever side fits; kept on screen in every case.
  const right = clientX + GAP_PX;
  const left = clientX - GAP_PX - sizePx;
  const above = clientY - GAP_PX - sizePx;
  const below = clientY + GAP_PX;
  const level = clientY - sizePx / 2;
  const places = [
    [right, above],
    [left, above],
    [right, below],
    [left, below],
    [right, level],
    [left, level],
  ];
  const fits = ([x, y]: number[]) => x >= 0 && y >= 0 && x + sizePx <= window.innerWidth && y + sizePx <= window.innerHeight;
  const [chosenX, chosenY] = places.find(fits) ?? (clientX < window.innerWidth / 2 ? [right, level] : [left, level]);
  const keepOnScreen = (value: number, room: number) => Math.min(Math.max(value, 0), Math.max(room - sizePx, 0));
  canvas.style.left = `${keepOnScreen(chosenX, window.innerWidth)}px`;
  canvas.style.top = `${keepOnScreen(chosenY, window.innerHeight)}px`;
}
