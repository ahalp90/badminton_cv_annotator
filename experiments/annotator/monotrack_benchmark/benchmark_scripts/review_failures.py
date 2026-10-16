"""Make labelled review clips for MonoTrack inference failures."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from common import load_config, load_manifest, read_json, write_json


def subtitle_time(seconds: float) -> str:
    milliseconds = round(max(0, seconds) * 1000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'


def make_clip(source: Path, destination: Path, span: dict[str, Any], fps: float) -> None:
    start, end = span['start_frame'], span['end_frame']
    subtitles = []
    for index, (frame, side) in enumerate(zip(span['label_frames'], span['label_sides'], strict=True), start=1):
        before = subtitle_time((max(start, frame - 3) - start) / fps)
        after = subtitle_time((min(end, frame + 4) - start) / fps)
        subtitles.append(f'{index}\n{before} --> {after}\nGT hit: frame {frame}, {side}\n')
    with tempfile.TemporaryDirectory(prefix='monotrack-review-') as directory:
        (Path(directory) / 'labels.srt').write_text('\n'.join(subtitles))
        frame_text = f"drawtext=text='Source frame %{{eif\\:n+{start}\\:d}}':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.7"
        subprocess.run([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
            '-ss', str(start / fps), '-i', str(source.resolve()), '-frames:v', str(end - start),
            '-vf', f'subtitles=labels.srt,{frame_text}', '-an', '-c:v', 'libx264', '-crf', '20',
            '-preset', 'fast', str(destination.resolve()),
        ], cwd=directory, check=True)


def review(config: dict[str, Any], results: Path, output: Path) -> list[dict[str, Any]]:
    items = {item['item_id']: item for item in load_manifest(config['paths']['inputs'])['items']}
    failures: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in sorted(results.glob('mono_*')):
        if not run.is_dir():
            continue
        for identity in items:
            path = run / f'{identity}.json.gz'
            if not path.is_file():
                continue
            for failure in read_json(path)['failed_rallies']:
                if failure['reason'] == 'court_detection':
                    continue
                failures.setdefault((identity, failure['span_id']), []).append({'run': run.name, **failure})
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for (identity, span_id), conditions in sorted(failures.items()):
        item = items[identity]
        span = next(span for span in item['spans'] if span['span_id'] == span_id)
        filename = f'{identity}_{span_id:04d}.mp4'
        make_clip(config['paths']['sources'] / item['source_filename'], output / filename, span, item['fps'])
        records.append({'item_id': identity, 'clip': filename, 'fps': item['fps'], **span, 'failures': conditions})
    write_json(output / 'index.json.gz', records)
    print(f'Created {len(records)} review clips', flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(load_config(args.config), args.results, args.output)


if __name__ == '__main__':
    main()
