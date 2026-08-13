#!/usr/bin/env bash
# Run the dataset-builder contract suite and its affected shared boundaries.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

exec uv run --extra dev python -m pytest -q \
    tests/test_dataset_builder_cli.py \
    tests/test_dataset_builder_manifest.py \
    tests/test_dataset_builder_pose_sharding.py \
    tests/test_dataset_builder_records.py \
    tests/test_dataset_builder_selection.py \
    tests/test_dataset_builder_shuttle_evidence.py \
    tests/test_dataset_builder_tracknet_input.py \
    tests/test_dataset_builder_vision.py \
    tests/test_inpaint_guard.py \
    tests/test_scraper_download_videos.py \
    tests/test_validation_overlay_assembly.py
