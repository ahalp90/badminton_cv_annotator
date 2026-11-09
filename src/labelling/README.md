# Manual footage labelling

Use `python -m labelling.scene_editor --help` to label scene intervals as live,
replay, cutaway or other footage. Use `python -m labelling.rally_start_editor
--help` to review prepared rally-start cases and mark a visible serve, broadcast
return or first visible rally frame.

Both tools use OpenCV windows and need a GUI-capable OpenCV environment.
The rally-start editor expects prepared target rows; it is not a general rally
or contact editor. Neither tool currently provides the complete amateur-footage
workflow of rally start/stop, all contact times and top/bottom attribution.

- `scene_editor.py` and `scene_labels.py`: scene editor and saved interval format.
- `rally_start_editor.py` and `rally_starts.py`: rally-start editor, saved decisions
  and undo state.
- `gui.py`: the small display helpers used by both editors.

See the [scene-labelling quickstart](../../docs/scraper_pipeline/broadcast_nonstandard_camera_id/manual_broadcast_timeline_annotation_quickstart.md)
for controls and input preparation. Court-corner labelling remains in
[`scripts/court_annotation`](../../scripts/court_annotation/).
