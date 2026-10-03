# ShuttleSet22 official annotations

`set/` contains the official match table, homographies and per-match contact
annotations. Each `.csv.gz` decompresses to the original CSV bytes. Values and
corner ordering are unchanged. See [attribution](../ATTRIBUTION.md#shuttleset22-datashuttleset22-mit-annotations-broadcaster-video-rights).

The dataset exporter, calibration tools and court evaluator read these compressed
files directly. Source videos and extracted model arrays are stored separately.
