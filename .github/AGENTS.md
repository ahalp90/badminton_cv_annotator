## Compression
- Store *.npy as *.npy.xz
  - `with lzma.open` for dump/load; `format=lzma.FORMAT_XZ, preset=9`
- Store {*.json,*.csv} as {*.json.gz,*.csv.gz}