## Minimal implementation
- Before editing, inspect the code and data transformations between the relevant input and requested output
- Prefer boring, readable code.
- Reuse repo code, stdlib/NumPy/scikit-learn, or installed dependencies before writing new code.
- Avoid magic numbers; use named single sources of truth.

## Proportional verification
- This is a student project, not an enterprise build. Avoid verification theatre.
- Use the smallest check that materially validates the result.
  - Prefer direct equality over hashes; use MD5 when checksum portability matters.
- Test the main path and meaningful failures, then stop. Add checks only for a specific, plausible silent failure that could corrupt data or make results untraceable.

## Structure
- Aim to keep modules <1000 LoC and functions <100 LoC.
  - Split only at clear responsibility boundaries. Do not refactor unrelated code just to meet these limits.
- Prefer self-documenting code. Comment non-obvious intent or constraints. Avoid comments longer than the code they explain.

## Compression
- Store *.npy as *.npy.xz
  - `with lzma.open` for dump/load; `format=lzma.FORMAT_XZ, preset=9`
- Store {*.json,*.csv} as {*.json.gz,*.csv.gz}