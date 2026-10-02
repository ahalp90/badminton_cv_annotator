#!/usr/bin/env bash
set -eu

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
python=${ANNOTATOR_PYTHON:-$HOME/.venvs/venv-annotator/bin/python}

if [[ ! -x $python ]]; then
    printf 'Annotator Python is not executable: %s\nSet ANNOTATOR_PYTHON to the Python 3.12 annotator environment.\n' \
        "$python" >&2
    exit 2
fi

cd "$repo_root"
PYTHONPATH=.:src "$python" -m experiments.annotator.good_court_refit.runner "$@"
