"""Canonical contact matching retains label alignment and deterministic ties."""

import gzip
import json
from pathlib import Path

from annotator.training.sequences import match_contacts

REFERENCE_DIR = Path(__file__).parent / 'fixtures' / 'annotator_reference'


def test_matching_preserves_tie_and_duplicate_rules() -> None:
    cases = json.loads(gzip.decompress((REFERENCE_DIR / 'contact_matching.json.gz').read_bytes()))
    for case in cases:
        expected = [tuple(match) for match in case['matches']]
        assert match_contacts(case['labels'], case['predictions'], case['tolerance']) == expected
