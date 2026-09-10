"""Rule tests for VFDB episode construction under the official NOISE statement."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "vfdb_episodes", Path(__file__).resolve().parents[1] / "scripts/audit_vfdb_episodes.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_noise_does_not_terminate_previous_rhythm():
    markers = [(0, "N"), (100, "NOISE"), (200, "VT"), (300, "N")]
    spans, noise = MODULE.build_episodes(markers, 400)
    assert spans == [(0, 200, "N"), (200, 300, "VT"), (300, 400, "N")]
    assert noise == [(100, 200)]


def test_samples_before_first_rhythm_marker_are_unlabeled():
    spans, noise = MODULE.build_episodes([(0, "NOISE"), (50, "VT")], 400)
    assert spans == [(0, 50, "UNLABELED_START"), (50, 400, "VT")]
    assert noise == [(0, 50)]


def test_late_first_marker_leaves_unlabeled_prefix():
    spans, _ = MODULE.build_episodes([(18, "N"), (100, "VFL")], 200)
    assert spans == [(0, 18, "UNLABELED_START"), (18, 100, "N"), (100, 200, "VFL")]


def test_same_rhythm_reannounced_after_noise_merges_into_one_episode():
    """The merge step is what realises 'the previous rhythm continues'. It fires 53 times
    on real data, 52 of them this case; none of the other tests reach it."""
    markers = [(0, "N"), (100, "NOISE"), (200, "N"), (300, "VT")]
    spans, noise = MODULE.build_episodes(markers, 400)
    assert spans == [(0, 300, "N"), (300, 400, "VT")]
    assert noise == [(100, 200)]


def test_plain_duplicate_rhythm_markers_also_merge():
    spans, _ = MODULE.build_episodes([(0, "BI"), (50, "BI"), (90, "VT")], 200)
    assert spans == [(0, 90, "BI"), (90, 200, "VT")]


def test_distinct_rhythms_are_never_merged():
    spans, _ = MODULE.build_episodes([(0, "N"), (50, "VT"), (90, "N")], 200)
    assert spans == [(0, 50, "N"), (50, 90, "VT"), (90, 200, "N")]


def test_trailing_noise_marker_extends_to_end_of_record():
    """The official rule is silent here: there is no 'next annotation'. Six of 22 records
    end this way, and the fallback supplies 29.2% of all reported noise seconds."""
    spans, noise = MODULE.build_episodes([(0, "VT"), (300, "NOISE")], 400)
    assert spans == [(0, 400, "VT")]
    assert noise == [(300, 400)]


def test_consecutive_noise_markers_each_end_at_the_next_marker():
    spans, noise = MODULE.build_episodes([(0, "N"), (100, "NOISE"), (150, "NOISE"), (300, "VT")], 400)
    assert spans == [(0, 300, "N"), (300, 400, "VT")]
    assert noise == [(100, 150), (150, 300)]


def test_naive_last_label_parser_would_differ_on_noise():
    """Regression guard: a last-label parser labels the post-noise gap as NOISE."""
    markers = [(0, "N"), (100, "NOISE"), (200, "VT")]
    naive = {(100, 200): "NOISE"}
    spans, _ = MODULE.build_episodes(markers, 300)
    official = {(s, e): label for s, e, label in spans}
    assert official[(0, 200)] == "N"
    assert (100, 200) not in official
    assert naive[(100, 200)] == "NOISE"
