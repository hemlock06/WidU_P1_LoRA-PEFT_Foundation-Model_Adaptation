"""Schema and rule tests for the external-cohort lineage audits (no downloads)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PTBXL = load("audit_ptbxl_cross_version")
INCART = load("audit_incart_annotations")


def test_changelog_parser_maps_every_dropped_id_to_one_kept_partner():
    text = (
        "#### Change 1\n->drop row with ecg_id=137 [strat_fold=8]\n"
        "->keep row with ecg_id=138 [strat_fold=8]\n"
        "#### Change 2\n->drop row with ecg_id=3800 [strat_fold=9]\n"
        "->drop row with ecg_id=3801 [strat_fold=6]\n->keep row with ecg_id=3802 [strat_fold=6]\n"
    )
    assert PTBXL.parse_changelog(text) == {137: 138, 3800: 3802, 3801: 3802}


def test_changelog_parser_rejects_ambiguous_keep_block():
    text = "#### Change 1\n->drop row with ecg_id=1 [f]\n->keep row with ecg_id=2 [f]\n->keep row with ecg_id=3 [f]\n"
    with pytest.raises(ValueError):
        PTBXL.parse_changelog(text)


def test_records_list_parser_tolerates_missing_newline_between_blocks():
    text = "records100/21000/21837_lrrecords500/00000/00001_hr\nrecords500/21000/21837_hr\n"
    assert PTBXL.records500_ids(text) == {1, 21837}


def test_challenge_folder_rule_matches_published_paths():
    assert PTBXL.challenge_url(9205, "mat").endswith("/g10/HR09205.mat")
    assert PTBXL.challenge_url(18943, "hea").endswith("/g19/HR18943.hea")
    assert PTBXL.challenge_url(1000, "hea").endswith("/g2/HR01000.hea")


def test_incart_rhythm_marker_persists_until_next_marker():
    ann = SimpleNamespace(sample=np.array([10, 100, 250]), symbol=["+", "N", "+"],
                          aux_note=["(AFIB\x00", "", "(N\x00"])
    assert INCART.rhythm_intervals(ann, 400) == [(10, 250, "(AFIB"), (250, 400, "(N")]


def test_incart_window_index_at_257hz_equals_historical_500hz_windowing():
    samples = np.array([0, 2569, 2570, 462599])
    historical = ((samples.astype(float) * (INCART.FS_OUT / INCART.FS)) / INCART.SEG_LEN_OUT).astype(int)
    direct = samples // INCART.WINDOW_SAMPLES_257
    assert historical.tolist() == direct.tolist() == [0, 0, 1, 179]


def test_incart_patient_file_parser_handles_blank_diagnosis():
    text = "patient 4\nI08\n\npatient 5\nI09 I10 I11\n\npatient 6\nI12\nAcute MI\n"
    parsed = INCART.parse_patients(text)
    assert parsed[4] == {"records": ["I08"], "diagnosis": ""}
    assert parsed[5]["records"] == ["I09", "I10", "I11"]
    assert parsed[6]["diagnosis"] == "Acute MI"
