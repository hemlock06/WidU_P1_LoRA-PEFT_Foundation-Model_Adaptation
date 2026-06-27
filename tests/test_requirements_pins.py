# -*- coding: utf-8 -*-
"""의존성 핀 파일 검증 (HANDOFF_ISSUES P1-1).

torch 등 무거운 런타임 의존성 없이 실행된다 — requirements.txt / environment.yml의
형식 유효성과 '문서로 검증된' 버전 핀이 정확히 고정됐는지만 단정한다.
(스크립트 import 가 아니라 파일 파싱이므로 stage0_spine 의 torch import 와 무관.)
"""
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements.txt"
ENV = ROOT / "environment.yml"

# README.md / REPRODUCIBILITY.md §1 에 명시되어 핀 고정한 버전(검증된 사실).
DOC_PINS = {
    "torch": "==2.1.2+cu118",
    "wfdb": "==4.3.1",
}


def _parse_requirements(text: str):
    """requirements.txt → [Requirement]. 주석·옵션(--...)·빈 줄은 제외,
    pip 규칙대로 공백 뒤 인라인 주석은 절단."""
    reqs = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # 공백으로 구분된 인라인 주석 제거 (pip 규칙)
        for sep in (" #", "\t#"):
            if sep in line:
                line = line.split(sep, 1)[0].strip()
        if line:
            reqs.append(Requirement(line))
    return reqs


def test_requirements_file_exists():
    assert REQ.is_file(), "requirements.txt 가 레포 루트에 있어야 한다 (P1-1)"


def test_every_requirement_line_parses():
    """모든 의존성 줄이 PEP 508 로 파싱돼야 한다(형식 깨짐 방지)."""
    reqs = _parse_requirements(REQ.read_text(encoding="utf-8"))
    assert reqs, "최소 1개 이상의 의존성이 있어야 한다"
    # Requirement() 생성 자체가 파싱 검증 — 예외 없이 여기 도달하면 통과.
    names = {r.name.lower() for r in reqs}
    # 핵심 런타임 의존성 누락 방지(실제 import 기준).
    for must in ("torch", "wfdb", "numpy", "scipy", "scikit-learn", "h5py", "pandas"):
        assert must in names, f"requirements.txt 에 {must} 누락"


def test_documented_versions_are_pinned_exactly():
    """문서로 검증된 버전(torch·wfdb)은 정확히 == 로 고정돼야 한다."""
    by_name = {r.name.lower(): r for r in _parse_requirements(REQ.read_text(encoding="utf-8"))}
    for name, spec in DOC_PINS.items():
        assert name in by_name, f"{name} 핀 누락"
        assert str(by_name[name].specifier) == spec, (
            f"{name} 는 '{spec}' 로 고정돼야 한다 (문서 검증값). "
            f"실제: '{by_name[name].specifier}'"
        )


def test_environment_yml_pins_python_39():
    assert ENV.is_file(), "environment.yml 가 있어야 한다"
    text = ENV.read_text(encoding="utf-8")
    assert "python=3.9" in text, "environment.yml 는 python=3.9 를 고정해야 한다(문서 검증값)"
