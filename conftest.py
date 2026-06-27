# -*- coding: utf-8 -*-
"""pytest 부트스트랩 — scripts/ 를 import 경로에 추가(P1은 패키지가 아닌 스크립트 모음)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
