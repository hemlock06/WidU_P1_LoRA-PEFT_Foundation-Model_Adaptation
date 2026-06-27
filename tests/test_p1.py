# -*- coding: utf-8 -*-
"""P1 순수 유틸 스모크 테스트 (ECG-FM·데이터 비의존, 고정값).

stage0_spine 의 단일리드 마스킹·민감도/특이도 계산 같은 순수 함수만 검증한다.
(백본 추론·외부 DB가 필요한 평가는 별도 스크립트로 수행.)
"""
import numpy as np

import stage0_spine as s0


def test_sens_spec_known_values():
    # preds@0.5 = [1,0,0,1] vs y=[1,1,0,0] → tp1 fn1 tn1 fp1
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.8])
    sens, spec, far = s0.sens_spec(y, p, 0.5)
    assert abs(sens - 0.5) < 1e-9
    assert abs(spec - 0.5) < 1e-9
    assert abs(far - 0.5) < 1e-9


def test_single_lead_keeps_only_target_lead():
    sig = np.ones((12, 100), dtype=np.float32)
    out = s0.single_lead(sig)
    # 대상 리드(slot LEAD)만 보존, 나머지는 0-fill
    assert out[s0.LEAD].sum() == 100.0
    others = np.ones(12, dtype=bool)
    others[s0.LEAD] = False
    assert out[others].sum() == 0.0
