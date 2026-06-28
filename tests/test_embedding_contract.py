# -*- coding: utf-8 -*-
"""embedding[768] 출력 계약 회귀 테스트 (HANDOFF_ISSUES P0-3).

infer() / to_record() 가 P2 융합 입력인 mean-pool 임베딩(768)을 노출하는지 단정한다.
백본 추론이 필요하므로 torch + ECG-FM/P1 체크포인트가 있어야 실행된다.
없으면 skip — 검증 환경(Python 3.9 + torch 2.1.2 + 체크포인트)에서 실행할 것.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 미설치 — 검증 환경에서 실행")

# 체크포인트 경로(환경변수 우선, 기본값). 없으면 skip.
import p1_cardiac_channel as pcc  # noqa: E402

_CKPTS = [pcc.CKPT_FM, pcc.CKPT_P1]
_have_ckpts = all(os.path.exists(p) for p in _CKPTS)
pytestmark = pytest.mark.skipif(
    not _have_ckpts, reason=f"체크포인트 없음(재다운로드 필요): {_CKPTS}"
)

EMB_DIM = 768


@pytest.fixture(scope="module")
def channel():
    return pcc.P1CardiacChannel(device="cpu")


def test_infer_single_exposes_embedding(channel):
    sig = np.zeros((12, 5000), dtype=np.float32)
    out = channel.infer(sig)
    assert "embedding" in out, "infer() 는 embedding 키를 반환해야 한다(P0-3)"
    assert out["embedding"].shape == (EMB_DIM,), "단일 입력 → embedding (768,)"
    # 기존 계약 키 불변(회귀 방지)
    for k in ("emergency_score", "cardiac_probs", "benign_flag"):
        assert k in out


def test_infer_batch_embedding_shape(channel):
    sig = np.zeros((3, 12, 5000), dtype=np.float32)
    out = channel.infer(sig)
    assert out["embedding"].shape == (3, EMB_DIM), "배치 N → embedding (N,768)"

# 주의: to_record() 영속화(parquet 레코드에 embedding 포함)는 직렬화 스키마
# 설계결정이라 본 변경에서 제외했다 — docs/IMPROVE_PROPOSALS.md 참조.
