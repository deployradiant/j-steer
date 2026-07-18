from types import SimpleNamespace

import pytest
import torch

from jsteer.hf_engine import HFJSpaceEngine
from jsteer.schemas import LayerActivation, TraceFrame


def test_model_id_guard_accepts_case_difference() -> None:
    assert HFJSpaceEngine._model_ids_match("Qwen/Qwen3-1.7B", "qwen/qwen3-1.7b")


def test_model_id_guard_rejects_wrong_qwen_size() -> None:
    assert not HFJSpaceEngine._model_ids_match("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")


def test_lens_config_filename_uses_artifact_directory() -> None:
    assert (
        HFJSpaceEngine._lens_config_filename(
            "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"
        )
        == "qwen3-1.7b/jlens/Salesforce-wikitext/config.yaml"
    )


def test_direct_lens_match_prefers_best_vocabulary_rank() -> None:
    readouts = [
        LayerActivation(
            layer=8,
            concept_id="data_leakage",
            label="Data Leakage",
            score=0.6,
            polarity="risk",
            rank=26,
            matched_token="private",
            top_tokens=["private", "records"],
        ),
        LayerActivation(
            layer=14,
            concept_id="data_leakage",
            label="Data Leakage",
            score=0.9,
            polarity="risk",
            rank=4,
            matched_token="leak",
            top_tokens=["leak", "secret"],
        ),
    ]
    frame = TraceFrame(
        step=1,
        phase="decode",
        layer=14,
        layer_activations=readouts,
    )

    match = HFJSpaceEngine._best_lens_match(frame)
    scores = HFJSpaceEngine._lens_match_scores(readouts)

    assert match is not None
    assert (match.matched_token, match.rank, match.layer) == ("leak", 4, 14)
    assert scores[0].label == "Data Leakage"
    assert scores[0].evidence == ["leak", "L14 rank 4"]


def test_rewrite_delta_backpropagates_through_selected_jlens_matrix() -> None:
    class LinearLensModel:
        def __init__(self, unembedding: torch.Tensor) -> None:
            self.unembedding = unembedding

        def unembed(self, residual: torch.Tensor) -> torch.Tensor:
            return residual @ self.unembedding.T

    engine = HFJSpaceEngine()
    engine.torch = torch
    engine.layer = 0
    matrix = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    engine.jacobian_lens = SimpleNamespace(
        matrices={0: matrix},
        lens_model=LinearLensModel(torch.eye(2)),
    )

    delta = engine._jlens_rewrite_delta(
        hidden=torch.zeros(2),
        target_token_ids=[0],
        suppress_token_ids=[1],
        strength=3.0,
    )

    assert float(delta.norm()) == pytest.approx(3.0)
    assert float(delta[0]) > 0.0
    assert float(delta[1]) < 0.0
