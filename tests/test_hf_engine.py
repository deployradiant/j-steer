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


def test_policy_term_validation_uses_single_qwen_token_variant() -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return {"secret": [10, 11], " secret": [42]}.get(text, [1, 2])

        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            del skip_special_tokens
            return " secret" if token_ids == [42] else ""

    engine = HFJSpaceEngine()
    engine.tokenizer = FakeTokenizer()

    validation = engine._validate_policy_term("secret")

    assert validation.valid is True
    assert validation.token_id == 42
    assert validation.decoded_token == "secret"


def test_rewrite_swaps_jlens_coordinates_and_preserves_orthogonal_residual() -> None:
    class LinearLensModel:
        def __init__(self, unembedding: torch.Tensor) -> None:
            self.unembedding = unembedding

        def unembed(self, residual: torch.Tensor) -> torch.Tensor:
            return residual @ self.unembedding.T

    engine = HFJSpaceEngine()
    engine.torch = torch
    engine.layer = 0
    matrix = torch.eye(3)
    engine.jacobian_lens = SimpleNamespace(
        matrices={0: matrix},
        lens_model=LinearLensModel(torch.eye(3)),
    )

    hidden = torch.tensor([2.0, 1.0, 5.0])
    delta, target_token_id = engine._jlens_coordinate_swap_delta(
        hidden=hidden,
        source_token_id=0,
        target_token_ids=[1],
        scale=1.0,
    )
    edited = hidden + delta

    assert target_token_id == 1
    assert edited.tolist() == pytest.approx([1.0, 2.0, 5.0], abs=1e-5)
