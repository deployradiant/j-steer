from types import SimpleNamespace

import pytest
import torch

from jsteer.concepts import REWRITE_PROFILES
from jsteer.hf_engine import HFJSpaceEngine
from jsteer.schemas import LayerActivation, RunRequest, TraceFrame


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


def test_policy_term_validation_accepts_capitalized_single_token_variant() -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return {"venus": [10, 11], " Venus": [84]}.get(text, [1, 2])

        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            del skip_special_tokens
            return " Venus" if token_ids == [84] else ""

    engine = HFJSpaceEngine()
    engine.tokenizer = FakeTokenizer()

    validation = engine._validate_policy_term("venus")

    assert validation.valid is True
    assert validation.token_id == 84
    assert validation.decoded_token == "Venus"


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
    engine.model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.eye(3)))
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
    metrics = engine._jlens_intervention_metrics(hidden, edited, 0, 1)

    assert target_token_id == 1
    assert edited.tolist() == pytest.approx([1.0, 2.0, 5.0], abs=1e-5)
    assert (metrics["source_rank_before"], metrics["source_rank_after"]) == (2, 3)
    assert (metrics["target_rank_before"], metrics["target_rank_after"]) == (3, 2)
    assert metrics["source_probability_after"] < metrics["source_probability_before"]
    assert metrics["target_probability_after"] > metrics["target_probability_before"]
    assert metrics["jlens_kl_divergence"] > 0
    assert metrics["jlens_total_variation"] > 0


def test_rewrite_uses_detected_jlens_layer_as_control_layer() -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [sum(text.encode()) % 1000]

    engine = HFJSpaceEngine()
    engine.torch = torch
    engine.tokenizer = FakeTokenizer()
    engine.jacobian_lens = SimpleNamespace()
    match = LayerActivation(
        layer=21,
        concept_id="sport_source",
        label="Soccer Report",
        score=0.9,
        polarity="risk",
        rank=3,
        matched_token="soccer",
    )

    state = engine._steering_state(
        REWRITE_PROFILES["sport_report_swap"],
        RunRequest(
            mode="rewrite",
            prompt="test",
            unsafe_words=["soccer"],
            replacement_words=["tennis"],
        ),
        initial_match=match,
    )

    assert state["control_layer"] == 21
