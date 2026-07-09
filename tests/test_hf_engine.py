from jsteer.hf_engine import HFJSpaceEngine


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
