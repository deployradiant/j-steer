from __future__ import annotations

from dataclasses import dataclass

from .schemas import LensPublic

DEFAULT_LENS_ID = "anthropic-qwen3-1.7b-wikitext"
QWEN_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
JLENS_ARTIFACT_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"


@dataclass(frozen=True)
class LensSpec:
    id: str
    label: str
    description: str
    source: str
    revision: str
    artifact: str
    target_model: str
    dataset: str
    fitted_layers: int


LENSES: dict[str, LensSpec] = {
    DEFAULT_LENS_ID: LensSpec(
        id=DEFAULT_LENS_ID,
        label="Anthropic J-lens - Qwen3 1.7B",
        description="Published Jacobian lens fit for the exact Qwen checkpoint used by this runtime.",
        source="neuronpedia/jacobian-lens",
        revision=JLENS_ARTIFACT_REVISION,
        artifact="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
        target_model="Qwen/Qwen3-1.7B",
        dataset="WikiText-103",
        fitted_layers=27,
    )
}


def public_lenses() -> list[LensPublic]:
    return [
        LensPublic(
            id=lens.id,
            label=lens.label,
            description=lens.description,
            source=lens.source,
            revision=lens.revision,
            artifact=lens.artifact,
            target_model=lens.target_model,
            dataset=lens.dataset,
            fitted_layers=lens.fitted_layers,
        )
        for lens in LENSES.values()
    ]
