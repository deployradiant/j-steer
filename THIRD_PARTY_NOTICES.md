# Third-Party Notices

J-Steer does not vendor or redistribute model weights or fitted Jacobian-lens artifacts. The runtime downloads them from their respective repositories. Users are responsible for reviewing the terms that apply to those artifacts.

| Component | Source | License |
| --- | --- | --- |
| Anthropic Jacobian lens code | `anthropics/jacobian-lens` | Apache-2.0 |
| Qwen3-1.7B model and tokenizer | `Qwen/Qwen3-1.7B` | Apache-2.0 |
| Neuronpedia fitted Jacobian lenses | `neuronpedia/jacobian-lens` | MIT |

Python and frontend packages retain their own licenses. Exact package versions and sources are recorded in `uv.lock` and `frontend/package-lock.json`.
