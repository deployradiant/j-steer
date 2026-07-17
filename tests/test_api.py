import json

from fastapi.testclient import TestClient

from jsteer.api import create_app


def test_static_public_asset(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<div>app</div>", encoding="utf-8")
    (tmp_path / "radiant-favicon.png").write_bytes(b"png")
    client = TestClient(create_app(engine_mode="mock", static_dir=tmp_path))

    response = client.get("/radiant-favicon.png")

    assert response.status_code == 200
    assert response.content == b"png"


def test_profiles_endpoint() -> None:
    client = TestClient(create_app(engine_mode="mock"))
    response = client.get("/api/profiles")

    assert response.status_code == 200
    body = response.json()
    assert body["concepts"]
    assert body["rewrite_profiles"]
    assert body["scenarios"]


def test_run_endpoint() -> None:
    client = TestClient(create_app(engine_mode="mock"))
    response = client.post(
        "/api/run",
        json={
            "mode": "monitor",
            "prompt": "Write a normal enterprise checklist.",
            "monitor_ids": ["credential_exfiltration"],
            "action": "observe",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["frames"][0]["proxy_concepts"] == []
    assert body["frames"][0]["jacobian_tokens"]
    assert body["frames"][0]["layer_activations"]
    assert body["jspace_injections"] == []
    assert {
        "layer",
        "label",
        "score",
        "polarity",
        "rank",
        "matched_token",
        "top_tokens",
    }.issubset(body["frames"][0]["layer_activations"][0])


def test_run_stream_endpoint() -> None:
    client = TestClient(create_app(engine_mode="mock"))

    with client.stream(
        "POST",
        "/api/run/stream",
        json={
            "mode": "monitor",
            "prompt": "Write a normal enterprise checklist.",
            "monitor_ids": ["credential_exfiltration"],
            "action": "observe",
        },
    ) as response:
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert events[0]["type"] == "start"
    assert events[0]["response"]["status"] == "running"
    assert any(event["type"] == "frame" for event in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["response"]["status"] == "completed"
    assert events[-1]["response"]["frames"]


def test_custom_lens_endpoint() -> None:
    client = TestClient(create_app(engine_mode="mock"))
    response = client.post(
        "/api/custom-lens",
        json={
            "prompt": "The buyer asks about SOC 2 audit logs, pricing, and Salesforce alternatives.",
            "concepts": [
                {
                    "label": "Security Assurance",
                    "keywords": ["SOC 2", "audit logs", "SSO"],
                    "polarity": "target",
                },
                {
                    "label": "Competitor Drift",
                    "keywords": ["Salesforce", "HubSpot", "alternative"],
                    "polarity": "risk",
                },
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["frame"]["proxy_concepts"]
    assert body["frame"]["jacobian_concepts"]
    assert body["frame"]["layer_activations"]
    assert body["jspace_injections"]
    assert body["jspace_injections"][0]["keyword"].lower() in {"soc 2", "audit logs"}
    assert body["concepts"][0]["calibration_prompts"]
    assert any("Lookup injection" in prompt for prompt in body["concepts"][0]["calibration_prompts"])


def test_rewrite_includes_triggered_lookup_injection() -> None:
    client = TestClient(create_app(engine_mode="mock"))
    response = client.post(
        "/api/run",
        json={
            "mode": "rewrite",
            "prompt": "Write HeliosCRM product copy for buyers asking about SOC 2 audit logs and Salesforce.",
            "rewrite_profile_id": "brand_canon_injection",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rewritten"
    assert body["jspace_injections"]
    assert body["jspace_injections"][0]["target"] == "rewrite_controller"
    assert "Retrieved context" in body["output"]
