from fastapi.testclient import TestClient

from jsteer.api import create_app


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
    assert body["frames"][0]["proxy_concepts"]
    assert body["frames"][0]["jacobian_concepts"]
    assert body["frames"][0]["jacobian_tokens"]
