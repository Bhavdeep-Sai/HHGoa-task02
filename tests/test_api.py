import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "IndicVoiceRAG" in data["service"]


def test_config_endpoint():
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "fast_path_threshold" in data


def test_text_query_api():
    res = client.post("/api/query", json={"query": "Manhattan project successful hone ke baad kya hua?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "stage_latencies" in data
    assert data["detected_language"] in ["hi", "en"]


def test_metrics_endpoint():
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "total_queries" in data
