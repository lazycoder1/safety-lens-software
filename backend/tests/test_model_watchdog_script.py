from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = ROOT / "scripts" / "ops" / "rakshak-model-watchdog.sh"


def test_model_watchdog_defaults_to_container_network_health():
    script = WATCHDOG.read_text(encoding="utf-8")

    assert 'HEALTH_CHECK_MODE="${HEALTH_CHECK_MODE:-container}"' in script
    assert "container_http_healthy()" in script
    assert "docker exec \"$container\" python3" in script
    assert 'http_healthy "$MODEL_CONTAINER" "$MODEL_HEALTH_URL"' in script


def test_model_watchdog_reuses_container_health_after_restart():
    script = WATCHDOG.read_text(encoding="utf-8")

    assert script.count('http_healthy "$MODEL_CONTAINER" "$MODEL_HEALTH_URL"') == 2
    assert 'http_healthy "$EDGE_CONTAINER" "$BACKEND_HEALTH_URL"' in script
