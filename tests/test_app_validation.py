import app


def test_engine_order_follows_registry_not_completion_order():
    assert app.ordered_model_keys(["edge", "base", "piper", "edge"]) == ["base", "piper", "edge"]


def test_missing_optional_engine_does_not_break_app_import():
    assert "base" in app.MODEL_REGISTRY
