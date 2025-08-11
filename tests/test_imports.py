import pytest

def test_rediswriter_import():
    try:
        from movementpredictor.config import ModelConfig
    except ImportError as e:
        pytest.fail(f"Failed to import ModelConfig: {e}")

    assert ModelConfig is not None, "ModelConfig should be imported successfully"