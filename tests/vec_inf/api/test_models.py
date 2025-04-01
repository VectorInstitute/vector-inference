"""Tests for the Vector Inference API data models."""

from vec_inf.api import LaunchOptions, ModelInfo, ModelStatus, ModelType


def test_model_info_creation():
    """Test creating a ModelInfo instance."""
    model = ModelInfo(
        name="test-model",
        family="test-family",
        variant="test-variant",
        type=ModelType.LLM,
        config={"num_gpus": 1},
    )

    assert model.name == "test-model"
    assert model.family == "test-family"
    assert model.variant == "test-variant"
    assert model.type == ModelType.LLM
    assert model.config["num_gpus"] == 1


def test_model_info_optional_fields():
    """Test ModelInfo with optional fields omitted."""
    model = ModelInfo(
        name="test-model",
        family="test-family",
        variant=None,
        type=ModelType.LLM,
        config={},
    )

    assert model.name == "test-model"
    assert model.family == "test-family"
    assert model.variant is None
    assert model.type == ModelType.LLM


def test_launch_options_default_values():
    """Test LaunchOptions with default values."""
    options = LaunchOptions()

    assert options.num_gpus is None
    assert options.partition is None
    assert options.data_type is None
    assert options.num_nodes is None
    assert options.model_family is None


def test_model_status_enum():
    """Test ModelStatus enum values."""
    assert ModelStatus.PENDING.value == "PENDING"
    assert ModelStatus.LAUNCHING.value == "LAUNCHING"
    assert ModelStatus.READY.value == "READY"
    assert ModelStatus.FAILED.value == "FAILED"
    assert ModelStatus.SHUTDOWN.value == "SHUTDOWN"
