"""Tests for the Vector Inference API data models."""

from vec_inf.client import LaunchOptions, ModelInfo, ModelStatus, ModelType


def test_model_info_creation():
    """Test creating a ModelInfo instance."""
    model = ModelInfo(
        name="test-model",
        family="test-family",
        variant="test-variant",
        model_type=ModelType.LLM,
        config={"gpus_per_node": 1},
    )

    assert model.name == "test-model"
    assert model.family == "test-family"
    assert model.variant == "test-variant"
    assert model.model_type == ModelType.LLM
    assert model.config["gpus_per_node"] == 1


def test_model_info_optional_fields():
    """Test ModelInfo with optional fields omitted."""
    model = ModelInfo(
        name="test-model",
        family="test-family",
        variant=None,
        model_type=ModelType.LLM,
        config={},
    )

    assert model.name == "test-model"
    assert model.family == "test-family"
    assert model.variant is None
    assert model.model_type == ModelType.LLM


def test_launch_options_default_values():
    """Test LaunchOptions with default values."""
    options = LaunchOptions()

    assert options.gpus_per_node is None
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
