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
    assert options.engine is None
    assert options.vllm_args is None
    assert options.sglang_args is None


def test_model_status_enum():
    """Test ModelStatus enum values."""
    assert ModelStatus.PENDING.value == "PENDING"
    assert ModelStatus.LAUNCHING.value == "LAUNCHING"
    assert ModelStatus.READY.value == "READY"
    assert ModelStatus.FAILED.value == "FAILED"
    assert ModelStatus.SHUTDOWN.value == "SHUTDOWN"


def test_launch_options_with_engine():
    """Test LaunchOptions with engine parameter."""
    options = LaunchOptions(engine="sglang")

    assert options.engine == "sglang"


def test_launch_options_with_vllm_args():
    """Test LaunchOptions with vLLM args."""
    options = LaunchOptions(vllm_args="--max-model-len=8192,--tensor-parallel-size=4")

    assert options.vllm_args == "--max-model-len=8192,--tensor-parallel-size=4"


def test_launch_options_with_sglang_args():
    """Test LaunchOptions with SGLang args."""
    options = LaunchOptions(
        sglang_args="--context-length=8192,--tensor-parallel-size=4"
    )

    assert options.sglang_args == "--context-length=8192,--tensor-parallel-size=4"


def test_launch_options_with_all_engine_fields():
    """Test LaunchOptions with all engine-related fields."""
    options = LaunchOptions(
        engine="vllm",
        vllm_args="--max-model-len=8192",
        sglang_args=None,
    )

    assert options.engine == "vllm"
    assert options.vllm_args == "--max-model-len=8192"
    assert options.sglang_args is None
