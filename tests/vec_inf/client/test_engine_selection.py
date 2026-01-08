"""Unit tests for engine selection and validation logic."""

from pathlib import Path
from unittest.mock import patch

import pytest

from vec_inf.client._client_vars import (
    ENGINE_SHORT_TO_LONG_MAP,
    SGLANG_SHORT_TO_LONG_MAP,
    SUPPORTED_ENGINES,
    VLLM_SHORT_TO_LONG_MAP,
)
from vec_inf.client._helper import ModelLauncher
from vec_inf.client.config import ModelConfig


class TestEngineSelection:
    """Tests for engine selection logic."""

    @pytest.fixture(autouse=True)
    def _set_required_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provide dummy required Slurm env vars so tests don't depend on host env."""
        monkeypatch.setenv("VEC_INF_ACCOUNT", "test-account")
        monkeypatch.setenv("VEC_INF_WORK_DIR", "/tmp")

    @pytest.fixture
    def model_config_vllm(self) -> ModelConfig:
        """Fixture providing a vLLM model configuration."""
        return ModelConfig(
            model_name="test-model",
            model_family="test-family",
            model_type="LLM",
            gpus_per_node=1,
            num_nodes=1,
            vocab_size=32000,
            model_weights_parent_dir=Path("/path/to/models"),
            engine="vllm",
            vllm_args={"--max-model-len": "8192"},
            sglang_args={},
        )

    @pytest.fixture
    def model_config_sglang(self) -> ModelConfig:
        """Fixture providing a SGLang model configuration."""
        return ModelConfig(
            model_name="test-model",
            model_family="test-family",
            model_type="LLM",
            gpus_per_node=1,
            num_nodes=1,
            vocab_size=32000,
            model_weights_parent_dir=Path("/path/to/models"),
            engine="sglang",
            vllm_args={},
            sglang_args={"--context-length": "8192"},
        )

    @pytest.fixture
    def model_config_default(self) -> ModelConfig:
        """Fixture providing a model configuration with default engine."""
        return ModelConfig(
            model_name="test-model",
            model_family="test-family",
            model_type="LLM",
            gpus_per_node=1,
            num_nodes=1,
            vocab_size=32000,
            model_weights_parent_dir=Path("/path/to/models"),
            vllm_args={},
            sglang_args={},
        )

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_selection_default(self, mock_load_config, model_config_default):
        """Test default engine selection (vllm)."""
        mock_load_config.return_value = [model_config_default]

        launcher = ModelLauncher("test-model", {})
        params = launcher.params

        assert launcher.engine == "vllm"
        assert params["engine"] == "vllm"
        assert "engine_args" in params

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_selection_explicit_vllm(
        self, mock_load_config, model_config_default
    ):
        """Test explicit vLLM engine selection."""
        mock_load_config.return_value = [model_config_default]

        launcher = ModelLauncher("test-model", {"engine": "vllm"})
        params = launcher.params

        assert launcher.engine == "vllm"
        assert params["engine"] == "vllm"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_selection_explicit_sglang(
        self, mock_load_config, model_config_sglang
    ):
        """Test explicit SGLang engine selection."""
        mock_load_config.return_value = [model_config_sglang]

        launcher = ModelLauncher("test-model", {"engine": "sglang"})
        params = launcher.params

        assert launcher.engine == "sglang"
        assert params["engine"] == "sglang"
        assert params["engine_args"] == model_config_sglang.sglang_args

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_selection_inferred_from_vllm_args(
        self, mock_load_config, model_config_default
    ):
        """Test engine inference from vLLM args."""
        mock_load_config.return_value = [model_config_default]

        launcher = ModelLauncher("test-model", {"vllm_args": "--max-model-len=8192"})

        assert launcher.engine == "vllm"
        assert launcher.params["engine_inferred"] is True

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_selection_inferred_from_sglang_args(
        self, mock_load_config, model_config_default
    ):
        """Test engine inference from SGLang args."""
        mock_load_config.return_value = [model_config_default]

        launcher = ModelLauncher("test-model", {"sglang_args": "--context-length=8192"})

        assert launcher.engine == "sglang"
        assert launcher.params["engine_inferred"] is True

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_selection_mismatch_error(
        self, mock_load_config, model_config_default
    ):
        """Test error when engine and args mismatch."""
        mock_load_config.return_value = [model_config_default]

        with pytest.raises(ValueError, match="Mismatch between provided engine"):
            ModelLauncher(
                "test-model",
                {"engine": "vllm", "sglang_args": "--context-length=8192"},
            )

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_selection_multiple_args_error(
        self, mock_load_config, model_config_default
    ):
        """Test error when multiple engine args provided."""
        mock_load_config.return_value = [model_config_default]

        with pytest.raises(
            ValueError,
            match="Cannot provide engine-specific args for multiple engines",
        ):
            ModelLauncher(
                "test-model",
                {
                    "vllm_args": "--max-model-len=8192",
                    "sglang_args": "--context-length=8192",
                },
            )

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_selection_unsupported_engine(
        self, mock_load_config, model_config_default
    ):
        """Test that unsupported engine values raise an error."""
        mock_load_config.return_value = [model_config_default]

        # Unsupported engines should raise KeyError when trying to access engine_args
        with pytest.raises(KeyError):
            ModelLauncher("test-model", {"engine": "unsupported"})


class TestEngineArgsProcessing:
    """Tests for engine argument processing."""

    @pytest.fixture(autouse=True)
    def _set_required_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provide dummy required Slurm env vars so tests don't depend on host env."""
        monkeypatch.setenv("VEC_INF_ACCOUNT", "test-account")
        monkeypatch.setenv("VEC_INF_WORK_DIR", "/tmp")

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Fixture providing a basic model configuration."""
        return ModelConfig(
            model_name="test-model",
            model_family="test-family",
            model_type="LLM",
            gpus_per_node=1,
            num_nodes=1,
            vocab_size=32000,
            model_weights_parent_dir=Path("/path/to/models"),
            vllm_args={},
            sglang_args={},
        )

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_args_processing_vllm(self, mock_load_config, model_config):
        """Test vLLM args processing."""
        updated_config = model_config.model_copy(
            update={
                "gpus_per_node": 4,
                "vllm_args": {"--tensor-parallel-size": "4"},
            }
        )
        mock_load_config.return_value = [updated_config]

        launcher = ModelLauncher(
            "test-model",
            {
                "engine": "vllm",
                "vllm_args": "--max-model-len=8192,--tensor-parallel-size=4,--enforce-eager",
            },
        )

        processed_args = launcher._process_engine_args(
            "--max-model-len=8192,--tensor-parallel-size=4,--enforce-eager", "vllm"
        )

        assert processed_args["--max-model-len"] == "8192"
        assert processed_args["--tensor-parallel-size"] == "4"
        assert processed_args["--enforce-eager"] is True

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_args_processing_sglang(self, mock_load_config, model_config):
        """Test SGLang args processing."""
        mock_load_config.return_value = [model_config]

        launcher = ModelLauncher("test-model", {})
        processed_args = launcher._process_engine_args(
            "--context-length=8192,--tensor-parallel-size=4,--mem-fraction-static=0.85",
            "sglang",
        )

        assert processed_args["--context-length"] == "8192"
        assert processed_args["--tensor-parallel-size"] == "4"
        assert processed_args["--mem-fraction-static"] == "0.85"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_args_short_to_long_mapping_vllm(
        self, mock_load_config, model_config
    ):
        """Test vLLM short arg to long arg mapping."""
        mock_load_config.return_value = [model_config]

        launcher = ModelLauncher("test-model", {})
        processed_args = launcher._process_engine_args("-tp=4,-pp=2,-q=awq", "vllm")

        assert processed_args["--tensor-parallel-size"] == "4"
        assert processed_args["--pipeline-parallel-size"] == "2"
        assert processed_args["--quantization"] == "awq"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_args_short_to_long_mapping_sglang(
        self, mock_load_config, model_config
    ):
        """Test SGLang short arg to long arg mapping."""
        mock_load_config.return_value = [model_config]

        launcher = ModelLauncher("test-model", {})
        processed_args = launcher._process_engine_args("--tp=4,--pp=2,--dp=1", "sglang")

        assert processed_args["--tensor-parallel-size"] == "4"
        assert processed_args["--pipeline-parallel-size"] == "2"
        assert processed_args["--data-parallel-size"] == "1"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_args_compilation_config_vllm_only(
        self, mock_load_config, model_config
    ):
        """Test -O flag only works for vLLM."""
        mock_load_config.return_value = [model_config]

        launcher = ModelLauncher("test-model", {})

        # Should work for vLLM
        processed_args = launcher._process_engine_args("-O3", "vllm")
        assert processed_args["--compilation-config"] == "3"

        # Should raise error for SGLang
        with pytest.raises(ValueError, match="-O is only supported for vLLM"):
            launcher._process_engine_args("-O3", "sglang")

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_args_boolean_values(self, mock_load_config, model_config):
        """Test boolean argument values."""
        mock_load_config.return_value = [model_config]

        launcher = ModelLauncher("test-model", {})
        processed_args = launcher._process_engine_args(
            "--trust-remote-code,--disable-log-stats,--max-model-len=8192", "vllm"
        )

        assert processed_args["--trust-remote-code"] is True
        assert processed_args["--disable-log-stats"] is True
        assert processed_args["--max-model-len"] == "8192"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_args_with_spaces(self, mock_load_config, model_config):
        """Test argument parsing with spaces."""
        mock_load_config.return_value = [model_config]

        launcher = ModelLauncher("test-model", {})
        processed_args = launcher._process_engine_args(
            " --max-model-len=8192 , --tensor-parallel-size=4 ", "vllm"
        )

        assert processed_args["--max-model-len"] == "8192"
        assert processed_args["--tensor-parallel-size"] == "4"


class TestEngineConstants:
    """Tests for engine-related constants."""

    def test_supported_engines(self):
        """Test SUPPORTED_ENGINES constant."""
        assert "vllm" in SUPPORTED_ENGINES
        assert "sglang" in SUPPORTED_ENGINES
        assert len(SUPPORTED_ENGINES) == 2

    def test_engine_short_to_long_map(self):
        """Test ENGINE_SHORT_TO_LONG_MAP structure."""
        assert "vllm" in ENGINE_SHORT_TO_LONG_MAP
        assert "sglang" in ENGINE_SHORT_TO_LONG_MAP
        assert ENGINE_SHORT_TO_LONG_MAP["vllm"] == VLLM_SHORT_TO_LONG_MAP
        assert ENGINE_SHORT_TO_LONG_MAP["sglang"] == SGLANG_SHORT_TO_LONG_MAP

    def test_vllm_short_to_long_map(self):
        """Test vLLM short to long argument mappings."""
        assert "-tp" in VLLM_SHORT_TO_LONG_MAP
        assert VLLM_SHORT_TO_LONG_MAP["-tp"] == "--tensor-parallel-size"
        assert "-O" in VLLM_SHORT_TO_LONG_MAP
        assert VLLM_SHORT_TO_LONG_MAP["-O"] == "--compilation-config"

    def test_sglang_short_to_long_map(self):
        """Test SGLang short to long argument mappings."""
        assert "--tp" in SGLANG_SHORT_TO_LONG_MAP
        assert SGLANG_SHORT_TO_LONG_MAP["--tp"] == "--tensor-parallel-size"
        assert "--tp-size" in SGLANG_SHORT_TO_LONG_MAP
        assert SGLANG_SHORT_TO_LONG_MAP["--tp-size"] == "--tensor-parallel-size"
