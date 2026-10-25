"""Unit tests for helper components in the vec_inf.client module."""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import requests

from vec_inf.client._client_vars import SRC_DIR
from vec_inf.client._exceptions import (
    MissingRequiredFieldsError,
    ModelConfigurationError,
    ModelNotFoundError,
    SlurmJobError,
)
from vec_inf.client._helper import (
    BatchModelLauncher,
    ModelLauncher,
    ModelRegistry,
    ModelStatusMonitor,
    PerformanceMetricsCollector,
)
from vec_inf.client._slurm_script_generator import format_engine_arg_value
from vec_inf.client.config import ModelConfig
from vec_inf.client.models import (
    ModelStatus,
    ModelType,
    StatusResponse,
)


class TestModelLauncher:
    """Tests for the ModelLauncher class."""

    @pytest.fixture(autouse=True)
    def _mock_validate_weights_path(self) -> None:
        """Avoid disk checks for fake model paths in fixtures."""
        with patch("vec_inf.client._helper.utils.validate_weights_path"):
            yield

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Fixture providing a basic model configuration for tests."""
        return ModelConfig(
            model_name="test-model",
            model_family="test-family",
            model_type="LLM",
            gpus_per_node=1,
            num_nodes=1,
            vocab_size=32000,
            model_weights_parent_dir=Path("/path/to/models"),
            resource_type="l40s",
            partition="gpu",
            qos="normal",
            time="01:00:00",
            cpus_per_task=8,
            mem_per_node="32G",
            account="test-account",
            work_dir="/tmp/test-work",
            engine="vllm",
            vllm_args={},
            sglang_args={},
        )

    @pytest.fixture
    def mock_configs(self, model_config):
        """Fixture providing a list of model configurations."""
        return [model_config]

    @patch("vec_inf.client._helper.utils.load_config")
    def test_init_with_existing_config(self, mock_load_config, mock_configs):
        """Test launcher initializes correctly when config exists."""
        mock_load_config.return_value = mock_configs
        launcher = ModelLauncher("test-model", {})

        assert launcher.model_name == "test-model"
        assert launcher.model_config.model_name == "test-model"
        assert launcher.model_config.model_family == "test-family"

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_init_with_missing_config_and_existing_weights(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Test fallback to dummy config when config is missing but weights exist."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        with pytest.warns(UserWarning):
            launcher = ModelLauncher(
                "unknown-model",
                {"account": "test-account", "work_dir": "/tmp/test-work"},
            )

        assert launcher.model_name == "unknown-model"
        assert launcher.model_config.model_name == "unknown-model"
        assert launcher.model_config.model_family == "model_family_placeholder"

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_init_with_missing_config_and_missing_weights(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Test error is raised when both config and weights are missing."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = False

        with pytest.raises(ModelConfigurationError):
            ModelLauncher("unknown-model", {})

    @patch("vec_inf.client._helper.utils.load_config")
    def test_init_with_empty_configs(self, mock_load_config):
        """Test error is raised when config list is empty."""
        mock_load_config.return_value = []

        with pytest.raises(ModelNotFoundError):
            ModelLauncher("test-model", {})

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_vllm(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Test vLLM args are parsed correctly into key-value pairs."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        launcher = ModelLauncher("test-model", {})
        vllm_args = launcher._process_engine_args(
            "--tensor-parallel-size=4,--quantization=awq,-O3", "vllm"
        )

        assert vllm_args["--tensor-parallel-size"] == "4"
        assert vllm_args["--quantization"] == "awq"
        assert vllm_args["--optimization-level"] == "3"

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_vllm_short_aliases(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Test vLLM short aliases, including boolean ones, map to long names."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        launcher = ModelLauncher("test-model", {})
        vllm_args = launcher._process_engine_args(
            "-tp=2, -pp=2, -dp=2, -ep, -q=fp8, --max-model-len=4096", "vllm"
        )

        assert vllm_args == {
            "--tensor-parallel-size": "2",
            "--pipeline-parallel-size": "2",
            "--data-parallel-size": "2",
            "--enable-expert-parallel": True,
            "--quantization": "fp8",
            "--max-model-len": "4096",
        }

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_json_values(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Commas inside JSON objects, lists and quotes do not split arguments."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        launcher = ModelLauncher("test-model", {})
        vllm_args = launcher._process_engine_args(
            '-sc={"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4},'
            '-cc={"cudagraph_capture_sizes":[1,2,4]},'
            "--max-model-len=8192,"
            '-ac={"backend": "FLASH_ATTN"},'
            '-dc={"steps": 4}',
            "vllm",
        )

        assert vllm_args == {
            "--speculative-config": (
                '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4}'
            ),
            "--compilation-config": '{"cudagraph_capture_sizes":[1,2,4]}',
            "--max-model-len": "8192",
            "--attention-config": '{"backend": "FLASH_ATTN"}',
            "--diffusion-config": '{"steps": 4}',
        }

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_multi_value_flags(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Flags taking several values keep all of them, in both forms."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True
        launcher = ModelLauncher("test-model", {})

        for arg_string in (
            "--cudagraph-capture-sizes 1 2 4 8,--served-model-name a b",
            "--cudagraph-capture-sizes=1 2 4 8,--served-model-name=a b",
        ):
            args = launcher._process_engine_args(arg_string, "vllm")
            assert args == {
                "--cudagraph-capture-sizes": "1 2 4 8",
                "--served-model-name": "a b",
            }
            # Each value becomes separate words in the generated script
            assert format_engine_arg_value(args["--cudagraph-capture-sizes"]) == (
                "1 2 4 8"
            )
            assert format_engine_arg_value(args["--served-model-name"]) == "a b"

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_quotes_outside_json(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """A quote outside braces does not swallow the following arguments."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True
        launcher = ModelLauncher("test-model", {})

        args = launcher._process_engine_args("--x=it's,-tp=2", "vllm")

        assert args == {"--x": "it's", "--tensor-parallel-size": "2"}

    @pytest.mark.parametrize(
        "arg_string",
        [
            '-sc={"method":"ngram",-tp=2',
            "-cc=[1,2,-tp=2",
            '-sc={"method":"ngram},-tp=2',
            "-tp=2}",
        ],
    )
    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_unbalanced(
        self, mock_path_exists, mock_load_config, arg_string, mock_configs
    ):
        """Unbalanced braces, brackets or quotes raise a clear error."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True
        launcher = ModelLauncher("test-model", {})

        with pytest.raises(ValueError, match="engine arguments"):
            launcher._process_engine_args(arg_string, "vllm")

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_space_separated_values(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """'-tp 2' is handled like '-tp=2', and JSON may follow a space."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        launcher = ModelLauncher("test-model", {})
        vllm_args = launcher._process_engine_args(
            '-tp 2, --max-model-len 4096, -sc {"method": "ngram"}, -ep', "vllm"
        )

        assert vllm_args == {
            "--tensor-parallel-size": "2",
            "--max-model-len": "4096",
            "--speculative-config": '{"method": "ngram"}',
            "--enable-expert-parallel": True,
        }
        sglang_args = launcher._process_engine_args("--tp 4, --ep-size 2", "sglang")
        assert sglang_args == {
            "--tensor-parallel-size": "4",
            "--expert-parallel-size": "2",
        }

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_value_with_equals(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Test only the first '=' separates the flag from its value."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        launcher = ModelLauncher("test-model", {})
        vllm_args = launcher._process_engine_args(
            "--override-generation-config=temperature=0.5", "vllm"
        )

        assert vllm_args == {"--override-generation-config": "temperature=0.5"}

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_sglang_expert_parallel_aliases(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Test --ep and --ep-size both map to SGLang's --expert-parallel-size."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        launcher = ModelLauncher("test-model", {})
        assert launcher._process_engine_args("--ep-size=4", "sglang") == {
            "--expert-parallel-size": "4"
        }
        assert launcher._process_engine_args("--ep=2", "sglang") == {
            "--expert-parallel-size": "2"
        }

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_engine_args_sglang(
        self, mock_path_exists, mock_load_config, mock_configs
    ):
        """Test SGLang args are parsed correctly into key-value pairs."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        launcher = ModelLauncher("test-model", {})
        sglang_args = launcher._process_engine_args(
            "--context-length=8192,--tensor-parallel-size=4,--mem-fraction-static=0.85",
            "sglang",
        )

        assert sglang_args["--context-length"] == "8192"
        assert sglang_args["--tensor-parallel-size"] == "4"
        assert sglang_args["--mem-fraction-static"] == "0.85"

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.exists")
    def test_process_env_vars(self, mock_path_exists, mock_load_config, mock_configs):
        """Test that vars from `--env` flag are parsed correctly."""
        mock_load_config.return_value = mock_configs
        mock_path_exists.return_value = True

        # Get filepath of dummy env file
        file_path = Path(__file__).parent / "test_vars.env"

        launcher = ModelLauncher("test-model", {})
        env_vars = launcher._process_env_vars(f"CACHE_DIR=/cache,{file_path}")
        assert env_vars["CACHE_DIR"] == "/cache"
        assert env_vars["MY_VAR"] == "5"
        assert env_vars["VLLM_CACHE_ROOT"] == "/cache/vllm"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_launch_params_merges_config_and_cli_args(
        self, mock_load_config, model_config
    ):
        """Test _get_launch_params merges config and CLI engine_args correctly."""
        mock_load_config.return_value = [model_config]
        cli_kwargs = {"vllm_args": "--num-scheduler-steps=16, -pp=4", "num_nodes": 4}

        launcher = ModelLauncher("test-model", cli_kwargs)
        params = launcher.params

        assert params["num_nodes"] == "4"
        assert params["engine_args"]["--num-scheduler-steps"] == "16"
        assert params["engine_args"]["--pipeline-parallel-size"] == "4"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_launch_params_with_multi_gpu_no_tp(
        self, mock_load_config, model_config
    ):
        """Test the case tensor parallelism is missing in multi-GPU setup."""
        updated_config = model_config.model_copy(
            update={
                "gpus_per_node": 2,
            }
        )
        mock_load_config.return_value = [updated_config]

        with pytest.raises(MissingRequiredFieldsError) as excinfo:
            ModelLauncher("test-model", {})

        assert "--tensor-parallel-size" in str(excinfo.value)

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_check_override_with_vllm(self, mock_load_config, model_config):
        """Test vLLM engine selection with explicit engine arg."""
        mock_load_config.return_value = [model_config]

        launcher = ModelLauncher("test-model", {"engine": "vllm"})
        params = launcher.params

        assert launcher.engine == "vllm"
        assert params["engine"] == "vllm"
        assert "engine_args" in params

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_check_override_with_sglang(self, mock_load_config, model_config):
        """Test SGLang engine selection with explicit engine arg."""
        updated_config = model_config.model_copy(
            update={
                "engine": "sglang",
                "sglang_args": {"--context-length": "8192"},
            }
        )
        mock_load_config.return_value = [updated_config]

        launcher = ModelLauncher("test-model", {"engine": "sglang"})
        params = launcher.params

        assert launcher.engine == "sglang"
        assert params["engine"] == "sglang"
        assert params["engine_args"] == updated_config.sglang_args

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_check_override_inferred_from_args(
        self, mock_load_config, model_config
    ):
        """Test engine inference from engine-specific args."""
        mock_load_config.return_value = [model_config]

        launcher = ModelLauncher("test-model", {"sglang_args": "--context-length=8192"})

        assert launcher.engine == "sglang"
        assert launcher.params["engine_inferred"] is True

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_check_override_mismatch_error(self, mock_load_config, model_config):
        """Test error when engine and args mismatch."""
        mock_load_config.return_value = [model_config]

        with pytest.raises(ValueError, match="Mismatch between provided engine"):
            ModelLauncher(
                "test-model",
                {"engine": "vllm", "sglang_args": "--context-length=8192"},
            )

    @patch("vec_inf.client._helper.utils.load_config")
    def test_engine_check_override_multiple_engine_args_error(
        self, mock_load_config, model_config
    ):
        """Test error when multiple engine args provided."""
        mock_load_config.return_value = [model_config]

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
    def test_validate_resource_allocation_with_sglang(
        self, mock_load_config, model_config
    ):
        """Test resource validation with SGLang engine."""
        updated_config = model_config.model_copy(
            update={
                "engine": "sglang",
                "gpus_per_node": 2,
                "sglang_args": {"--tensor-parallel-size": "2"},
            }
        )
        mock_load_config.return_value = [updated_config]

        launcher = ModelLauncher("test-model", {})
        params = launcher.params

        assert launcher.engine == "sglang"
        assert params["engine_args"]["--tensor-parallel-size"] == "2"

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("vec_inf.client._helper.SlurmScriptGenerator")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.open")
    @patch("pathlib.Path.rename")
    def test_launch(
        self,
        mock_rename,
        mock_open,
        mock_touch,
        mock_mkdir,
        mock_script_gen,
        mock_run_bash,
        mock_load_config,
        mock_configs,
    ):
        """Test successful model launch returns expected response and creates logs."""
        mock_open.return_value = mock.mock_open().return_value
        mock_load_config.return_value = mock_configs
        mock_script_gen_instance = MagicMock()
        mock_script_gen_instance.write_to_log_dir.return_value = Path(
            "/path/to/slurm_script.sh"
        )
        mock_script_gen.return_value = mock_script_gen_instance
        mock_run_bash.return_value = ("Submitted batch job 12345", "")

        launcher = ModelLauncher("test-model", {})
        response = launcher.launch()

        assert response.slurm_job_id == "12345"
        assert response.model_name == "test-model"
        assert "slurm_job_id" in response.config
        assert response.config["slurm_job_id"] == "12345"

        mock_mkdir.assert_called()
        mock_touch.assert_called()
        mock_rename.assert_called()

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("vec_inf.client._helper.utils.run_bash_command")
    def test_launch_with_slurm_error(
        self, mock_run_bash, mock_load_config, model_config
    ):
        """Test launch raises error on SLURM submission failure."""
        mock_load_config.return_value = [model_config]
        mock_run_bash.return_value = ("", "sbatch: error: Invalid partition specified")

        launcher = ModelLauncher("test-model", {})
        with pytest.raises(SlurmJobError):
            launcher.launch()

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("vec_inf.client._helper.SlurmScriptGenerator")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.open")
    @patch("pathlib.Path.rename")
    def test_launch_with_sglang_engine(
        self,
        mock_rename,
        mock_open,
        mock_touch,
        mock_mkdir,
        mock_script_gen,
        mock_run_bash,
        mock_load_config,
        model_config,
    ):
        """Test launch with SGLang engine."""
        updated_config = model_config.model_copy(
            update={
                "engine": "sglang",
                "sglang_args": {"--context-length": "8192"},
            }
        )
        mock_open.return_value = mock.mock_open().return_value
        mock_load_config.return_value = [updated_config]
        mock_script_gen_instance = MagicMock()
        mock_script_gen_instance.write_to_log_dir.return_value = Path(
            "/path/to/slurm_script.sh"
        )
        mock_script_gen.return_value = mock_script_gen_instance
        mock_run_bash.return_value = ("Submitted batch job 12345", "")

        launcher = ModelLauncher("test-model", {"engine": "sglang"})
        response = launcher.launch()

        assert response.slurm_job_id == "12345"
        assert launcher.engine == "sglang"
        assert response.config["engine"] == "sglang"


class TestBatchModelLauncher:
    """Tests for the BatchModelLauncher class."""

    @pytest.fixture(autouse=True)
    def _mock_validate_weights_path(self) -> None:
        """Avoid disk checks for fake model paths in fixtures."""
        with patch("vec_inf.client._helper.utils.validate_weights_path"):
            yield

    @pytest.fixture
    def batch_model_configs(self) -> list[ModelConfig]:
        """Fixture providing batch model configurations for tests."""
        return [
            ModelConfig(
                model_name="family1-variant1",
                model_family="family1",
                model_variant="variant1",
                model_type="LLM",
                gpus_per_node=1,
                num_nodes=1,
                vocab_size=32000,
                model_weights_parent_dir=Path("/path/to/models"),
                resource_type="l40s",
                partition="gpu",
                qos="normal",
                time="01:00:00",
                cpus_per_task=8,
                mem_per_node="32G",
                account="test-account",
                work_dir="/tmp/test-work",
                engine="vllm",
                vllm_args={},
                sglang_args={},
            ),
            ModelConfig(
                model_name="family2-variant1",
                model_family="family2",
                model_variant="variant1",
                model_type="VLM",
                gpus_per_node=1,
                num_nodes=1,
                vocab_size=65536,
                model_weights_parent_dir=Path("/path/to/models"),
                resource_type="l40s",
                partition="gpu",
                qos="normal",
                time="01:00:00",
                cpus_per_task=8,
                mem_per_node="32G",
                account="test-account",
                work_dir="/tmp/test-work",
                engine="vllm",
                vllm_args={},
                sglang_args={},
            ),
            ModelConfig(
                model_name="family1-variant2",
                model_family="family1",
                model_variant="variant2",
                model_type="LLM",
                gpus_per_node=1,
                num_nodes=1,
                vocab_size=32000,
                model_weights_parent_dir=Path("/path/to/models"),
                resource_type="l40s",
                partition="gpu",
                qos="normal",
                time="01:00:00",
                cpus_per_task=8,
                mem_per_node="32G",
                account="test-account",
                work_dir="/tmp/test-work",
                engine="vllm",
                vllm_args={},
                sglang_args={},
            ),
        ]

    @patch("vec_inf.client._helper.utils.load_config")
    def test_init_with_valid_configs(self, mock_load_config, batch_model_configs):
        """Test launcher initializes correctly with valid model configurations."""
        mock_load_config.return_value = batch_model_configs
        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1"],
            account="test-account",
            work_dir="/tmp/test-work",
        )

        assert launcher.model_names == ["family1-variant1", "family2-variant1"]
        assert launcher.slurm_job_name == "BATCH-family1-variant1-family2-variant1"
        assert len(launcher.model_configs) == 2
        assert "family1-variant1" in launcher.model_configs
        assert "family2-variant1" in launcher.model_configs

    @patch("vec_inf.client._helper.utils.load_config")
    def test_init_with_missing_model_config(
        self, mock_load_config, batch_model_configs
    ):
        """Test error is raised when one of the models is missing from config."""
        mock_load_config.return_value = batch_model_configs

        with pytest.raises(ModelConfigurationError) as excinfo:
            BatchModelLauncher(["family1-variant1", "nonexistent-model"])

        assert "nonexistent-model" in str(excinfo.value)
        assert "not found in configuration" in str(excinfo.value)

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_slurm_job_name(self, mock_load_config, batch_model_configs):
        """Test SLURM job name is constructed correctly from model names."""
        mock_load_config.return_value = batch_model_configs
        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1", "family1-variant2"],
            account="test-account",
            work_dir="/tmp/test-work",
        )

        assert (
            launcher.slurm_job_name
            == "BATCH-family1-variant1-family2-variant1-family1-variant2"
        )

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("pathlib.Path.mkdir")
    def test_get_launch_params_creates_log_dirs(
        self, mock_mkdir, mock_load_config, batch_model_configs
    ):
        """Test launch parameters preparation creates log directories."""
        mock_load_config.return_value = batch_model_configs

        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1", "family1-variant2"],
            account="test-account",
            work_dir="/tmp/test-work",
        )
        params = launcher.params

        assert "models" in params
        assert "family1-variant1" in params["models"]
        assert "family2-variant1" in params["models"]
        assert "family1-variant2" in params["models"]
        assert (
            params["slurm_job_name"]
            == "BATCH-family1-variant1-family2-variant1-family1-variant2"
        )
        assert params["src_dir"] == str(SRC_DIR)

        # Check that log directories are created
        mock_mkdir.assert_called()

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_launch_params_with_multi_gpu_no_tp(
        self, mock_load_config, batch_model_configs
    ):
        """Test error when tensor parallelism is missing in multi-GPU setup."""
        updated_configs = [
            batch_model_configs[0].model_copy(update={"gpus_per_node": 2}),
            batch_model_configs[1],
        ]
        mock_load_config.return_value = updated_configs

        with pytest.raises(MissingRequiredFieldsError) as excinfo:
            BatchModelLauncher(
                ["family1-variant1", "family2-variant1"],
                account="test-account",
                work_dir="/tmp/test-work",
            )

        assert "--tensor-parallel-size" in str(excinfo.value)
        assert "family1-variant1" in str(excinfo.value)

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_launch_params_with_non_power_of_two_gpus(
        self, mock_load_config, batch_model_configs
    ):
        """Test error when total GPUs is not a power of two."""
        # Need to add tensor parallelism to avoid the first validation error
        updated_configs = [
            batch_model_configs[0].model_copy(
                update={
                    "gpus_per_node": 3,
                    "vllm_args": {"--tensor-parallel-size": "3"},
                }
            ),
            batch_model_configs[1],
        ]
        mock_load_config.return_value = updated_configs

        with pytest.raises(ValueError) as excinfo:
            BatchModelLauncher(
                ["family1-variant1", "family2-variant1"],
                account="test-account",
                work_dir="/tmp/test-work",
            )

        assert "power of two" in str(excinfo.value)
        assert "family1-variant1" in str(excinfo.value)

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_launch_params_with_mismatched_batch_args(
        self, mock_load_config, batch_model_configs
    ):
        """Test error when batch mode required arguments don't match."""
        # Create a scenario where the batch argument validation actually fails
        # Let's use different GPU configurations that will pass all other validations
        # but fail on batch argument matching
        updated_configs = [
            batch_model_configs[0].model_copy(),
            batch_model_configs[1].model_copy(
                update={
                    "gpus_per_node": 1,
                    "num_nodes": 2,  # This will cause the mismatch
                    "vllm_args": {"--tensor-parallel-size": "1"},
                }
            ),
        ]
        mock_load_config.return_value = updated_configs

        with pytest.raises(ValueError) as excinfo:
            BatchModelLauncher(
                ["family1-variant1", "family2-variant1"],
                account="test-account",
                work_dir="/tmp/test-work",
            )

        assert "Mismatch between total number of GPUs requested" in str(excinfo.value)

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("vec_inf.client._helper.BatchSlurmScriptGenerator")
    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.open")
    @patch("pathlib.Path.rename")
    @patch("vec_inf.client._helper.copy2")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.write_text")
    def test_launch_success(
        self,
        mock_write_text,
        mock_read_text,
        mock_exists,
        mock_copy2,
        mock_rename,
        mock_open,
        mock_touch,
        mock_mkdir,
        mock_run_bash,
        mock_script_gen,
        mock_load_config,
        batch_model_configs,
    ):
        """Test successful batch model launch returns expected response."""
        mock_open.return_value = mock.mock_open().return_value
        mock_load_config.return_value = batch_model_configs
        mock_script_gen_instance = MagicMock()
        mock_script_gen_instance.generate_batch_slurm_script.return_value = Path(
            "/path/to/batch_slurm_script.sh"
        )
        mock_script_gen_instance.script_paths = [
            Path("/path/to/script1.sh"),
            Path("/path/to/script2.sh"),
        ]
        mock_script_gen.return_value = mock_script_gen_instance
        mock_run_bash.return_value = ("Submitted batch job 12345", "")
        # Mock that the script files exist and can be read/written
        mock_exists.return_value = True
        mock_read_text.return_value = "mock script content"
        mock_write_text.return_value = None
        # Mock copy2 to do nothing (avoid file operations)
        mock_copy2.return_value = None

        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1"],
            account="test-account",
            work_dir="/tmp/test-work",
        )
        response = launcher.launch()

        assert response.slurm_job_id == "12345"
        assert response.slurm_job_name == "BATCH-family1-variant1-family2-variant1"
        assert response.model_names == ["family1-variant1", "family2-variant1"]
        assert "slurm_job_id" in response.config
        assert response.config["slurm_job_id"] == "12345"

        # Verify log directories and files are created
        mock_mkdir.assert_called()
        mock_touch.assert_called()
        mock_copy2.assert_called()
        mock_rename.assert_called()

    @patch("vec_inf.client._helper.utils.load_config")
    @patch("vec_inf.client._helper.utils.run_bash_command")
    def test_launch_with_slurm_error(
        self, mock_run_bash, mock_load_config, batch_model_configs
    ):
        """Test launch raises error on SLURM submission failure."""
        mock_load_config.return_value = batch_model_configs
        mock_run_bash.return_value = ("", "sbatch: error: Invalid partition specified")

        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1"],
            account="test-account",
            work_dir="/tmp/test-work",
        )
        with pytest.raises(SlurmJobError):
            launcher.launch()

    @patch("vec_inf.client._helper.utils.load_config")
    def test_launch_params_het_group_ids(self, mock_load_config, batch_model_configs):
        """Test that heterogeneous group IDs are assigned correctly."""
        mock_load_config.return_value = batch_model_configs

        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1"],
            account="test-account",
            work_dir="/tmp/test-work",
        )
        params = launcher.params

        assert params["models"]["family1-variant1"]["het_group_id"] == 0
        assert params["models"]["family2-variant1"]["het_group_id"] == 1

    @patch("vec_inf.client._helper.utils.load_config")
    def test_launch_params_log_file_paths(self, mock_load_config, batch_model_configs):
        """Test that log file paths are constructed correctly."""
        mock_load_config.return_value = batch_model_configs

        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1"],
            account="test-account",
            work_dir="/tmp/test-work",
        )
        params = launcher.params

        # Check individual model log files
        assert (
            "family1-variant1.%j.out"
            in params["models"]["family1-variant1"]["out_file"]
        )
        assert (
            "family1-variant1.%j.err"
            in params["models"]["family1-variant1"]["err_file"]
        )
        assert (
            "family1-variant1.$SLURM_JOB_ID.json"
            in params["models"]["family1-variant1"]["json_file"]
        )

        assert (
            "family2-variant1.%j.out"
            in params["models"]["family2-variant1"]["out_file"]
        )
        assert (
            "family2-variant1.%j.err"
            in params["models"]["family2-variant1"]["err_file"]
        )
        assert (
            "family2-variant1.$SLURM_JOB_ID.json"
            in params["models"]["family2-variant1"]["json_file"]
        )

        # Check batch-level log files
        assert "BATCH-family1-variant1-family2-variant1.%j.out" in params["out_file"]
        assert "BATCH-family1-variant1-family2-variant1.%j.err" in params["err_file"]

    @patch("vec_inf.client._helper.utils.load_config")
    def test_init_with_batch_config(self, mock_load_config, batch_model_configs):
        """Test launcher initializes correctly with custom batch config."""
        mock_load_config.return_value = batch_model_configs

        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1"],
            batch_config="custom_config.yaml",
            account="test-account",
            work_dir="/tmp/test-work",
        )

        assert launcher.batch_config == "custom_config.yaml"
        # Verify load_config was called with the custom config
        mock_load_config.assert_called_with("custom_config.yaml")

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_launch_params_with_sglang_engine(
        self, mock_load_config, batch_model_configs
    ):
        """Test batch launch with SGLang engine."""
        updated_configs = [
            batch_model_configs[0].model_copy(
                update={
                    "engine": "sglang",
                    "sglang_args": {"--context-length": "8192"},
                }
            ),
            batch_model_configs[1],
        ]
        mock_load_config.return_value = updated_configs

        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1"],
            account="test-account",
            work_dir="/tmp/test-work",
        )
        params = launcher.params

        assert params["models"]["family1-variant1"]["engine"] == "sglang"
        assert params["models"]["family1-variant1"]["engine_args"] == {
            "--context-length": "8192"
        }
        assert params["models"]["family2-variant1"]["engine"] == "vllm"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_launch_params_engine_args_selection(
        self, mock_load_config, batch_model_configs
    ):
        """Test correct engine args selection per model."""
        updated_configs = [
            batch_model_configs[0].model_copy(
                update={
                    "engine": "sglang",
                    "sglang_args": {"--context-length": "8192"},
                }
            ),
            batch_model_configs[1].model_copy(
                update={
                    "engine": "vllm",
                    "vllm_args": {"--max-model-len": "4096"},
                }
            ),
        ]
        mock_load_config.return_value = updated_configs

        launcher = BatchModelLauncher(
            ["family1-variant1", "family2-variant1"],
            account="test-account",
            work_dir="/tmp/test-work",
        )
        params = launcher.params

        # Verify each model uses its correct engine args
        assert params["models"]["family1-variant1"]["engine_args"] == {
            "--context-length": "8192"
        }
        assert params["models"]["family2-variant1"]["engine_args"] == {
            "--max-model-len": "4096"
        }


class TestModelStatusMonitor:
    """Tests for the ModelStatusMonitor class."""

    @pytest.fixture
    def mock_scontrol_output(self):
        """Fixture returning mock `scontrol` output for a running SLURM job."""
        return "JobId=12345 JobName=test-model UserId=user(1000) GroupId=group(1000) MCS_label=N/A Priority=1 Nice=0 Account=account QOS=normal JobState=RUNNING Reason=None Dependency=(null) Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0 RunTime=00:10:00 TimeLimit=1-00:00:00 TimeMin=N/A SubmitTime=2023-01-01T12:00:00 EligibleTime=2023-01-01T12:00:00 AccrueTime=2023-01-01T12:00:00 StartTime=2023-01-01T12:01:00 EndTime=2023-01-02T12:01:00 Deadline=N/A PreemptEligibleTime=2023-01-01T12:01:00 PreemptTime=None SuspendTime=None SecsPreSuspend=0 LastSchedEval=2023-01-01T12:00:30 Partition=gpu AllocNode:Sid=login01:123 ReqNodeList=(null) ExcNodeList=(null) NodeList=gpu01 BatchHost=gpu01 NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 ReqB:S:C:T=0:0:*:* TRES=cpu=8,mem=32G,node=1,billing=8,gres/gpu=1 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=1 MinMemoryNode=32G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null) Command=/path/to/slurm_script.sh WorkDir=/home/user StdErr=/path/to/log/test-family/test-model.12345/test-model.12345.err StdIn=/dev/null StdOut=/path/to/log/test-family/test-model.12345/test-model.12345.out Power="

    @pytest.fixture
    def mock_pending_scontrol_output(self):
        """Fixture returning mock scontrol output for a pending job."""
        return "JobId=12345 JobName=test-model UserId=user(1000) GroupId=group(1000) MCS_label=N/A Priority=1 Nice=0 Account=account QOS=normal JobState=PENDING Reason=Resources NodeList=(null) BatchHost=gpu01 NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 ReqB:S:C:T=0:0:*:* TRES=cpu=8,mem=32G,node=1,billing=8,gres/gpu=1 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=1 MinMemoryNode=32G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null) Command=/path/to/slurm_script.sh WorkDir=/home/user StdErr=/path/to/log/test-family/test-model.12345/test-model.12345.err StdIn=/dev/null StdOut=/path/to/log/test-family/test-model.12345/test-model.12345.out Power="

    @patch("vec_inf.client._helper.utils.run_bash_command")
    def test_init_sets_fields_correctly(self, mock_run_bash, mock_scontrol_output):
        """Test __init__ sets job ID, output, and initial status info."""
        mock_run_bash.return_value = (mock_scontrol_output, "")

        monitor = ModelStatusMonitor(slurm_job_id="12345")

        assert monitor.slurm_job_id == "12345"
        assert monitor.output == mock_scontrol_output
        assert monitor.job_status["JobName"] == "test-model"
        assert monitor.job_status["JobState"] == "RUNNING"
        assert monitor.log_dir == "/path/to/log/test-family/test-model.12345"
        assert monitor.status_info.model_name == "test-model"
        assert monitor.status_info.job_state == "RUNNING"

    @patch("vec_inf.client._helper.utils.run_bash_command")
    def test_init_with_slurm_error(self, mock_run_bash):
        """Test __init__ raises error when SLURM command returns stderr."""
        mock_run_bash.return_value = ("", "scontrol: error: Invalid job id specified")

        with pytest.raises(SlurmJobError):
            ModelStatusMonitor("99999")

    @patch("vec_inf.client._helper.utils.run_bash_command")
    def test_process_pending_state(self, mock_run_bash, mock_pending_scontrol_output):
        """Test pending job sets server status and pending reason correctly."""
        mock_run_bash.return_value = (mock_pending_scontrol_output, "")

        monitor = ModelStatusMonitor(12345)
        status = monitor.process_model_status()

        assert status.server_status == ModelStatus.PENDING
        assert status.pending_reason == "Resources"

    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("vec_inf.client._helper.utils.is_server_running")
    @patch("vec_inf.client._helper.utils.model_health_check")
    @patch("vec_inf.client._helper.utils.get_base_url")
    def test_process_running_state_server_ready(
        self,
        mock_get_base_url,
        mock_health_check,
        mock_is_server_running,
        mock_run_bash,
        mock_scontrol_output,
    ):
        """Test READY status and base URL are set when server is healthy and running."""
        mock_run_bash.return_value = (mock_scontrol_output, "")
        mock_is_server_running.return_value = "RUNNING"
        mock_health_check.return_value = (ModelStatus.READY, 200)
        mock_get_base_url.return_value = "http://gpu01:8000/v1"

        monitor = ModelStatusMonitor(12345)
        status = monitor.process_model_status()

        assert status.server_status == ModelStatus.READY
        assert status.base_url == "http://gpu01:8000/v1"

    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("vec_inf.client._helper.utils.is_server_running")
    def test_process_running_state_server_failed(
        self, mock_is_server_running, mock_run_bash, mock_scontrol_output
    ):
        """Test that FAILED status and reason are set when server startup fails."""
        mock_run_bash.return_value = (mock_scontrol_output, "")
        mock_is_server_running.return_value = (
            ModelStatus.FAILED,
            "RuntimeError: CUDA out of memory",
        )

        monitor = ModelStatusMonitor(12345)
        status = monitor.process_model_status()

        assert status.server_status == ModelStatus.FAILED
        assert status.failed_reason == "RuntimeError: CUDA out of memory"

    @patch("vec_inf.client._helper.utils.run_bash_command")
    @patch("vec_inf.client._helper.utils.is_server_running")
    @patch("vec_inf.client._helper.utils.model_health_check")
    def test_process_running_state_health_check_failed(
        self,
        mock_health_check,
        mock_is_server_running,
        mock_run_bash,
        mock_scontrol_output,
    ):
        """Test FAILED status is set when health check fails with non-200 response."""
        mock_run_bash.return_value = (mock_scontrol_output, "")
        mock_is_server_running.return_value = "RUNNING"
        mock_health_check.return_value = (ModelStatus.FAILED, "500")

        monitor = ModelStatusMonitor(12345)
        status = monitor.process_model_status()

        assert status.server_status == ModelStatus.FAILED
        assert status.failed_reason == "500"


@pytest.fixture
def mock_status_response():
    """Fixture returning a mock StatusResponse instance."""
    return StatusResponse(
        model_name="test-model",
        log_dir="/tmp/test_logs",  # Add this line
        server_status=ModelStatus.UNAVAILABLE,
        job_state="RUNNING",
        raw_output="mock scontrol output",
        base_url="http://gpu01:8000/v1",
        pending_reason=None,
        failed_reason=None,
    )


class TestPerformanceMetricsCollector:
    """Unit tests for the PerformanceMetricsCollector class."""

    @pytest.fixture
    def mock_metrics_text(self):
        """Fixture returning a sample metrics text used for parsing tests."""
        return """\
vllm:prompt_tokens_total{engine="0",model_name="test-model"} 5000.0
vllm:generation_tokens_total{engine="0",model_name="test-model"} 1000.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="test-model"} 6.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="test-model"} 2.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="test-model"} 1.5
vllm:request_queue_time_seconds_count{engine="0",model_name="test-model"} 2.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="test-model"} 2.0
vllm:num_requests_running{engine="0",model_name="test-model"} 1.0
vllm:num_requests_waiting{engine="0",model_name="test-model"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="test-model"} 0.123
"""

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    def test_init(self, mock_status_monitor, mock_status_response):
        """Test __init__ sets default state and initializes from model status."""
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )

        collector = PerformanceMetricsCollector("12345")

        assert collector.slurm_job_id == "12345"
        assert collector._prev_prompt_tokens == 0.0
        assert collector._prev_generation_tokens == 0.0
        assert collector._last_updated is None

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    @patch("vec_inf.client._helper.utils.get_base_url")
    def test_build_metrics_url_pending(
        self, mock_get_base_url, mock_status_monitor, mock_status_response
    ):
        """Test metrics URL returns placeholder when job is pending."""
        mock_status_response.job_state = ModelStatus.PENDING
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )

        collector = PerformanceMetricsCollector("12345")

        assert collector.metrics_url == "Pending resources for server initialization"

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    @patch("vec_inf.client._helper.utils.get_base_url")
    def test_build_metrics_url_running(
        self, mock_get_base_url, mock_status_monitor, mock_status_response
    ):
        """Test metrics URL is built correctly when job is running."""
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )
        mock_get_base_url.return_value = "http://gpu01:8000/v1"

        collector = PerformanceMetricsCollector("12345")

        assert collector.metrics_url == "http://gpu01:8000/metrics"

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    def test_parse_metrics(
        self, mock_status_monitor, mock_metrics_text, mock_status_response
    ):
        """Test parsing of raw metrics text into structured dictionary values."""
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )

        collector = PerformanceMetricsCollector("12345")

        parsed = collector._parse_metrics(mock_metrics_text)

        assert parsed["total_prompt_tokens"] == 5000.0
        assert parsed["total_generation_tokens"] == 1000.0
        assert parsed["request_latency_sum"] == 6.0
        assert parsed["request_latency_count"] == 2.0
        assert parsed["requests_running"] == 1.0
        assert parsed["requests_waiting"] == 0.0

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    @patch("vec_inf.client._helper.requests.get")
    @patch("time.time")
    def test_fetch_metrics_second_call(
        self,
        mock_time,
        mock_requests_get,
        mock_status_monitor,
        mock_metrics_text,
        mock_status_response,
    ):
        """Test throughput calculation between two fetch_metrics calls."""
        mock_time.return_value = 1000.0
        mock_status_monitor.return_value.process_model_status.return_value = (
            mock_status_response
        )
        mock_response = MagicMock()
        mock_response.text = mock_metrics_text
        mock_requests_get.return_value = mock_response

        collector = PerformanceMetricsCollector("12345")
        collector.metrics_url = "http://gpu01:8000/metrics"
        collector.fetch_metrics()

        mock_time.return_value = 1006.0
        updated_metrics = """\
vllm:prompt_tokens_total{engine="0",model_name="test-model"} 5048.0
vllm:generation_tokens_total{engine="0",model_name="test-model"} 1012.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="test-model"} 6.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="test-model"} 2.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="test-model"} 1.5
vllm:request_queue_time_seconds_count{engine="0",model_name="test-model"} 2.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="test-model"} 2.0
vllm:num_requests_running{engine="0",model_name="test-model"} 1.0
vllm:num_requests_waiting{engine="0",model_name="test-model"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="test-model"} 0.123
"""
        mock_response.text = updated_metrics
        metrics = collector.fetch_metrics()

        assert metrics["prompt_tokens_per_sec"] == pytest.approx(
            (5048.0 - 5000.0) / 6.0
        )
        assert metrics["generation_tokens_per_sec"] == pytest.approx(
            (1012.0 - 1000.0) / 6.0
        )
        assert metrics["total_prompt_tokens"] == 5048.0
        assert metrics["total_generation_tokens"] == 1012.0
        assert metrics["request_latency_sum"] == 6.0
        assert metrics["request_latency_count"] == 2.0
        assert metrics["queue_time_sum"] == 1.5
        assert metrics["successful_requests_total"] == 2.0
        assert metrics["requests_running"] == 1.0
        assert metrics["requests_waiting"] == 0.0
        assert metrics["kv_cache_usage"] == 0.123
        assert metrics["avg_request_latency"] == pytest.approx(3.0)
        assert metrics["avg_queue_time"] == pytest.approx(0.75)

    @patch("vec_inf.client._helper.ModelStatusMonitor")
    @patch("vec_inf.client._helper.requests.get")
    def test_fetch_metrics_request_exception(
        self, mock_requests_get, mock_status_monitor
    ):
        """Test that fetch_metrics handles request exceptions."""
        mock_status = MagicMock()
        mock_status_monitor.return_value.process_model_status.return_value = mock_status
        mock_requests_get.side_effect = requests.RequestException("Connection refused")

        collector = PerformanceMetricsCollector("12345")
        collector.metrics_url = "http://gpu01:8000/metrics"

        result = collector.fetch_metrics()

        assert isinstance(result, str)
        assert "Metrics request failed" in result
        assert "Connection refused" in result


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestPrometheusMetricsParsing:
    """Parsing of real vLLM 0.32 and SGLang Prometheus exposition text."""

    @pytest.fixture
    def collector(self, mock_status_response):
        """Return a collector whose status lookup is mocked out."""
        with patch("vec_inf.client._helper.ModelStatusMonitor") as monitor:
            monitor.return_value.process_model_status.return_value = (
                mock_status_response
            )
            yield PerformanceMetricsCollector("12345")

    def test_parse_vllm_0_32_live_capture(self, collector):
        """Parse /metrics captured from a live vLLM 0.32.0 server.

        The fixture was scraped from Llama-3.2-1B-Instruct served by vec-inf on
        two A40 nodes (pipeline parallel 2) while 48 requests were in flight.
        """
        text = (FIXTURES_DIR / "vllm_0.32.0_metrics.txt").read_text()

        parsed = collector._parse_metrics(text)

        assert parsed["requests_running"] == 5.0
        assert parsed["requests_waiting"] == 0.0
        assert parsed["kv_cache_usage"] == pytest.approx(0.0001261901306698343)
        assert parsed["preemptions_total"] == 0.0
        assert parsed["total_prompt_tokens"] == 572.0
        assert parsed["total_generation_tokens"] == 217.0
        assert parsed["request_latency_sum"] == pytest.approx(6.5164635181427)
        assert parsed["request_latency_count"] == 8.0
        assert parsed["queue_time_count"] == 8.0
        # stop (1) + length (7); abort, error and repetition are all 0 here
        assert parsed["successful_requests_total"] == 8.0
        assert parsed["prefix_cache_queries"] == 1307.0
        assert parsed["prefix_cache_hits"] == 960.0
        assert parsed["prefix_cache_hit_rate"] == pytest.approx(960.0 / 1307.0)
        # The removed V0 metrics are not in the output, so they are not reported
        assert "gpu_cache_usage" not in parsed
        assert "requests_swapped" not in parsed

    def test_parse_sglang_0_5_12_live_capture(self, collector):
        """Parse /metrics captured from a live SGLang 0.5.12.post1 server.

        The fixture was scraped from Qwen2.5-0.5B-Instruct on one A40, started
        with --enable-metrics, after a health check, one chat and five
        completion requests.
        """
        text = (FIXTURES_DIR / "sglang_0.5.12.post1_metrics.txt").read_text()

        parsed = collector._parse_metrics(text)

        assert parsed["total_prompt_tokens"] == 65.0
        assert parsed["total_generation_tokens"] == 183.0
        assert parsed["finished_requests_total"] == 7.0
        assert "successful_requests_total" not in parsed
        assert parsed["request_latency_count"] == 7.0
        assert parsed["queue_time_count"] == 8.0
        assert parsed["requests_running"] == 1.0
        assert parsed["requests_waiting"] == 0.0
        assert parsed["kv_cache_usage"] == 0.0
        assert parsed["prefix_cache_hit_rate"] == 0.0

    def test_parse_sglang_per_rank_gauges(self, collector):
        """SGLang scheduler gauges repeated per TP/PP rank are counted once.

        With pipeline parallelism (or --enable-metrics-for-all-schedulers) each
        rank reports the same running and waiting requests. Data parallel ranks
        serve different requests and are summed.
        """
        text = """\
sglang:num_running_reqs{dp_rank="0",model_name="m",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 4.0
sglang:num_running_reqs{dp_rank="0",model_name="m",moe_ep_rank="0",pp_rank="1",tp_rank="0"} 4.0
sglang:num_running_reqs{dp_rank="0",model_name="m",moe_ep_rank="0",pp_rank="0",tp_rank="1"} 4.0
sglang:num_running_reqs{dp_rank="1",model_name="m",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 3.0
sglang:num_running_reqs{dp_rank="1",model_name="m",moe_ep_rank="0",pp_rank="1",tp_rank="0"} 3.0
sglang:num_queue_reqs{model_name="m",pp_rank="0",tp_rank="0"} 2.0
sglang:num_queue_reqs{model_name="m",pp_rank="1",tp_rank="0"} 2.0
sglang:token_usage{model_name="m",pp_rank="0",tp_rank="0"} 0.5
sglang:token_usage{model_name="m",pp_rank="1",tp_rank="0"} 0.1
sglang:num_requests_total{model_name="m"} 9.0
"""
        parsed = collector._parse_metrics(text)

        assert parsed["requests_running"] == 7.0
        assert parsed["requests_waiting"] == 2.0
        assert parsed["kv_cache_usage"] == 0.5
        assert parsed["finished_requests_total"] == 9.0

    def test_parse_sglang_dp_attention_ranks(self, collector):
        """With --enable-dp-attention each DP group reports from its own rank.

        Here TP=4 is split into two DP attention groups; group 0 reports from
        tp_rank 0 and group 1 from tp_rank 2. Both groups must be counted, and
        a group that reports from several ranks must be counted once.
        """
        text = """\
sglang:num_running_reqs{dp_rank="0",model_name="m",pp_rank="0",tp_rank="0"} 5.0
sglang:num_running_reqs{dp_rank="0",model_name="m",pp_rank="0",tp_rank="1"} 5.0
sglang:num_running_reqs{dp_rank="1",model_name="m",pp_rank="0",tp_rank="2"} 3.0
sglang:num_running_reqs{dp_rank="1",model_name="m",pp_rank="0",tp_rank="3"} 3.0
sglang:num_queue_reqs{dp_rank="0",model_name="m",pp_rank="0",tp_rank="0"} 1.0
sglang:num_queue_reqs{dp_rank="1",model_name="m",pp_rank="0",tp_rank="2"} 2.0
sglang:token_usage{dp_rank="0",model_name="m",pp_rank="0",tp_rank="0"} 0.2
sglang:token_usage{dp_rank="1",model_name="m",pp_rank="0",tp_rank="2"} 0.4
"""
        parsed = collector._parse_metrics(text)

        assert parsed["requests_running"] == 8.0
        assert parsed["requests_waiting"] == 3.0
        assert parsed["kv_cache_usage"] == pytest.approx(0.3)

    def test_parse_vllm_data_parallel_engines(self, collector):
        """Samples from several engines are summed, KV cache usage is averaged."""
        text = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="1",model_name="m"} 2.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:num_requests_waiting{engine="1",model_name="m"} 4.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.2
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.6
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{engine="0",model_name="m"} 100.0
vllm:prompt_tokens_total{engine="1",model_name="m"} 50.0
vllm:prompt_tokens_created{engine="0",model_name="m"} 1.7929e+09
vllm:num_preemptions_total{engine="0",model_name="m"} 1.0
vllm:num_preemptions_total{engine="1",model_name="m"} 2.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 100.0
vllm:prefix_cache_queries_total{engine="1",model_name="m"} 300.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 50.0
vllm:prefix_cache_hits_total{engine="1",model_name="m"} 50.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 5.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 2.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="m"} 7.0
vllm:request_success_total{engine="1",finished_reason="error",model_name="m"} 9.0
vllm:request_success_total{engine="1",finished_reason="repetition",model_name="m"} 1.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="m"} 4.0 1792949107000
vllm:e2e_request_latency_seconds_count{engine="0",model_name="m"} 2.0
vllm:e2e_request_latency_seconds_sum{engine="1",model_name="m"} 2.0
vllm:e2e_request_latency_seconds_count{engine="1",model_name="m"} 2.0
"""
        parsed = collector._parse_metrics(text)

        assert parsed["requests_running"] == 5.0
        assert parsed["requests_waiting"] == 5.0
        assert parsed["kv_cache_usage"] == pytest.approx(0.4)
        assert parsed["total_prompt_tokens"] == 150.0
        assert parsed["preemptions_total"] == 3.0
        assert parsed["prefix_cache_hit_rate"] == pytest.approx(100.0 / 400.0)
        # abort and error are excluded, repetition counts as finished
        assert parsed["successful_requests_total"] == 8.0
        # trailing exposition timestamps are ignored
        assert parsed["request_latency_sum"] == 6.0
        assert parsed["request_latency_count"] == 4.0

    def test_parse_prefix_cache_without_queries(self, collector):
        """A fresh server reports 0 queries; the hit rate is 0, not a crash."""
        text = """\
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 0.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 0.0
"""
        parsed = collector._parse_metrics(text)

        assert parsed["prefix_cache_hit_rate"] == 0.0

    def test_parse_label_values_with_spaces_and_quotes(self, collector):
        """Label values may contain spaces, commas and escaped quotes."""
        text = (
            'vllm:request_success_total{engine="0",finished_reason="abort",'
            'model_name="my model, \\"v2\\""} 3.0\n'
            'vllm:request_success_total{engine="0",finished_reason="stop",'
            'model_name="my model, \\"v2\\""} 4.0\n'
            'vllm:num_requests_running{engine="0",model_name="a b"} 2.0\n'
        )
        parsed = collector._parse_metrics(text)

        assert parsed["successful_requests_total"] == 4.0
        assert parsed["requests_running"] == 2.0

    def test_parse_sglang_metrics(self, collector):
        """Parse SGLang metrics (names from SGLang 0.5.12.post1)."""
        text = """\
# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="m",pp_rank="0",tp_rank="0"} 3.0
sglang:num_queue_reqs{model_name="m",pp_rank="0",tp_rank="0"} 2.0
sglang:token_usage{model_name="m",pp_rank="0",tp_rank="0"} 0.25
sglang:cache_hit_rate{model_name="m",pp_rank="0",tp_rank="0"} 0.5
sglang:num_retracted_requests_total{model_name="m",pp_rank="0",tp_rank="0"} 1.0
sglang:prompt_tokens_total{model_name="m"} 1000.0
sglang:generation_tokens_total{model_name="m"} 400.0
sglang:num_requests_total{model_name="m"} 10.0
sglang:e2e_request_latency_seconds_sum{model_name="m"} 5.0
sglang:e2e_request_latency_seconds_count{model_name="m"} 10.0
sglang:queue_time_seconds_sum{model_name="m"} 0.5
sglang:queue_time_seconds_count{model_name="m"} 10.0
"""
        parsed = collector._parse_metrics(text)

        assert parsed["requests_running"] == 3.0
        assert parsed["requests_waiting"] == 2.0
        assert parsed["kv_cache_usage"] == 0.25
        assert parsed["prefix_cache_hit_rate"] == 0.5
        assert parsed["preemptions_total"] == 1.0
        assert parsed["total_prompt_tokens"] == 1000.0
        assert parsed["total_generation_tokens"] == 400.0
        assert parsed["finished_requests_total"] == 10.0
        assert parsed["request_latency_sum"] == 5.0
        assert parsed["queue_time_sum"] == 0.5


class TestModelRegistry:
    """Unit tests for the ModelRegistry class."""

    @pytest.fixture
    def mock_configs(self):
        """Return a list of mock ModelConfig for testing."""
        return [
            ModelConfig(
                model_name="model1",
                model_family="family1",
                model_variant="variant1",
                model_type="LLM",
                gpus_per_node=1,
                num_nodes=1,
                vocab_size=32000,
                model_weights_parent_dir=Path("/path/to/models"),
                resource_type="l40s",
                partition="gpu",
                qos="normal",
                time="01:00:00",
                cpus_per_task=8,
                mem_per_node="32G",
                account="test-account",
                work_dir="/tmp/test-work",
            ),
            ModelConfig(
                model_name="model2",
                model_family="family2",
                model_variant="variant2",
                model_type="VLM",
                gpus_per_node=2,
                num_nodes=1,
                vocab_size=32000,
                model_weights_parent_dir=Path("/path/to/models"),
                resource_type="l40s",
                partition="gpu",
                qos="normal",
                time="01:00:00",
                cpus_per_task=8,
                mem_per_node="32G",
                account="test-account",
                work_dir="/tmp/test-work",
            ),
        ]

    @patch("vec_inf.client._helper.utils.load_config")
    def test_init(self, mock_load_config, mock_configs):
        """Test ModelRegistry init."""
        mock_load_config.return_value = mock_configs

        registry = ModelRegistry()

        assert registry.model_configs[0].model_name == "model1"
        assert registry.model_configs[1].model_name == "model2"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_all_models(self, mock_load_config, mock_configs):
        """Test get_all_models returns correct model info."""
        mock_load_config.return_value = mock_configs

        registry = ModelRegistry()
        models = registry.get_all_models()

        assert len(models) == 2
        assert models[0].name == "model1"
        assert models[0].family == "family1"
        assert models[0].variant == "variant1"
        assert models[0].model_type == ModelType.LLM

        assert models[1].name == "model2"
        assert models[1].family == "family2"
        assert models[1].variant == "variant2"
        assert models[1].model_type == ModelType.VLM

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_single_model_config(self, mock_load_config, mock_configs):
        """Test retrieving the config for an existing model."""
        mock_load_config.return_value = mock_configs

        registry = ModelRegistry()
        config = registry.get_single_model_config("model1")

        assert config.model_name == "model1"
        assert config.model_family == "family1"
        assert config.model_variant == "variant1"

    @patch("vec_inf.client._helper.utils.load_config")
    def test_get_single_model_config_not_found(self, mock_load_config, mock_configs):
        """Test retrieving a config for a non-existent model."""
        mock_load_config.return_value = mock_configs

        registry = ModelRegistry()

        with pytest.raises(ModelNotFoundError):
            registry.get_single_model_config("nonexistent_model")
