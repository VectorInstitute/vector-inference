"""Unit tests for slurm script generator component in the vec_inf.client module."""

import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from vec_inf.client._slurm_script_generator import (
    BatchSlurmScriptGenerator,
    SlurmScriptGenerator,
)


class TestSlurmScriptGenerator:
    """Tests for SlurmScriptGenerator class."""

    @pytest.fixture
    def basic_params(self):
        """Generate basic SLURM configuration parameters."""
        return {
            "model_name": "test-model",
            "model_weights_parent_dir": "/path/to/model_weights",
            "src_dir": "/path/to/src",
            "log_dir": "/path/to/logs",
            "num_nodes": "1",
            "venv": "/path/to/venv",
            "gpus_per_node": "4",
            "partition": "gpu",
            "account": "test-account",
            "time": "01:00:00",
            "engine": "vllm",
            "engine_args": {
                "--tensor-parallel-size": "4",
                "--max-model-len": "8192",
                "--enforce-eager": True,
            },
        }

    @pytest.fixture
    def multinode_params(self, basic_params):
        """Generate multi-node SLURM configuration parameters."""
        multinode = basic_params.copy()
        multinode.update(
            {
                "num_nodes": "2",
            }
        )
        return multinode

    @pytest.fixture
    def singularity_params(self, basic_params):
        """Generate singularity-based SLURM configuration parameters."""
        singularity = basic_params.copy()
        singularity.update(
            {
                "venv": "apptainer",
                "bind": "/scratch:/scratch,/data:/data",
                "env": {
                    "CACHE_DIR": "/cache",
                    "MY_VAR": "5",
                    "VLLM_CACHE_ROOT": "/cache/vllm",
                },
            }
        )
        return singularity

    @pytest.fixture
    def temp_log_dir(self):
        """Generate temporary directory for log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_init_single_node(self, basic_params):
        """Test initialization with single-node configuration."""
        generator = SlurmScriptGenerator(basic_params)

        assert generator.params == basic_params
        assert not generator.is_multinode
        assert not generator.use_container
        assert generator.additional_binds == ""
        assert generator.model_weights_path == "/path/to/model_weights/test-model"
        assert generator.env_str == ""
        assert generator.engine == "vllm"

    def test_init_with_sglang_engine(self, basic_params):
        """Test initialization with SGLang engine."""
        sglang_params = basic_params.copy()
        sglang_params["engine"] = "sglang"
        sglang_params["engine_args"] = {
            "--context-length": "8192",
            "--tensor-parallel-size": "4",
        }

        generator = SlurmScriptGenerator(sglang_params)

        assert generator.engine == "sglang"
        assert generator.params["engine"] == "sglang"

    def test_init_multinode(self, multinode_params):
        """Test initialization with multi-node configuration."""
        generator = SlurmScriptGenerator(multinode_params)

        assert generator.params == multinode_params
        assert generator.is_multinode
        assert not generator.use_container
        assert generator.additional_binds == ""
        assert generator.model_weights_path == "/path/to/model_weights/test-model"
        assert generator.env_str == ""

    def test_init_singularity(self, singularity_params):
        """Test initialization with Singularity configuration."""
        generator = SlurmScriptGenerator(singularity_params)

        assert generator.params == singularity_params
        assert generator.use_container
        assert not generator.is_multinode
        assert generator.additional_binds == ",/scratch:/scratch,/data:/data"
        assert generator.model_weights_path == "/path/to/model_weights/test-model"
        assert (
            generator.env_str
            == "--env CACHE_DIR=/cache,MY_VAR=5,VLLM_CACHE_ROOT=/cache/vllm"
        )

    def test_init_singularity_no_bind(self, basic_params):
        """Test Singularity initialization without additional binds."""
        params = basic_params.copy()
        params["venv"] = "apptainer"
        generator = SlurmScriptGenerator(params)

        assert generator.params == params
        assert generator.use_container
        assert not generator.is_multinode
        assert generator.additional_binds == ""
        assert generator.model_weights_path == "/path/to/model_weights/test-model"
        assert generator.env_str == ""

    def test_generate_shebang_single_node(self, basic_params):
        """Test shebang generation for single-node setup."""
        generator = SlurmScriptGenerator(basic_params)
        shebang = generator._generate_shebang()

        assert shebang.startswith("#!/bin/bash")
        assert "#SBATCH --job-name=test-model" in shebang
        assert "#SBATCH --partition=gpu" in shebang
        assert "#SBATCH --nodes=1" in shebang
        assert "#SBATCH --exclusive" not in shebang

    def test_generate_shebang_multinode(self, multinode_params):
        """Test shebang generation for multi-node setup."""
        generator = SlurmScriptGenerator(multinode_params)
        shebang = generator._generate_shebang()

        assert "#SBATCH --nodes=2" in shebang
        assert "#SBATCH --ntasks-per-node=1" in shebang
        # Note: --exclusive may not always be present depending on template

    def test_generate_server_setup_single_node(self, basic_params):
        """Test server setup generation for single-node."""
        generator = SlurmScriptGenerator(basic_params)
        setup = generator._generate_server_setup()

        assert "source /path/to/src/find_port.sh" in setup
        assert "ray start --head" not in setup
        # Check for port finding logic
        assert "find_available_port" in setup or "server_port" in setup.lower()

    def test_generate_server_setup_multinode(self, multinode_params):
        """Test server setup generation for multi-node."""
        generator = SlurmScriptGenerator(multinode_params)
        setup = generator._generate_server_setup()

        assert "ray start --head" in setup
        assert "ray start --address" in setup
        assert "scontrol show hostnames" in setup
        assert "worker_num=$((SLURM_JOB_NUM_NODES - 1))" in setup

    def test_generate_server_setup_singularity(self, singularity_params):
        """Test server setup with Singularity container."""
        generator = SlurmScriptGenerator(singularity_params)
        setup = generator._generate_server_setup()

        # For single node, ray stop may not be present
        # Just check for container setup
        assert (
            "module load " in setup or "apptainer" in setup.lower()
        )  # Remove module name since it's inconsistent between clusters

    def test_generate_launch_cmd_venv(self, basic_params):
        """Test launch command generation with virtual environment."""
        generator = SlurmScriptGenerator(basic_params)
        launch_cmd = generator._generate_launch_cmd()

        assert "vllm serve /path/to/model_weights/test-model" in launch_cmd
        assert "--served-model-name test-model" in launch_cmd
        assert "--tensor-parallel-size 4" in launch_cmd
        assert "--max-model-len 8192" in launch_cmd
        assert "--enforce-eager" in launch_cmd

    def test_generate_launch_cmd_sglang(self, basic_params):
        """Test SGLang launch command generation."""
        sglang_params = basic_params.copy()
        sglang_params["engine"] = "sglang"
        sglang_params["engine_args"] = {
            "--context-length": "8192",
            "--tensor-parallel-size": "4",
            "--mem-fraction-static": "0.85",
        }

        generator = SlurmScriptGenerator(sglang_params)
        launch_cmd = generator._generate_launch_cmd()

        assert "sglang.launch_server" in launch_cmd
        assert "/path/to/model_weights/test-model" in launch_cmd
        assert "--context-length 8192" in launch_cmd
        assert "--tensor-parallel-size 4" in launch_cmd
        assert "--mem-fraction-static 0.85" in launch_cmd

    def test_generate_launch_cmd_sglang_multinode(self, basic_params):
        """Test multi-node SGLang launch command generation."""
        sglang_params = basic_params.copy()
        sglang_params["engine"] = "sglang"
        sglang_params["num_nodes"] = "2"
        sglang_params["engine_args"] = {
            "--context-length": "8192",
            "--tensor-parallel-size": "4",
        }

        generator = SlurmScriptGenerator(sglang_params)
        launch_cmd = generator._generate_launch_cmd()

        assert "sglang.launch_server" in launch_cmd
        assert "--context-length 8192" in launch_cmd
        assert "--tensor-parallel-size 4" in launch_cmd

    def test_generate_server_setup_sglang_multinode(self, basic_params):
        """Test SGLang multi-node setup."""
        sglang_params = basic_params.copy()
        sglang_params["engine"] = "sglang"
        sglang_params["num_nodes"] = "2"

        generator = SlurmScriptGenerator(sglang_params)
        setup = generator._generate_server_setup()

        # SGLang multi-node setup should not include Ray
        assert "ray start --head" not in setup
        # Check for NCCL setup which is used for SGLang multi-node
        assert "nccl" in setup.lower() or "NCCL" in setup

    def test_generate_launch_cmd_sglang_with_args(self, basic_params):
        """Test SGLang launch command with various arguments."""
        sglang_params = basic_params.copy()
        sglang_params["engine"] = "sglang"
        sglang_params["engine_args"] = {
            "--context-length": "16384",
            "--tensor-parallel-size": "8",
            "--mem-fraction-static": "0.9",
            "--trust-remote-code": True,
        }

        generator = SlurmScriptGenerator(sglang_params)
        launch_cmd = generator._generate_launch_cmd()

        assert "--context-length 16384" in launch_cmd
        assert "--tensor-parallel-size 8" in launch_cmd
        assert "--mem-fraction-static 0.9" in launch_cmd
        assert "--trust-remote-code" in launch_cmd

    def test_generate_launch_cmd_sglang_boolean_args(self, basic_params):
        """Test boolean arguments for SGLang."""
        sglang_params = basic_params.copy()
        sglang_params["engine"] = "sglang"
        sglang_params["engine_args"] = {
            "--trust-remote-code": True,
            "--disable-log-stats": True,
            "--context-length": "8192",
        }

        generator = SlurmScriptGenerator(sglang_params)
        launch_cmd = generator._generate_launch_cmd()

        assert "--trust-remote-code" in launch_cmd
        assert "--disable-log-stats" in launch_cmd
        assert "--context-length 8192" in launch_cmd

    def test_generate_launch_cmd_sglang_singularity(self, singularity_params):
        """Test SGLang launch command with Singularity container."""
        sglang_params = singularity_params.copy()
        sglang_params["engine"] = "sglang"
        sglang_params["engine_args"] = {
            "--context-length": "8192",
            "--tensor-parallel-size": "4",
        }

        generator = SlurmScriptGenerator(sglang_params)
        launch_cmd = generator._generate_launch_cmd()

        assert "apptainer exec --nv" in launch_cmd
        assert "sglang.launch_server" in launch_cmd
        assert "source" not in launch_cmd

    def test_generate_script_content_sglang(self, basic_params):
        """Test complete SGLang script generation."""
        sglang_params = basic_params.copy()
        sglang_params["engine"] = "sglang"
        sglang_params["engine_args"] = {
            "--context-length": "8192",
        }

        generator = SlurmScriptGenerator(sglang_params)
        content = generator._generate_script_content()

        assert content.startswith("#!/bin/bash")
        assert "sglang.launch_server" in content
        assert "find_available_port" in content

    def test_generate_launch_cmd_singularity(self, singularity_params):
        """Test launch command generation with Singularity."""
        generator = SlurmScriptGenerator(singularity_params)
        launch_cmd = generator._generate_launch_cmd()

        assert "apptainer exec --nv" in launch_cmd
        assert "source" not in launch_cmd

    def test_generate_launch_cmd_boolean_args(self, basic_params):
        """Test launch command with boolean vLLM arguments."""
        params = basic_params.copy()
        params["engine_args"] = {
            "--trust-remote-code": True,
            "--disable-log-stats": True,
            "--tensor-parallel-size": "2",
        }

        generator = SlurmScriptGenerator(params)
        launch_cmd = generator._generate_launch_cmd()

        assert "--trust-remote-code" in launch_cmd
        assert "--disable-log-stats" in launch_cmd
        assert "--tensor-parallel-size 2" in launch_cmd

    @patch("builtins.open", new_callable=mock_open)
    @patch("vec_inf.client._slurm_script_generator.datetime")
    def test_write_to_log_dir(
        self, mock_datetime, mock_file, basic_params, temp_log_dir
    ):
        """Test writing SLURM script to log directory."""
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        params = basic_params.copy()
        params["log_dir"] = str(temp_log_dir)

        generator = SlurmScriptGenerator(params)

        with patch.object(Path, "write_text") as mock_write:
            script_path = generator.write_to_log_dir()

            expected_path = (
                temp_log_dir / "launch_test-model_20240101_120000.sbatch"
            )  # Changed from .slurm to .sbatch
            assert script_path == expected_path
            mock_write.assert_called_once()

    def test_generate_script_content_integration(self, basic_params):
        """Test complete script generation integration."""
        generator = SlurmScriptGenerator(basic_params)
        content = generator._generate_script_content()

        assert content.startswith("#!/bin/bash")
        assert "vllm serve" in content
        assert "find_available_port" in content
        assert "source /path/to/venv/bin/activate" in content


class TestBatchSlurmScriptGenerator:
    """Tests for BatchSlurmScriptGenerator class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Generate temporary directory for log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def batch_params(self):
        """Generate batch SLURM configuration parameters."""
        return {
            "models": {
                "model1": {
                    "model_name": "model1",
                    "model_weights_parent_dir": "/path/to/model_weights",
                    "src_dir": "/path/to/src",
                    "log_dir": "/path/to/logs",
                    "num_nodes": "1",
                    "venv": "/path/to/venv",
                    "gpus_per_node": "4",
                    "partition": "gpu",
                    "account": "test-account",
                    "time": "01:00:00",
                    "engine": "vllm",
                    "engine_args": {
                        "--tensor-parallel-size": "4",
                        "--max-model-len": "8192",
                        "--enforce-eager": True,
                    },
                    "het_group_id": 0,
                    "out_file": "/path/to/logs/model1.%j.out",
                    "err_file": "/path/to/logs/model1.%j.err",
                },
                "model2": {
                    "model_name": "model2",
                    "model_weights_parent_dir": "/path/to/model_weights",
                    "src_dir": "/path/to/src",
                    "log_dir": "/path/to/logs",
                    "num_nodes": "1",
                    "venv": "/path/to/venv",
                    "gpus_per_node": "2",
                    "partition": "gpu",
                    "account": "test-account",
                    "time": "01:00:00",
                    "engine": "vllm",
                    "engine_args": {
                        "--tensor-parallel-size": "2",
                        "--max-model-len": "4096",
                        "--disable-log-stats": True,
                    },
                    "het_group_id": 1,
                    "out_file": "/path/to/logs/model2.%j.out",
                    "err_file": "/path/to/logs/model2.%j.err",
                },
            },
            "slurm_job_name": "BATCH-model1-model2",
            "src_dir": "/path/to/src",
            "out_file": "/path/to/logs/BATCH-model1-model2.%j.out",
            "err_file": "/path/to/logs/BATCH-model1-model2.%j.err",
            "log_dir": "/path/to/logs",
            "venv": "/path/to/venv",
        }

    @pytest.fixture
    def batch_singularity_params(self, batch_params):
        """Generate batch SLURM configuration parameters with Singularity."""
        singularity_params = batch_params.copy()
        singularity_params["venv"] = "apptainer"  # Set top-level venv to apptainer
        for model_name in singularity_params["models"]:
            singularity_params["models"][model_name]["venv"] = "apptainer"
            singularity_params["models"][model_name]["bind"] = (
                "/scratch:/scratch,/data:/data"
            )
        return singularity_params

    def test_init_basic(self, batch_params):
        """Test initialization with basic configuration."""
        generator = BatchSlurmScriptGenerator(batch_params)

        assert generator.params == batch_params
        assert not generator.use_container
        assert len(generator.script_paths) == 0
        assert "model1" in generator.params["models"]
        assert "model2" in generator.params["models"]

    def test_init_singularity(self, batch_singularity_params):
        """Test initialization with Singularity configuration."""
        generator = BatchSlurmScriptGenerator(batch_singularity_params)

        print(generator.params["models"]["model1"]["additional_binds"])
        print(generator.params["models"]["model2"]["additional_binds"])

        assert generator.use_container
        assert (
            generator.params["models"]["model1"]["additional_binds"]
            == ",/scratch:/scratch,/data:/data"
        )
        assert (
            generator.params["models"]["model2"]["additional_binds"]
            == ",/scratch:/scratch,/data:/data"
        )

    def test_init_singularity_no_bind(self, batch_params):
        """Test Singularity initialization without additional binds."""
        params = batch_params.copy()
        params["venv"] = "apptainer"  # Set top-level venv to apptainer
        for model_name in params["models"]:
            params["models"][model_name]["venv"] = "apptainer"

        generator = BatchSlurmScriptGenerator(params)

        assert generator.use_container
        assert generator.params["models"]["model1"]["additional_binds"] == ""
        assert generator.params["models"]["model2"]["additional_binds"] == ""

    def test_generate_batch_slurm_script_shebang(self, batch_params):
        """Test shebang generation for batch mode."""
        generator = BatchSlurmScriptGenerator(batch_params)
        shebang = generator._generate_batch_slurm_script_shebang()

        assert "#SBATCH --output=/path/to/logs/BATCH-model1-model2.%j.out" in shebang
        assert "#SBATCH --error=/path/to/logs/BATCH-model1-model2.%j.err" in shebang
        assert "# ===== Resource group for model1 =====" in shebang
        assert "# ===== Resource group for model2 =====" in shebang
        assert "#SBATCH hetjob" in shebang

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    def test_generate_model_launch_script_basic(
        self, mock_write_text, mock_touch, batch_params
    ):
        """Test generation of individual model launch scripts."""
        generator = BatchSlurmScriptGenerator(batch_params)
        script_path = generator._generate_model_launch_script("model1")

        assert script_path.name == "launch_model1.sh"
        assert len(generator.script_paths) == 1
        assert generator.script_paths[0] == script_path
        mock_touch.assert_called_once()
        mock_write_text.assert_called_once()

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    def test_generate_model_launch_script_singularity(
        self, mock_write_text, mock_touch, batch_singularity_params
    ):
        """Test generation of individual model launch scripts with Singularity."""
        generator = BatchSlurmScriptGenerator(batch_singularity_params)
        script_path = generator._generate_model_launch_script("model1")

        assert script_path.name == "launch_model1.sh"
        assert len(generator.script_paths) == 1
        mock_touch.assert_called_once()
        mock_write_text.assert_called_once()

    @patch("vec_inf.client._slurm_script_generator.datetime")
    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    def test_generate_batch_slurm_script(
        self, mock_write_text, mock_touch, mock_datetime, batch_params, temp_log_dir
    ):
        """Test complete batch SLURM script generation."""
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        params = batch_params.copy()
        params["log_dir"] = str(temp_log_dir)

        generator = BatchSlurmScriptGenerator(params)

        script_path = generator.generate_batch_slurm_script()

        expected_path = (
            temp_log_dir / "BATCH-model1-model2_20240101_120000.sbatch"
        )  # Changed from .slurm to .sbatch
        assert script_path == expected_path
        assert len(generator.script_paths) == 2
        assert any("launch_model1.sh" in str(p) for p in generator.script_paths)
        assert any("launch_model2.sh" in str(p) for p in generator.script_paths)
        mock_touch.assert_called()
        mock_write_text.assert_called()

    @patch("vec_inf.client._slurm_script_generator.datetime")
    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    def test_generate_batch_slurm_script_singularity(
        self,
        mock_write_text,
        mock_touch,
        mock_datetime,
        batch_singularity_params,
        temp_log_dir,
    ):
        """Test complete batch SLURM script generation with Singularity."""
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        params = batch_singularity_params.copy()
        params["log_dir"] = str(temp_log_dir)

        generator = BatchSlurmScriptGenerator(params)

        script_path = generator.generate_batch_slurm_script()

        expected_path = (
            temp_log_dir / "BATCH-model1-model2_20240101_120000.sbatch"
        )  # Changed from .slurm to .sbatch
        assert script_path == expected_path
        assert len(generator.script_paths) == 2
        mock_touch.assert_called()
        mock_write_text.assert_called()

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    def test_write_to_log_dir(
        self, mock_write_text, mock_touch, batch_params, temp_log_dir
    ):
        """Test writing script to log directory."""
        params = batch_params.copy()
        params["log_dir"] = str(temp_log_dir)

        generator = BatchSlurmScriptGenerator(params)

        script_path = generator._write_to_log_dir(
            ["#!/bin/bash", "echo 'test'"], "test_script.sh"
        )

        expected_path = temp_log_dir / "test_script.sh"
        assert script_path == expected_path
        mock_touch.assert_called_once()
        mock_write_text.assert_called_once_with("#!/bin/bash\necho 'test'")

    def test_model_weights_path_construction(self, batch_params):
        """Test model weights path construction."""
        generator = BatchSlurmScriptGenerator(batch_params)

        assert (
            generator.params["models"]["model1"]["model_weights_path"]
            == "/path/to/model_weights/model1"
        )
        assert (
            generator.params["models"]["model2"]["model_weights_path"]
            == "/path/to/model_weights/model2"
        )

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    def test_multiple_models_script_generation(
        self, mock_write_text, mock_touch, batch_params
    ):
        """Test generation of scripts for multiple models."""
        generator = BatchSlurmScriptGenerator(batch_params)

        # Generate scripts for both models
        script1_path = generator._generate_model_launch_script("model1")
        script2_path = generator._generate_model_launch_script("model2")

        assert len(generator.script_paths) == 2
        assert script1_path in generator.script_paths
        assert script2_path in generator.script_paths
        assert script1_path.name == "launch_model1.sh"
        assert script2_path.name == "launch_model2.sh"

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    def test_generate_model_launch_script_sglang(
        self, mock_write_text, mock_touch, batch_params
    ):
        """Test generation of batch launch script for SGLang."""
        sglang_params = batch_params.copy()
        sglang_params["models"]["model1"]["engine"] = "sglang"
        sglang_params["models"]["model1"]["engine_args"] = {
            "--context-length": "8192",
            "--tensor-parallel-size": "4",
        }

        generator = BatchSlurmScriptGenerator(sglang_params)
        script_path = generator._generate_model_launch_script("model1")

        assert script_path.name == "launch_model1.sh"
        # Verify the script content includes SGLang command
        call_args = mock_write_text.call_args[0][0]
        assert "sglang.launch_server" in call_args

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    def test_generate_batch_slurm_script_mixed_engines(
        self, mock_write_text, mock_touch, batch_params
    ):
        """Test batch launch script with different engines per model."""
        mixed_params = batch_params.copy()
        mixed_params["models"]["model1"]["engine"] = "sglang"
        mixed_params["models"]["model1"]["engine_args"] = {
            "--context-length": "8192",
        }
        mixed_params["models"]["model2"]["engine"] = "vllm"
        mixed_params["models"]["model2"]["engine_args"] = {
            "--max-model-len": "4096",
        }

        generator = BatchSlurmScriptGenerator(mixed_params)
        script_path = generator.generate_batch_slurm_script()

        assert script_path.name.startswith("BATCH-model1-model2")
        # Verify individual launch scripts were generated for each model
        assert len(generator.script_paths) == 2
        # Check that write_text was called multiple times
        assert mock_write_text.call_count >= 3  # 2 model scripts + 1 batch script
