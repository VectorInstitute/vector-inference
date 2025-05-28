"""Unit tests for slurm script generator component in the vec_inf.client module."""

import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from vec_inf.client._slurm_script_generator import (
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
            "vllm_args": {
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
                "venv": "singularity",
                "bind": "/scratch:/scratch,/data:/data",
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
        assert not generator.use_singularity
        assert generator.additional_binds == ""
        assert generator.model_weights_path == "/path/to/model_weights/test-model"

    def test_init_multinode(self, multinode_params):
        """Test initialization with multi-node configuration."""
        generator = SlurmScriptGenerator(multinode_params)

        assert generator.params == multinode_params
        assert generator.is_multinode
        assert not generator.use_singularity
        assert generator.additional_binds == ""
        assert generator.model_weights_path == "/path/to/model_weights/test-model"

    def test_init_singularity(self, singularity_params):
        """Test initialization with Singularity configuration."""
        generator = SlurmScriptGenerator(singularity_params)

        assert generator.params == singularity_params
        assert generator.use_singularity
        assert not generator.is_multinode
        assert generator.additional_binds == " --bind /scratch:/scratch,/data:/data"
        assert generator.model_weights_path == "/path/to/model_weights/test-model"

    def test_init_singularity_no_bind(self, basic_params):
        """Test Singularity initialization without additional binds."""
        params = basic_params.copy()
        params["venv"] = "singularity"
        generator = SlurmScriptGenerator(params)

        assert generator.params == params
        assert generator.use_singularity
        assert not generator.is_multinode
        assert generator.additional_binds == ""
        assert generator.model_weights_path == "/path/to/model_weights/test-model"

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
        assert "#SBATCH --exclusive" in shebang
        assert "#SBATCH --tasks-per-node=1" in shebang

    def test_generate_server_setup_single_node(self, basic_params):
        """Test server setup generation for single-node."""
        generator = SlurmScriptGenerator(basic_params)
        setup = generator._generate_server_setup()

        assert "head_node_ip=${SLURMD_NODENAME}" in setup
        assert "source /path/to/src/find_port.sh" in setup
        assert "export LD_LIBRARY_PATH=" in setup
        assert "ray start --head" not in setup

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

        assert "singularity exec" in setup
        assert "ray stop" in setup
        assert "module load singularity" in setup

    def test_generate_launch_cmd_venv(self, basic_params):
        """Test launch command generation with virtual environment."""
        generator = SlurmScriptGenerator(basic_params)
        launch_cmd = generator._generate_launch_cmd()

        assert "source /path/to/venv/bin/activate" in launch_cmd
        assert "vllm serve /path/to/model_weights/test-model" in launch_cmd
        assert "--served-model-name test-model" in launch_cmd
        assert "--tensor-parallel-size 4" in launch_cmd
        assert "--max-model-len 8192" in launch_cmd
        assert "--enforce-eager" in launch_cmd

    def test_generate_launch_cmd_singularity(self, singularity_params):
        """Test launch command generation with Singularity."""
        generator = SlurmScriptGenerator(singularity_params)
        launch_cmd = generator._generate_launch_cmd()

        assert "singularity exec --nv" in launch_cmd
        assert "--bind /path/to/model_weights/test-model" in launch_cmd
        assert "--bind /scratch:/scratch,/data:/data" in launch_cmd
        assert "source" not in launch_cmd

    def test_generate_launch_cmd_boolean_args(self, basic_params):
        """Test launch command with boolean vLLM arguments."""
        params = basic_params.copy()
        params["vllm_args"] = {
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

            expected_path = temp_log_dir / "launch_test-model_20240101_120000.slurm"
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
