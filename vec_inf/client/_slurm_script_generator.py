"""Class for generating Slurm scripts to run inference servers.

This module provides functionality to generate Slurm scripts for running inference
servers in both single-node and multi-node configurations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from vec_inf.client._client_vars import SLURM_JOB_CONFIG_ARGS
from vec_inf.client._slurm_templates import (
    BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE,
    BATCH_SLURM_SCRIPT_TEMPLATE,
    SLURM_SCRIPT_TEMPLATE,
)
from vec_inf.client._slurm_vars import CONTAINER_MODULE_NAME, IMAGE_PATH


class SlurmScriptGenerator:
    """A class to generate Slurm scripts for running inference servers.

    This class handles the generation of Slurm scripts for both single-node and
    multi-node configurations, supporting different virtualization environments
    (venv or singularity/apptainer).

    Parameters
    ----------
        params : dict[str, Any]
            Configuration parameters for the Slurm script.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.engine = params.get("engine", "vllm")
        self.is_multinode = int(self.params["num_nodes"]) > 1
        self.use_container = self.params["venv"] == CONTAINER_MODULE_NAME
        self.additional_binds = (
            f",{self.params['bind']}" if self.params.get("bind") else ""
        )
        self.model_weights_path = str(
            Path(self.params["model_weights_parent_dir"], self.params["model_name"])
        )
        self.env_str = self._generate_env_str()

    def _generate_env_str(self) -> str:
        """Generate the environment variables string for the Slurm script.

        Returns
        -------
        str
            Formatted env vars string for container or shell export commands.
        """
        env_dict: dict[str, str] = self.params.get("env", {})

        if not env_dict:
            return ""

        if self.use_container:
            # Format for container: --env KEY1=VAL1,KEY2=VAL2
            env_pairs = [f"{key}={val}" for key, val in env_dict.items()]
            return f"--env {','.join(env_pairs)}"
        # Format for shell: export KEY1=VAL1\nexport KEY2=VAL2
        export_lines = [f"export {key}={val}" for key, val in env_dict.items()]
        return "\n".join(export_lines)

    def _generate_script_content(self) -> str:
        """Generate the complete Slurm script content.

        Returns
        -------
        str
            The complete Slurm script as a string.
        """
        script_content = []
        script_content.append(self._generate_shebang())
        script_content.append(self._generate_server_setup())
        script_content.append(self._generate_launch_cmd())
        return "\n".join(script_content)

    def _generate_shebang(self) -> str:
        """Generate the Slurm script shebang with job specifications.

        Returns
        -------
        str
            Slurm shebang containing job specifications.
        """
        shebang = [SLURM_SCRIPT_TEMPLATE["shebang"]["base"]]
        for arg, value in SLURM_JOB_CONFIG_ARGS.items():
            if self.params.get(value):
                shebang.append(f"#SBATCH --{arg}={self.params[value]}")
            if value == "model_name":
                shebang[-1] += "-vec-inf"
        if self.is_multinode:
            shebang += SLURM_SCRIPT_TEMPLATE["shebang"]["multinode"]
        return "\n".join(shebang)

    def _generate_server_setup(self) -> str:
        """Generate the server initialization script.

        Creates the script section that handles server setup, including Ray
        initialization for multi-node setups and port configuration.

        Returns
        -------
        str
            Server initialization script content.
        """
        server_script = ["\n"]
        if self.use_container:
            server_script.append("\n".join(SLURM_SCRIPT_TEMPLATE["container_setup"]))
            server_script.append(
                SLURM_SCRIPT_TEMPLATE["bind_path"].format(
                    model_weights_path=self.model_weights_path,
                    additional_binds=self.additional_binds,
                )
            )
        else:
            server_script.append(
                SLURM_SCRIPT_TEMPLATE["activate_venv"].format(venv=self.params["venv"])
            )
            server_script.append(self.env_str)
        server_script.append(
            SLURM_SCRIPT_TEMPLATE["imports"].format(src_dir=self.params["src_dir"])
        )

        if self.is_multinode and self.engine == "vllm":
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"]["multinode_vllm"]
            ).format(gpus_per_node=self.params["gpus_per_node"])
            if self.use_container:
                server_setup_str = server_setup_str.replace(
                    "CONTAINER_PLACEHOLDER",
                    SLURM_SCRIPT_TEMPLATE["container_command"].format(
                        env_str=self.env_str,
                        image_path=IMAGE_PATH[self.engine],
                    ),
                )
            else:
                server_setup_str = server_setup_str.replace(
                    "CONTAINER_PLACEHOLDER",
                    "\\",
                )
        elif self.is_multinode and self.engine == "sglang":
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"]["multinode_sglang"]
            )
        else:
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"]["single_node"]
            )
        server_script.append(server_setup_str)
        server_script.append("\n".join(SLURM_SCRIPT_TEMPLATE["find_server_port"]))
        server_script.append(
            "\n".join(SLURM_SCRIPT_TEMPLATE["write_to_json"]).format(
                log_dir=self.params["log_dir"], model_name=self.params["model_name"]
            )
        )
        return "\n".join(server_script)

    def _generate_launch_cmd(self) -> str:
        """Generate the inference server launch command.

        Creates the command to launch the inference server, handling different
        virtualization environments (venv or singularity/apptainer).

        Returns
        -------
        str
            Server launch command.
        """
        if self.is_multinode and self.engine == "sglang":
            return self._generate_multinode_sglang_launch_cmd()

        launch_cmd = ["\n"]
        if self.use_container:
            launch_cmd.append(
                SLURM_SCRIPT_TEMPLATE["container_command"].format(
                    env_str=self.env_str,
                    image_path=IMAGE_PATH[self.engine],
                )
            )

        launch_cmd.append(
            "\n".join(SLURM_SCRIPT_TEMPLATE["launch_cmd"][self.engine]).format(  # type: ignore[literal-required]
                model_weights_path=self.model_weights_path,
                model_name=self.params["model_name"],
            )
        )

        for arg, value in self.params["engine_args"].items():
            if isinstance(value, bool):
                launch_cmd.append(f"    {arg} \\")
            else:
                launch_cmd.append(f"    {arg} {value} \\")

        # A known bug in vLLM requires setting backend to ray for multi-node
        # Remove this when the bug is fixed
        if self.is_multinode:
            launch_cmd.append("    --distributed-executor-backend ray \\")

        return "\n".join(launch_cmd).rstrip(" \\")

    def _generate_multinode_sglang_launch_cmd(self) -> str:
        """Generate the launch command for multi-node sglang setup.

        Returns
        -------
        str
            Multi-node sglang launch command.
        """
        launch_cmd = "\n" + "\n".join(
            SLURM_SCRIPT_TEMPLATE["launch_cmd"]["sglang_multinode"]
        ).format(
            num_nodes=self.params["num_nodes"],
            model_weights_path=self.model_weights_path,
            model_name=self.params["model_name"],
        )

        container_placeholder = "\\"
        if self.use_container:
            container_placeholder = SLURM_SCRIPT_TEMPLATE["container_command"].format(
                env_str=self.env_str,
                image_path=IMAGE_PATH[self.engine],
            )
        launch_cmd = launch_cmd.replace(
            "CONTAINER_PLACEHOLDER",
            container_placeholder,
        )

        engine_arg_str = ""
        for arg, value in self.params["engine_args"].items():
            if isinstance(value, bool):
                engine_arg_str += f"            {arg} \\\n"
            else:
                engine_arg_str += f"            {arg} {value} \\\n"

        return launch_cmd.replace(
            "SGLANG_ARGS_PLACEHOLDER", engine_arg_str.rstrip("\\\n")
        )

    def write_to_log_dir(self) -> Path:
        """Write the generated Slurm script to the log directory.

        Creates a timestamped script file in the configured log directory.

        Returns
        -------
        Path
            Path to the generated Slurm script file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path: Path = (
            Path(self.params["log_dir"])
            / f"launch_{self.params['model_name']}_{timestamp}.sbatch"
        )

        content = self._generate_script_content()
        script_path.write_text(content)
        return script_path


class BatchSlurmScriptGenerator:
    """A class to generate Slurm scripts for batch mode.

    This class handles the generation of Slurm scripts for batch mode, which
    launches multiple inference servers with different configurations in parallel.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.script_paths: list[Path] = []
        self.use_container = self.params["venv"] == CONTAINER_MODULE_NAME
        for model_name in self.params["models"]:
            self.params["models"][model_name]["additional_binds"] = (
                f",{self.params['models'][model_name]['bind']}"
                if self.params["models"][model_name].get("bind")
                else ""
            )
            self.params["models"][model_name]["model_weights_path"] = str(
                Path(
                    self.params["models"][model_name]["model_weights_parent_dir"],
                    model_name,
                )
            )

    def _write_to_log_dir(self, script_content: list[str], script_name: str) -> Path:
        """Write the generated Slurm script to the log directory.

        Returns
        -------
        Path
            The Path object to the generated Slurm script file.
        """
        script_path = Path(self.params["log_dir"]) / script_name
        script_path.touch(exist_ok=True)
        script_path.write_text("\n".join(script_content))
        return script_path

    def _generate_model_launch_script(self, model_name: str) -> Path:
        """Generate the bash script for launching individual inference servers.

        Parameters
        ----------
        model_name : str
            The name of the model to launch.

        Returns
        -------
        Path
            The bash script path for launching the inference server.
        """
        # Generate the bash script content
        script_content = []
        model_params = self.params["models"][model_name]
        script_content.append(BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["shebang"])
        if self.use_container:
            script_content.append(BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["container_setup"])
        script_content.append(
            BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["bind_path"].format(
                model_weights_path=model_params["model_weights_path"],
                additional_binds=model_params["additional_binds"],
            )
        )
        script_content.append(
            "\n".join(
                BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["server_address_setup"]
            ).format(src_dir=self.params["src_dir"])
        )
        script_content.append(
            "\n".join(BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["write_to_json"]).format(
                het_group_id=model_params["het_group_id"],
                log_dir=self.params["log_dir"],
                slurm_job_name=self.params["slurm_job_name"],
                model_name=model_name,
            )
        )
        if self.use_container:
            script_content.append(
                BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["container_command"].format(
                    image_path=IMAGE_PATH[model_params["engine"]],
                )
            )
        script_content.append(
            "\n".join(
                BATCH_MODEL_LAUNCH_SCRIPT_TEMPLATE["launch_cmd"][model_params["engine"]]
            ).format(
                model_weights_path=model_params["model_weights_path"],
                model_name=model_name,
            )
        )
        for arg, value in model_params["engine_args"].items():
            if isinstance(value, bool):
                script_content.append(f"    {arg} \\")
            else:
                script_content.append(f"    {arg} {value} \\")
        script_content[-1] = script_content[-1].rstrip(" \\")
        # Write the bash script to the log directory
        launch_script_path = self._write_to_log_dir(
            script_content, f"launch_{model_name}.sh"
        )
        self.script_paths.append(launch_script_path)
        return launch_script_path

    def _generate_batch_slurm_script_shebang(self) -> str:
        """Generate the shebang for batch mode Slurm script.

        Returns
        -------
        str
            The shebang for batch mode Slurm script.
        """
        shebang = [BATCH_SLURM_SCRIPT_TEMPLATE["shebang"]]

        for arg, value in SLURM_JOB_CONFIG_ARGS.items():
            if self.params.get(value):
                shebang.append(f"#SBATCH --{arg}={self.params[value]}")
        shebang.append("#SBATCH --ntasks=1")
        shebang.append("\n")

        for model_name in self.params["models"]:
            shebang.append(f"# ===== Resource group for {model_name} =====")
            for arg, value in SLURM_JOB_CONFIG_ARGS.items():
                model_params = self.params["models"][model_name]
                if model_params.get(value) and value not in ["out_file", "err_file"]:
                    shebang.append(f"#SBATCH --{arg}={model_params[value]}")
                if value == "model_name":
                    shebang[-1] += "-vec-inf"
            shebang[-1] += "\n"
            shebang.append(BATCH_SLURM_SCRIPT_TEMPLATE["hetjob"])
        # Remove the last hetjob line
        shebang.pop()
        return "\n".join(shebang)

    def generate_batch_slurm_script(self) -> Path:
        """Generate the Slurm script for launching multiple inference servers in batch.

        Returns
        -------
        Path
            The Slurm script for launching multiple inference servers in batch.
        """
        script_content = []

        script_content.append(self._generate_batch_slurm_script_shebang())

        for model_name in self.params["models"]:
            model_params = self.params["models"][model_name]
            script_content.append(f"# ===== Launching {model_name} =====")
            launch_script_path = str(self._generate_model_launch_script(model_name))
            script_content.append(
                BATCH_SLURM_SCRIPT_TEMPLATE["permission_update"].format(
                    script_name=launch_script_path
                )
            )
            script_content.append(
                "\n".join(BATCH_SLURM_SCRIPT_TEMPLATE["launch_model_scripts"]).format(
                    het_group_id=model_params["het_group_id"],
                    out_file=model_params["out_file"],
                    err_file=model_params["err_file"],
                    script_name=launch_script_path,
                )
            )
        script_content.append("wait")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = f"{self.params['slurm_job_name']}_{timestamp}.sbatch"
        return self._write_to_log_dir(script_content, script_name)
