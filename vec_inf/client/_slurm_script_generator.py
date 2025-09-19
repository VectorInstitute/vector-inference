"""Class for generating SLURM scripts to run vLLM servers.

This module provides functionality to generate SLURM scripts for running vLLM servers
in both single-node and multi-node configurations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from vec_inf.client._client_vars import (
    SLURM_JOB_CONFIG_ARGS,
    SLURM_SCRIPT_TEMPLATE,
)


class SlurmScriptGenerator:
    """A class to generate SLURM scripts for running vLLM servers.

    This class handles the generation of SLURM scripts for both single-node and
    multi-node configurations, supporting different virtualization environments
    (venv or singularity).

    Parameters
    ----------
    params : dict[str, Any]
        Configuration parameters for the SLURM script. Contains settings for job
        configuration, model parameters, and virtualization environment.
    """

    def __init__(self, params: dict[str, Any]):
        """Initialize the SlurmScriptGenerator with configuration parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Configuration parameters for the SLURM script.
        """
        self.params = params
        self.is_multinode = int(self.params["num_nodes"]) > 1
        self.use_singularity = self.params["venv"] == "singularity"
        self.additional_binds = self.params.get("bind", "")
        if self.additional_binds:
            self.additional_binds = f" --bind {self.additional_binds}"
        self.model_weights_path = str(
            Path(params["model_weights_parent_dir"], params["model_name"])
        )

    def _generate_script_content(self) -> str:
        """Generate the complete SLURM script content.

        Returns
        -------
        str
            The complete SLURM script as a string.
        """
        script_content = []
        script_content.append(self._generate_shebang())
        script_content.append(self._generate_server_setup())
        script_content.append(self._generate_launch_cmd())
        return "\n".join(script_content)

    def _generate_shebang(self) -> str:
        """Generate the SLURM script shebang with job specifications.

        Returns
        -------
        str
            SLURM shebang containing job specifications.
        """
        shebang = [SLURM_SCRIPT_TEMPLATE["shebang"]["base"]]
        for arg, value in SLURM_JOB_CONFIG_ARGS.items():
            if self.params.get(value):
                shebang.append(f"#SBATCH --{arg}={self.params[value]}")
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
        if self.use_singularity:
            server_script.append("\n".join(SLURM_SCRIPT_TEMPLATE["singularity_setup"]))
        server_script.append("\n".join(SLURM_SCRIPT_TEMPLATE["env_vars"]))
        server_script.append(
            SLURM_SCRIPT_TEMPLATE["imports"].format(src_dir=self.params["src_dir"])
        )
        if self.is_multinode:
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"]["multinode"]
            )
            if self.use_singularity:
                server_setup_str = server_setup_str.replace(
                    "SINGULARITY_PLACEHOLDER",
                    SLURM_SCRIPT_TEMPLATE["singularity_command"].format(
                        model_weights_path=self.model_weights_path,
                        additional_binds=self.additional_binds,
                    ),
                )
        else:
            server_setup_str = "\n".join(
                SLURM_SCRIPT_TEMPLATE["server_setup"]["single_node"]
            )
        server_script.append(server_setup_str)
        server_script.append("\n".join(SLURM_SCRIPT_TEMPLATE["find_vllm_port"]))
        server_script.append(
            "\n".join(SLURM_SCRIPT_TEMPLATE["write_to_json"]).format(
                log_dir=self.params["log_dir"], model_name=self.params["model_name"]
            )
        )
        return "\n".join(server_script)

    def _generate_launch_cmd(self) -> str:
        """Generate the vLLM server launch command.

        Creates the command to launch the vLLM server, handling different virtualization
        environments (venv or singularity).

        Returns
        -------
        str
            Server launch command.
        """
        launcher_script = ["\n"]
        if self.use_singularity:
            launcher_script.append(
                SLURM_SCRIPT_TEMPLATE["singularity_command"].format(
                    model_weights_path=self.model_weights_path,
                    additional_binds=self.additional_binds,
                )
                + " \\"
            )
        else:
            launcher_script.append(
                SLURM_SCRIPT_TEMPLATE["activate_venv"].format(venv=self.params["venv"])
            )
        launcher_script.append(
            "\n".join(SLURM_SCRIPT_TEMPLATE["launch_cmd"]).format(
                model_weights_path=self.model_weights_path,
                model_name=self.params["model_name"],
            )
        )

        for arg, value in self.params["vllm_args"].items():
            if isinstance(value, bool):
                launcher_script.append(f"    {arg} \\")
            else:
                launcher_script.append(f"    {arg} {value} \\")
        return "\n".join(launcher_script)

    def write_to_log_dir(self) -> Path:
        """Write the generated SLURM script to the log directory.

        Creates a timestamped script file in the configured log directory.

        Returns
        -------
        Path
            Path to the generated SLURM script file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_suffix = uuid4().hex[:8]
        script_path: Path = (
            Path(self.params["log_dir"])
            / f"launch_{self.params['model_name']}_{timestamp}_{unique_suffix}.slurm"
        )

        content = self._generate_script_content()
        script_path.write_text(content)
        return script_path
