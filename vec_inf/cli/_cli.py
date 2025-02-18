"""Command line interface for Vector Inference."""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import click
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

import vec_inf.cli._utils as utils
from vec_inf.cli._config import ModelConfig


CONSOLE = Console()


@click.group()
def cli() -> None:
    """Vector Inference CLI."""
    pass


@cli.command("launch")
@click.argument("model-name", type=str, nargs=1)
@click.option("--model-family", type=str, help="The model family")
@click.option("--model-variant", type=str, help="The model variant")
@click.option(
    "--max-model-len",
    type=int,
    help="Model context length. Default value set based on suggested resource allocation.",
)
@click.option(
    "--max-num-seqs",
    type=int,
    help="Maximum number of sequences to process in a single request",
)
@click.option(
    "--partition",
    type=str,
    default="a40",
    help="Type of compute partition, default to a40",
)
@click.option(
    "--num-nodes",
    type=int,
    help="Number of nodes to use, default to suggested resource allocation for model",
)
@click.option(
    "--num-gpus",
    type=int,
    help="Number of GPUs/node to use, default to suggested resource allocation for model",
)
@click.option(
    "--qos",
    type=str,
    help="Quality of service",
)
@click.option(
    "--time",
    type=str,
    help="Time limit for job, this should comply with QoS limits",
)
@click.option(
    "--vocab-size",
    type=int,
    help="Vocabulary size, this option is intended for custom models",
)
@click.option(
    "--data-type", type=str, default="auto", help="Model data type, default to auto"
)
@click.option(
    "--venv",
    type=str,
    default="singularity",
    help="Path to virtual environment, default to preconfigured singularity container",
)
@click.option(
    "--log-dir",
    type=str,
    default="default",
    help="Path to slurm log directory, default to .vec-inf-logs in user home directory",
)
@click.option(
    "--model-weights-parent-dir",
    type=str,
    default="/model-weights",
    help="Path to parent directory containing model weights, default to '/model-weights' for supported models",
)
@click.option(
    "--pipeline-parallelism",
    type=str,
    help="Enable pipeline parallelism, accepts 'True' or 'False', default to 'True' for supported models",
)
@click.option(
    "--enforce-eager",
    type=str,
    help="Always use eager-mode PyTorch, accepts 'True' or 'False', default to 'False' for custom models if not set",
)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
def launch(
    model_name: str,
    **cli_kwargs: Optional[Union[str, int, bool]],
) -> None:
    """Launch a model on the cluster."""
    try:
        model_config = _get_model_configuration(model_name)
        base_command = _get_base_launch_command()
        params = _process_configuration(model_config, cli_kwargs)
        launch_command = _build_launch_command(base_command, params)
        command_output = utils.run_bash_command(launch_command)
        _handle_launch_output(command_output, bool(cli_kwargs.get("json_mode", False)))
    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Launch failed: {str(e)}") from e


def _get_model_configuration(model_name: str) -> ModelConfig:
    """Load and validate model configuration."""
    model_configs = utils.load_config()
    if config := next((m for m in model_configs if m.model_name == model_name), None):
        return config
    raise click.ClickException(f"Model '{model_name}' not found in configuration")


def _get_base_launch_command() -> str:
    """Construct base launch command."""
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "launch_server.sh"
    )
    return f"bash {script_path}"


def _process_configuration(
    model_config: ModelConfig, cli_overrides: Dict[str, Optional[Union[str, int, bool]]]
) -> Dict[str, Any]:
    """Merge config defaults with CLI overrides."""
    params = model_config.model_dump(exclude={"model_name"})

    # Process boolean fields
    for bool_field in ["pipeline_parallelism", "enforce_eager"]:
        if (value := cli_overrides.get(bool_field)) is not None:
            params[bool_field] = _convert_boolean_value(value)

    # Merge other overrides
    for key, value in cli_overrides.items():
        if value is not None and key not in [
            "json_mode",
            "pipeline_parallelism",
            "enforce_eager",
        ]:
            params[key] = value

    return params


def _convert_boolean_value(value: Union[str, int, bool]) -> bool:
    """Convert various input types to boolean."""
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _build_launch_command(base_command: str, params: Dict[str, Any]) -> str:
    """Construct the full launch command with parameters."""
    command = base_command
    for param_name, param_value in params.items():
        if param_value is None:
            continue

        arg_value = (
            "True"
            if param_value
            else "False"
            if isinstance(param_value, bool)
            else str(param_value)
        )

        arg_name = param_name.replace("_", "-")
        command += f" --{arg_name} {arg_value}"

    return command


def _handle_launch_output(output: str, json_mode: bool = False) -> None:
    """Process and display launch output."""
    slurm_job_id, output_lines = _parse_launch_output(output)

    if json_mode:
        output_data = _format_json_output(slurm_job_id, output_lines)
        click.echo(output_data)
    else:
        table = _format_table_output(slurm_job_id, output_lines)
        CONSOLE.print(table)


def _parse_launch_output(output: str) -> Tuple[str, List[str]]:
    """Extract job ID and output lines from command output."""
    slurm_job_id = output.split(" ")[-1].strip().strip("\n")
    output_lines = output.split("\n")[:-2]
    return slurm_job_id, output_lines


def _format_json_output(job_id: str, lines: List[str]) -> str:
    """Format output as JSON string with proper double quotes."""
    output_data = {"slurm_job_id": job_id}
    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            output_data[key.lower().replace(" ", "_")] = value
    return json.dumps(output_data)


def _format_table_output(job_id: str, lines: List[str]) -> Table:
    """Format output as rich Table."""
    table = utils.create_table(key_title="Job Config", value_title="Value")
    table.add_row("Slurm Job ID", job_id, style="blue")
    for line in lines:
        key, value = line.split(": ")
        table.add_row(key, value)
    return table


@cli.command("status")
@click.argument("slurm_job_id", type=int, nargs=1)
@click.option(
    "--log-dir",
    type=str,
    help="Path to slurm log directory. This is required if --log-dir was set in model launch",
)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
def status(
    slurm_job_id: int, log_dir: Optional[str] = None, json_mode: bool = False
) -> None:
    """Get the status of a running model on the cluster."""
    status_cmd = f"scontrol show job {slurm_job_id} --oneliner"
    output = utils.run_bash_command(status_cmd)

    base_data = _get_base_status_data(output)
    status_info = _process_job_state(output, base_data, slurm_job_id, log_dir)
    _display_status(status_info, json_mode)


def _get_base_status_data(output: str) -> Dict[str, Any]:
    """Extract basic job status information from scontrol output."""
    try:
        job_name = output.split(" ")[1].split("=")[1]
        job_state = output.split(" ")[9].split("=")[1]
    except IndexError:
        job_name = "UNAVAILABLE"
        job_state = "UNAVAILABLE"

    return {
        "model_name": job_name,
        "status": "SHUTDOWN",
        "base_url": "UNAVAILABLE",
        "state": job_state,
        "pending_reason": None,
        "failed_reason": None,
    }


def _process_job_state(
    output: str, status_info: Dict[str, Any], slurm_job_id: int, log_dir: Optional[str]
) -> Dict[str, Any]:
    """Process different job states and update status information."""
    if status_info["state"] == "PENDING":
        _process_pending_state(output, status_info)
    elif status_info["state"] == "RUNNING":
        _handle_running_state(status_info, slurm_job_id, log_dir)
    return status_info


def _process_pending_state(output: str, status_info: Dict[str, Any]) -> None:
    """Handle PENDING job state."""
    try:
        status_info["pending_reason"] = output.split(" ")[10].split("=")[1]
        status_info["status"] = "PENDING"
    except IndexError:
        status_info["pending_reason"] = "Unknown pending reason"


def _handle_running_state(
    status_info: Dict[str, Any], slurm_job_id: int, log_dir: Optional[str]
) -> None:
    """Handle RUNNING job state and check server status."""
    server_status = utils.is_server_running(
        status_info["model_name"], slurm_job_id, log_dir
    )

    if isinstance(server_status, tuple):
        status_info["status"], status_info["failed_reason"] = server_status
        return

    if server_status == "RUNNING":
        _check_model_health(status_info, slurm_job_id, log_dir)
    else:
        status_info["status"] = server_status


def _check_model_health(
    status_info: Dict[str, Any], slurm_job_id: int, log_dir: Optional[str]
) -> None:
    """Check model health and update status accordingly."""
    model_status = utils.model_health_check(
        status_info["model_name"], slurm_job_id, log_dir
    )
    status, failed_reason = model_status
    if status == "READY":
        status_info["base_url"] = utils.get_base_url(
            status_info["model_name"], slurm_job_id, log_dir
        )
        status_info["status"] = status
    else:
        status_info["status"], status_info["failed_reason"] = status, failed_reason


def _display_status(status_info: Dict[str, Any], json_mode: bool) -> None:
    """Display the status information in appropriate format."""
    if json_mode:
        _output_json(status_info)
    else:
        _output_table(status_info)


def _output_json(status_info: Dict[str, Any]) -> None:
    """Format and output JSON data."""
    json_data = {
        "model_name": status_info["model_name"],
        "model_status": status_info["status"],
        "base_url": status_info["base_url"],
    }
    if status_info["pending_reason"]:
        json_data["pending_reason"] = status_info["pending_reason"]
    if status_info["failed_reason"]:
        json_data["failed_reason"] = status_info["failed_reason"]
    click.echo(json_data)


def _output_table(status_info: Dict[str, Any]) -> None:
    """Create and display rich table."""
    table = utils.create_table(key_title="Job Status", value_title="Value")
    table.add_row("Model Name", status_info["model_name"])
    table.add_row("Model Status", status_info["status"], style="blue")

    if status_info["pending_reason"]:
        table.add_row("Pending Reason", status_info["pending_reason"])
    if status_info["failed_reason"]:
        table.add_row("Failed Reason", status_info["failed_reason"])

    table.add_row("Base URL", status_info["base_url"])
    CONSOLE.print(table)


@cli.command("shutdown")
@click.argument("slurm_job_id", type=int, nargs=1)
def shutdown(slurm_job_id: int) -> None:
    """Shutdown a running model on the cluster."""
    shutdown_cmd = f"scancel {slurm_job_id}"
    utils.run_bash_command(shutdown_cmd)
    click.echo(f"Shutting down model with Slurm Job ID: {slurm_job_id}")


@cli.command("list")
@click.argument("model-name", required=False)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
def list_models(model_name: Optional[str] = None, json_mode: bool = False) -> None:
    """List all available models, or get default setup of a specific model."""
    model_configs = utils.load_config()

    def list_single_model(target_name: str) -> None:
        config = next((c for c in model_configs if c.model_name == target_name), None)
        if not config:
            raise click.ClickException(
                f"Model '{target_name}' not found in configuration"
            )

        if json_mode:
            # Exclude non-essential fields from JSON output
            excluded = {"venv", "log_dir"}
            output = config.model_dump(exclude=excluded)
            click.echo(output)
        else:
            table = utils.create_table(key_title="Model Config", value_title="Value")
            for field, value in config.model_dump().items():
                if field not in {"venv", "log_dir"}:
                    table.add_row(field, str(value))
            CONSOLE.print(table)

    def list_all_models() -> None:
        if json_mode:
            click.echo([config.model_name for config in model_configs])
            return

        # Sort by model type priority
        type_priority = {"LLM": 0, "VLM": 1, "Text_Embedding": 2, "Reward_Modeling": 3}

        sorted_configs = sorted(
            model_configs, key=lambda x: type_priority.get(x.model_type, 4)
        )

        # Create panels with color coding
        model_type_colors = {
            "LLM": "cyan",
            "VLM": "bright_blue",
            "Text_Embedding": "purple",
            "Reward_Modeling": "bright_magenta",
        }

        panels = []
        for config in sorted_configs:
            color = model_type_colors.get(config.model_type, "white")
            variant = config.model_variant or ""
            display_text = f"[magenta]{config.model_family}[/magenta]"
            if variant:
                display_text += f"-{variant}"

            panels.append(Panel(display_text, expand=True, border_style=color))

        CONSOLE.print(Columns(panels, equal=True))

    if model_name:
        list_single_model(model_name)
    else:
        list_all_models()


@cli.command("metrics")
@click.argument("slurm_job_id", type=int, nargs=1)
@click.option(
    "--log-dir",
    type=str,
    help="Path to slurm log directory. This is required if --log-dir was set in model launch",
)
def metrics(slurm_job_id: int, log_dir: Optional[str] = None) -> None:
    """Stream performance metrics to the console."""
    status_cmd = f"scontrol show job {slurm_job_id} --oneliner"
    output = utils.run_bash_command(status_cmd)
    slurm_job_name = output.split(" ")[1].split("=")[1]

    with Live(refresh_per_second=1, console=CONSOLE) as live:
        while True:
            out_logs = utils.read_slurm_log(
                slurm_job_name, slurm_job_id, "out", log_dir
            )
            # if out_logs is a string, then it is an error message
            if isinstance(out_logs, str):
                live.update(out_logs)
                break
            latest_metrics = utils.get_latest_metric(out_logs)
            # if latest_metrics is a string, then it is an error message
            if isinstance(latest_metrics, str):
                live.update(latest_metrics)
                break
            table = utils.create_table(key_title="Metric", value_title="Value")
            for key, value in latest_metrics.items():
                table.add_row(key, value)

            live.update(table)

            time.sleep(2)


if __name__ == "__main__":
    cli()
