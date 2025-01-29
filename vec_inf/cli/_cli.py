"""Command line interface for Vector Inference."""

import os
import time
from typing import Any, Dict, Optional

import click
import polars as pl
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

import vec_inf.cli._utils as utils


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
    model_family: Optional[str] = None,
    model_variant: Optional[str] = None,
    max_model_len: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    partition: Optional[str] = None,
    num_nodes: Optional[int] = None,
    num_gpus: Optional[int] = None,
    qos: Optional[str] = None,
    time: Optional[str] = None,
    vocab_size: Optional[int] = None,
    data_type: Optional[str] = None,
    venv: Optional[str] = None,
    log_dir: Optional[str] = None,
    model_weights_parent_dir: Optional[str] = None,
    pipeline_parallelism: Optional[str] = None,
    enforce_eager: Optional[str] = None,
    json_mode: bool = False,
) -> None:
    """Launch a model on the cluster."""
    if isinstance(pipeline_parallelism, str):
        pipeline_parallelism = (
            "True" if pipeline_parallelism.lower() == "true" else "False"
        )

    launch_script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "launch_server.sh"
    )
    launch_cmd = f"bash {launch_script_path}"

    models_df = utils.load_models_df()

    if model_name in models_df["model_name"].to_list():
        default_args = utils.load_default_args(models_df, model_name)
        for arg in default_args:
            if arg in locals() and locals()[arg] is not None:
                default_args[arg] = locals()[arg]
            renamed_arg = arg.replace("_", "-")
            launch_cmd += f" --{renamed_arg} {default_args[arg]}"
    else:
        model_args = models_df.columns
        model_args.remove("model_name")
        model_args.remove("model_type")
        for arg in model_args:
            if locals()[arg] is not None:
                renamed_arg = arg.replace("_", "-")
                launch_cmd += f" --{renamed_arg} {locals()[arg]}"

    output = utils.run_bash_command(launch_cmd)

    slurm_job_id = output.split(" ")[-1].strip().strip("\n")
    output_lines = output.split("\n")[:-2]

    table = utils.create_table(key_title="Job Config", value_title="Value")
    table.add_row("Slurm Job ID", slurm_job_id, style="blue")
    output_dict = {"slurm_job_id": slurm_job_id}

    for line in output_lines:
        key, value = line.split(": ")
        table.add_row(key, value)
        output_dict[key.lower().replace(" ", "_")] = value

    if json_mode:
        click.echo(output_dict)
    else:
        CONSOLE.print(table)


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

    def list_model(model_name: str, models_df: pl.DataFrame, json_mode: bool) -> None:
        if model_name not in models_df["model_name"].to_list():
            raise ValueError(f"Model name {model_name} not found in available models")

        excluded_keys = {"venv", "log_dir"}
        model_row = models_df.filter(models_df["model_name"] == model_name)

        if json_mode:
            filtered_model_row = model_row.drop(excluded_keys, strict=False)
            click.echo(filtered_model_row.to_dicts()[0])
            return
        table = utils.create_table(key_title="Model Config", value_title="Value")
        for row in model_row.to_dicts():
            for key, value in row.items():
                if key not in excluded_keys:
                    table.add_row(key, str(value))
        CONSOLE.print(table)

    def list_all(models_df: pl.DataFrame, json_mode: bool) -> None:
        if json_mode:
            click.echo(models_df["model_name"].to_list())
            return
        panels = []
        model_type_colors = {
            "LLM": "cyan",
            "VLM": "bright_blue",
            "Text Embedding": "purple",
            "Reward Modeling": "bright_magenta",
        }

        models_df = models_df.with_columns(
            pl.when(pl.col("model_type") == "LLM")
            .then(0)
            .when(pl.col("model_type") == "VLM")
            .then(1)
            .when(pl.col("model_type") == "Text Embedding")
            .then(2)
            .when(pl.col("model_type") == "Reward Modeling")
            .then(3)
            .otherwise(-1)
            .alias("model_type_order")
        )

        models_df = models_df.sort("model_type_order")
        models_df = models_df.drop("model_type_order")

        for row in models_df.to_dicts():
            panel_color = model_type_colors.get(row["model_type"], "white")
            styled_text = (
                f"[magenta]{row['model_family']}[/magenta]-{row['model_variant']}"
            )
            panels.append(Panel(styled_text, expand=True, border_style=panel_color))
        CONSOLE.print(Columns(panels, equal=True))

    models_df = utils.load_models_df()

    if model_name:
        list_model(model_name, models_df, json_mode)
    else:
        list_all(models_df, json_mode)


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
