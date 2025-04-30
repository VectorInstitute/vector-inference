"""Command line interface for Vector Inference."""

import time
from typing import Optional, Union

import click
from rich.console import Console
from rich.live import Live

from vec_inf.cli._helper import (
    LaunchResponseFormatter,
    ListCmdDisplay,
    MetricsResponseFormatter,
    StatusResponseFormatter,
)
from vec_inf.client import LaunchOptions, LaunchOptionsDict, VecInfClient


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
    "--partition",
    type=str,
    help="Type of compute partition",
)
@click.option(
    "--num-nodes",
    type=int,
    help="Number of nodes to use, default to suggested resource allocation for model",
)
@click.option(
    "--gpus-per-node",
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
    "--venv",
    type=str,
    help="Path to virtual environment",
)
@click.option(
    "--log-dir",
    type=str,
    help="Path to slurm log directory",
)
@click.option(
    "--model-weights-parent-dir",
    type=str,
    help="Path to parent directory containing model weights",
)
@click.option(
    "--vllm-args",
    type=str,
    help="vLLM engine arguments to be set, use the format as specified in vLLM documentation and separate arguments with commas, e.g. --vllm-args '--max-model-len=8192,--max-num-seqs=256,--enable-prefix-caching'",
)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
def launch(
    model_name: str,
    **cli_kwargs: Optional[Union[str, int, float, bool]],
) -> None:
    """Launch a model on the cluster."""
    try:
        # Convert cli_kwargs to LaunchOptions
        kwargs = {k: v for k, v in cli_kwargs.items() if k != "json_mode"}
        # Cast the dictionary to LaunchOptionsDict
        options_dict: LaunchOptionsDict = kwargs  # type: ignore
        launch_options = LaunchOptions(**options_dict)

        # Start the client and launch model inference server
        client = VecInfClient()
        launch_response = client.launch_model(model_name, launch_options)

        # Display launch information
        launch_formatter = LaunchResponseFormatter(model_name, launch_response.config)
        if cli_kwargs.get("json_mode"):
            click.echo(launch_response.config)
        else:
            launch_info_table = launch_formatter.format_table_output()
            CONSOLE.print(launch_info_table)

    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Launch failed: {str(e)}") from e


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
    try:
        # Start the client and get model inference server status
        client = VecInfClient()
        status_response = client.get_status(slurm_job_id, log_dir)
        # Display status information
        status_formatter = StatusResponseFormatter(status_response)
        if json_mode:
            status_formatter.output_json()
        else:
            status_info_table = status_formatter.output_table()
            CONSOLE.print(status_info_table)

    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Status check failed: {str(e)}") from e


@cli.command("shutdown")
@click.argument("slurm_job_id", type=int, nargs=1)
def shutdown(slurm_job_id: int) -> None:
    """Shutdown a running model on the cluster."""
    try:
        client = VecInfClient()
        client.shutdown_model(slurm_job_id)
        click.echo(f"Shutting down model with Slurm Job ID: {slurm_job_id}")
    except Exception as e:
        raise click.ClickException(f"Shutdown failed: {str(e)}") from e


@cli.command("list")
@click.argument("model-name", required=False)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
def list_models(model_name: Optional[str] = None, json_mode: bool = False) -> None:
    """List all available models, or get default setup of a specific model."""
    try:
        # Start the client
        client = VecInfClient()
        list_display = ListCmdDisplay(CONSOLE, json_mode)
        if model_name:
            model_config = client.get_model_config(model_name)
            list_display.display_single_model_output(model_config)
        else:
            model_infos = client.list_models()
            list_display.display_all_models_output(model_infos)
    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"List models failed: {str(e)}") from e


@cli.command("metrics")
@click.argument("slurm_job_id", type=int, nargs=1)
@click.option(
    "--log-dir", type=str, help="Path to slurm log directory (if used during launch)"
)
def metrics(slurm_job_id: int, log_dir: Optional[str] = None) -> None:
    """Stream real-time performance metrics from the model endpoint."""
    try:
        # Start the client and get inference server metrics
        client = VecInfClient()
        metrics_response = client.get_metrics(slurm_job_id, log_dir)
        metrics_formatter = MetricsResponseFormatter(metrics_response.metrics)

        # Check if metrics response is ready
        if isinstance(metrics_response.metrics, str):
            metrics_formatter.format_failed_metrics(metrics_response.metrics)
            CONSOLE.print(metrics_formatter.table)
            return

        with Live(refresh_per_second=1, console=CONSOLE) as live:
            while True:
                metrics_response = client.get_metrics(slurm_job_id, log_dir)
                metrics_formatter = MetricsResponseFormatter(metrics_response.metrics)

                if isinstance(metrics_response.metrics, str):
                    # Show status information if metrics aren't available
                    metrics_formatter.format_failed_metrics(metrics_response.metrics)
                else:
                    metrics_formatter.format_metrics()

                live.update(metrics_formatter.table)
                time.sleep(2)
    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Metrics check failed: {str(e)}") from e


if __name__ == "__main__":
    cli()
