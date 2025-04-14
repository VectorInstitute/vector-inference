"""Command line interface for Vector Inference."""

import time
from typing import Optional, Union, cast

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
    "--gpu-memory-utilization",
    type=float,
    help="GPU memory utilization, default to 0.9",
)
@click.option(
    "--enable-prefix-caching",
    is_flag=True,
    help="Enables automatic prefix caching",
)
@click.option(
    "--enable-chunked-prefill",
    is_flag=True,
    help="Enable chunked prefill, enabled by default if max number of sequences > 32k",
)
@click.option(
    "--max-num-batched-tokens",
    type=int,
    help="Maximum number of batched tokens per iteration, defaults to 2048 if --enable-chunked-prefill is set, else None",
)
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
    "--vocab-size",
    type=int,
    help="Vocabulary size, this option is intended for custom models",
)
@click.option("--data-type", type=str, help="Model data type")
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
    "--pipeline-parallelism",
    is_flag=True,
    help="Enable pipeline parallelism, enabled by default for supported models",
)
@click.option(
    "--compilation-config",
    type=click.Choice(["0", "3"]),
    help="torch.compile optimization level, accepts '0' or '3', default to '0', which means no optimization is applied",
)
@click.option(
    "--enforce-eager",
    is_flag=True,
    help="Always use eager-mode PyTorch",
)
@click.option(
    "--json-mode",
    is_flag=True,
    help="Output in JSON string",
)
@click.option(
    "--vllm-arg",
    multiple=True,
    help='Extra vLLM server args (use --vllm-arg "--foo=bar")',
)
def launch(
    model_name: str,
    vllm_arg: tuple[str],
    json_mode: bool,
    **cli_kwargs: Optional[Union[str, int, float, bool]],
) -> None:
    """Launch a model on the cluster."""
    try:
        # Parse extra vLLM args
        vllm_optional_args = _parse_vllm_optional_args(vllm_arg)

        # Prepare LaunchOptions
        kwargs: dict[
            str, Union[str, int, float, bool, dict[str, Union[str, int, float, bool]]]
        ] = {k: v for k, v in cli_kwargs.items() if v is not None}
        kwargs["vllm_optional_args"] = vllm_optional_args

        options_dict: LaunchOptionsDict = cast(LaunchOptionsDict, kwargs)
        launch_options = LaunchOptions(**options_dict)

        # Launch
        client = VecInfClient()
        launch_response = client.launch_model(model_name, launch_options)

        formatter = LaunchResponseFormatter(model_name, launch_response.config)
        if json_mode:
            click.echo(launch_response.config)
        else:
            CONSOLE.print(formatter.format_table_output())

    except click.ClickException as e:
        raise e
    except Exception as e:
        raise click.ClickException(f"Launch failed: {str(e)}") from e


def _parse_vllm_optional_args(
    vllm_arg: tuple[str],
) -> dict[str, Union[str, int, float, bool]]:
    parsed: dict[str, Union[str, int, float, bool]] = {}
    for raw_arg in vllm_arg:
        arg = raw_arg.removeprefix("--")
        if "=" in arg:
            key, val = arg.split("=", maxsplit=1)
            if val.lower() == "true":
                parsed[key.replace("-", "_")] = True
            elif val.lower() == "false":
                parsed[key.replace("-", "_")] = False
            elif val.isdigit():
                parsed[key.replace("-", "_")] = int(val)
            else:
                try:
                    parsed[key.replace("-", "_")] = float(val)
                except ValueError:
                    parsed[key.replace("-", "_")] = val
        else:
            parsed[arg.replace("-", "_")] = True
    return parsed


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
