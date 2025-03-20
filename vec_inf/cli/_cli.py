"""Command line interface for Vector Inference."""

import time
from typing import Optional, Union

import click
from rich.console import Console
from rich.live import Live

import vec_inf.cli._utils as utils
from vec_inf.cli._helper import LaunchHelper, ListHelper, MetricsHelper, StatusHelper


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
def launch(
    model_name: str,
    **cli_kwargs: Optional[Union[str, int, bool]],
) -> None:
    """Launch a model on the cluster."""
    try:
        launch_helper = LaunchHelper(model_name, cli_kwargs)

        launch_helper.set_env_vars()
        launch_command = launch_helper.build_launch_command()
        command_output, stderr = utils.run_bash_command(launch_command)
        if stderr:
            raise click.ClickException(f"Error: {stderr}")
        launch_helper.post_launch_processing(command_output, CONSOLE)

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
    status_cmd = f"scontrol show job {slurm_job_id} --oneliner"
    output, stderr = utils.run_bash_command(status_cmd)
    if stderr:
        raise click.ClickException(f"Error: {stderr}")

    status_helper = StatusHelper(slurm_job_id, output, log_dir)

    status_helper.process_job_state()
    if json_mode:
        status_helper.output_json()
    else:
        status_helper.output_table(CONSOLE)


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
    list_helper = ListHelper(model_name, json_mode)
    list_helper.process_list_command(CONSOLE)


@cli.command("metrics")
@click.argument("slurm_job_id", type=int, nargs=1)
@click.option(
    "--log-dir", type=str, help="Path to slurm log directory (if used during launch)"
)
def metrics(slurm_job_id: int, log_dir: Optional[str] = None) -> None:
    """Stream real-time performance metrics from the model endpoint."""
    helper = MetricsHelper(slurm_job_id, log_dir)

    # Check if metrics URL is ready
    if not helper.metrics_url.startswith("http"):
        table = utils.create_table("Metric", "Value")
        helper.display_failed_metrics(
            table, f"Metrics endpoint unavailable - {helper.metrics_url}"
        )
        CONSOLE.print(table)
        return

    with Live(refresh_per_second=1, console=CONSOLE) as live:
        while True:
            metrics = helper.fetch_metrics()
            table = utils.create_table("Metric", "Value")

            if isinstance(metrics, str):
                # Show status information if metrics aren't available
                helper.display_failed_metrics(table, metrics)
            else:
                helper.display_metrics(table, metrics)

            live.update(table)
            time.sleep(2)


if __name__ == "__main__":
    cli()
