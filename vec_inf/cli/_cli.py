"""Command line interface for Vector Inference."""

import time
from typing import Optional, Union

import click
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

import vec_inf.cli._utils as utils
from vec_inf.cli._helper import LaunchHelper, MetricsHelper, StatusHelper


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
    help="Type of compute partition",
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
        launch_helper = LaunchHelper(model_name, cli_kwargs)

        base_command = launch_helper.get_base_launch_command()
        params = launch_helper.process_configuration()
        launch_command = launch_helper.build_launch_command(base_command, params)
        command_output = utils.run_bash_command(launch_command)
        launch_helper.handle_launch_output(command_output, CONSOLE)

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
    output = utils.run_bash_command(status_cmd)

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
    "--log-dir", type=str, help="Path to slurm log directory (if used during launch)"
)
def metrics(slurm_job_id: int, log_dir: Optional[str] = None) -> None:
    """Stream real-time performance metrics from the model endpoint."""
    with Live(refresh_per_second=1, console=CONSOLE) as live:
        while True:
            helper = MetricsHelper(slurm_job_id, log_dir)
            metrics = helper.fetch_metrics()

            table = utils.create_table("Metric", "Value")

            if isinstance(metrics, str):
                # Show status information if metrics aren't available
                table.add_row(
                    "System Status", helper.status_info["status"], style="yellow"
                )
                if helper.status_info["pending_reason"]:
                    table.add_row(
                        "Pending Reason", helper.status_info["pending_reason"]
                    )
                if helper.status_info["failed_reason"]:
                    table.add_row("Failure Reason", helper.status_info["failed_reason"])
                table.add_row("Message", metrics)
            else:
                # Display calculated rates
                if "prompt_tokens_per_sec" in metrics:
                    table.add_row(
                        "Prompt Throughput (tokens/s)",
                        f"{metrics['prompt_tokens_per_sec']:.2f}",
                    )
                if "generation_tokens_per_sec" in metrics:
                    table.add_row(
                        "Generation Throughput (tokens/s)",
                        f"{metrics['generation_tokens_per_sec']:.2f}",
                    )

                # Display cumulative totals
                table.add_row(
                    "Total Prompt Tokens",
                    f"{metrics.get('total_prompt_tokens', 0):.0f}",
                )
                table.add_row(
                    "Total Generation Tokens",
                    f"{metrics.get('total_generation_tokens', 0):.0f}",
                )

                # Other metrics
                table.add_row(
                    "Successful Requests",
                    f"{metrics.get('successful_requests_total', 0):.0f}",
                )
                table.add_row(
                    "Avg Request Latency (s)",
                    f"{metrics.get('request_latency_sum', 0):.2f}",
                )

            live.update(table)
            time.sleep(2)


if __name__ == "__main__":
    cli()
