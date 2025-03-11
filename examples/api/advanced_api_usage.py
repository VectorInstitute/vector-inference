#!/usr/bin/env python
"""Advanced usage examples for the Vector Inference Python API.

This script demonstrates more advanced patterns and techniques for
using the Vector Inference API programmatically.
"""

import argparse
import json
import time
from typing import Dict, Union

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from vec_inf.api import (
    LaunchOptions,
    LaunchOptionsDict,
    ModelStatus,
    StatusResponse,
    VecInfClient,
)


console = Console()


def create_openai_client(base_url: str) -> OpenAI:
    """Create an OpenAI client for a given base URL."""
    return OpenAI(base_url=base_url, api_key="EMPTY")


def export_model_configs(output_file: str) -> None:
    """Export all model configurations to a JSON file."""
    client = VecInfClient()
    models = client.list_models()

    # Convert model info to dictionaries
    model_dicts = []
    for model in models:
        model_dict = {
            "name": model.name,
            "family": model.family,
            "variant": model.variant,
            "type": str(model.type),
            "config": model.config,
        }
        model_dicts.append(model_dict)

    # Write to file
    with open(output_file, "w") as f:
        json.dump(model_dicts, f, indent=2)

    console.print(
        f"[green]Exported {len(models)} model configurations to {output_file}[/green]"
    )


def launch_with_custom_config(
    model_name: str, custom_options: Dict[str, Union[str, int, bool]]
) -> str:
    """Launch a model with custom configuration options."""
    client = VecInfClient()

    # Create LaunchOptions from dictionary
    options_dict: LaunchOptionsDict = {}
    for key, value in custom_options.items():
        if key in LaunchOptions.__annotations__:
            options_dict[key] = value  # type: ignore[literal-required]
        else:
            console.print(f"[yellow]Warning: Ignoring unknown option '{key}'[/yellow]")

    options = LaunchOptions(**options_dict)

    # Launch the model
    console.print(f"[blue]Launching model {model_name} with custom options:[/blue]")
    for key, value in options_dict.items():  # type: ignore[assignment]
        console.print(f" [cyan]{key}[/cyan]: {value}")

    response = client.launch_model(model_name, options)

    console.print("[green]Model launched successfully![/green]")
    console.print(f"Slurm Job ID: [bold]{response.slurm_job_id}[/bold]")

    return response.slurm_job_id


def monitor_with_rich_ui(
    job_id: str, poll_interval: int = 5, max_time: int = 1800
) -> StatusResponse:
    """Monitor a model's status with a rich UI."""
    client = VecInfClient()

    start_time = time.time()
    elapsed = 0

    with Progress() as progress:
        # Add tasks
        status_task = progress.add_task(
            "[cyan]Waiting for model to be ready...", total=None
        )
        time_task = progress.add_task("[yellow]Time elapsed", total=max_time)

        while elapsed < max_time:
            # Update time elapsed
            elapsed = int(time.time() - start_time)
            progress.update(time_task, completed=elapsed)

            # Get status
            try:
                status = client.get_status(job_id)

                # Update status message
                if status.status == ModelStatus.READY:
                    progress.update(
                        status_task,
                        description=f"[green]Model is READY at {status.base_url}[/green]",
                    )
                    break
                if status.status == ModelStatus.FAILED:
                    progress.update(
                        status_task,
                        description=f"[red]Model FAILED: {status.failed_reason}[/red]",
                    )
                    break
                if status.status == ModelStatus.PENDING:
                    progress.update(
                        status_task,
                        description=f"[yellow]Model is PENDING: {status.pending_reason}[/yellow]",
                    )
                elif status.status == ModelStatus.LAUNCHING:
                    progress.update(
                        status_task, description="[cyan]Model is LAUNCHING...[/cyan]"
                    )
                elif status.status == ModelStatus.SHUTDOWN:
                    progress.update(
                        status_task, description="[red]Model was SHUTDOWN[/red]"
                    )
                    break
            except Exception as e:
                progress.update(
                    status_task,
                    description=f"[red]Error checking status: {str(e)}[/red]",
                )

            # Wait before checking again
            time.sleep(poll_interval)

    return client.get_status(job_id)


def stream_metrics(job_id: str, duration: int = 60, interval: int = 5) -> None:
    """Stream metrics for a specified duration."""
    client = VecInfClient()

    console.print(f"[blue]Streaming metrics for {duration} seconds...[/blue]")

    end_time = time.time() + duration
    while time.time() < end_time:
        try:
            metrics_response = client.get_metrics(job_id)

            if metrics_response.metrics:
                table = Table(title="Performance Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                for key, value in metrics_response.metrics.items():
                    table.add_row(key, value)

                console.print(table)
            else:
                console.print("[yellow]No metrics available yet[/yellow]")

        except Exception as e:
            console.print(f"[red]Error retrieving metrics: {str(e)}[/red]")

        time.sleep(interval)


def batch_inference_example(
    base_url: str, model_name: str, input_file: str, output_file: str
) -> None:
    """Perform batch inference on inputs from a file."""
    # Read inputs
    with open(input_file, "r") as f:
        inputs = [line.strip() for line in f if line.strip()]

    openai_client = create_openai_client(base_url)

    results = []
    with Progress() as progress:
        task = progress.add_task("[green]Processing inputs...", total=len(inputs))

        for input_text in inputs:
            try:
                # Process using completions API
                completion = openai_client.completions.create(
                    model=model_name,
                    prompt=input_text,
                    max_tokens=100,
                )

                # Store result
                results.append(
                    {
                        "input": input_text,
                        "output": completion.choices[0].text,
                        "tokens": completion.usage.completion_tokens,
                    }
                )

            except Exception as e:
                results.append({"input": input_text, "error": str(e)})

            progress.update(task, advance=1)

    # Write results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(
        f"[green]Processed {len(inputs)} inputs and saved results to {output_file}[/green]"
    )


def main() -> None:
    """Parse arguments and run the selected function."""
    parser = argparse.ArgumentParser(
        description="Advanced Vector Inference API usage examples"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Export configs command
    export_parser = subparsers.add_parser(
        "export-configs", help="Export all model configurations to a JSON file"
    )
    export_parser.add_argument(
        "--output", "-o", default="model_configs.json", help="Output JSON file"
    )

    # Launch with custom config command
    launch_parser = subparsers.add_parser(
        "launch", help="Launch a model with custom configuration"
    )
    launch_parser.add_argument("model_name", help="Name of the model to launch")
    launch_parser.add_argument("--num-gpus", type=int, help="Number of GPUs to use")
    launch_parser.add_argument("--num-nodes", type=int, help="Number of nodes to use")
    launch_parser.add_argument(
        "--max-model-len", type=int, help="Maximum model context length"
    )
    launch_parser.add_argument(
        "--max-num-seqs", type=int, help="Maximum number of sequences"
    )
    launch_parser.add_argument("--partition", help="Partition to use")
    launch_parser.add_argument("--qos", help="Quality of service")
    launch_parser.add_argument("--time", help="Time limit")

    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor a model with rich UI"
    )
    monitor_parser.add_argument("job_id", help="Slurm job ID to monitor")
    monitor_parser.add_argument(
        "--interval", type=int, default=5, help="Polling interval in seconds"
    )
    monitor_parser.add_argument(
        "--max-time", type=int, default=1800, help="Maximum time to monitor in seconds"
    )

    # Stream metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Stream metrics for a model")
    metrics_parser.add_argument("job_id", help="Slurm job ID to get metrics for")
    metrics_parser.add_argument(
        "--duration", type=int, default=60, help="Duration to stream metrics in seconds"
    )
    metrics_parser.add_argument(
        "--interval", type=int, default=5, help="Polling interval in seconds"
    )

    # Batch inference command
    batch_parser = subparsers.add_parser("batch", help="Perform batch inference")
    batch_parser.add_argument("base_url", help="Base URL of the model server")
    batch_parser.add_argument("model_name", help="Name of the model to use")
    batch_parser.add_argument(
        "--input", "-i", required=True, help="Input file with one prompt per line"
    )
    batch_parser.add_argument(
        "--output", "-o", required=True, help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Run the selected command
    if args.command == "export-configs":
        export_model_configs(args.output)

    elif args.command == "launch":
        # Extract custom options from args
        options = {}
        for key, value in vars(args).items():
            if key not in ["command", "model_name"] and value is not None:
                options[key] = value

        job_id = launch_with_custom_config(args.model_name, options)

        # Ask if user wants to monitor
        if console.input("[cyan]Monitor this job? (y/n): [/cyan]").lower() == "y":
            monitor_with_rich_ui(job_id)

    elif args.command == "monitor":
        status = monitor_with_rich_ui(args.job_id, args.interval, args.max_time)

        if (status.status == ModelStatus.READY) and (
            console.input("[cyan]Stream metrics for this model? (y/n): [/cyan]").lower()
            == "y"
        ):
            stream_metrics(args.job_id)

    elif args.command == "metrics":
        stream_metrics(args.job_id, args.duration, args.interval)

    elif args.command == "batch":
        batch_inference_example(args.base_url, args.model_name, args.input, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
