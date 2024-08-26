import os

import click
import pandas as pd
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel

from _utils import *


CONSOLE = Console()


@click.group()
def cli():
    """Vector Inference CLI"""
    pass


@cli.command("launch")
@click.argument(
    "model-name",
    type=str,
    nargs=1
)
@click.option(
    "--model-family",
    type=str,
    help='The model family name according to the directories in `models`'
)
@click.option(
    "--model-variant",
    type=str,
    help='The model variant according to the README in `models/model-family`'
)
@click.option(
    "--max-model-len",
    type=int,
    help='Model context length. If unspecified, will be automatically derived from the model config.'
)
@click.option(
    "--partition",
    type=str,
    help='Type of compute partition, default to a40'
)
@click.option(
    "--num-nodes",
    type=int,
    help='Number of nodes to use, default to suggested resource allocation for model'
)
@click.option(
    "--num-gpus",
    type=int,
    help='Number of GPUs/node to use, default to suggested resource allocation for model'
)
@click.option(
    "--qos",
    type=str,
    help='Quality of service, default to m3'
)
@click.option(
    "--time",
    type=str,
    help='Time limit for job, this should comply with QoS, default to 4:00:00'
)
@click.option(
    "--data-type",
    type=str,
    help='Model data type, default to auto'
)
@click.option(
    "--venv",
    type=str,
    help='Path to virtual environment'
)
@click.option(
    "--log-dir",
    type=str,
    help='Path to slurm log directory'
)
@click.option(
    "--json-mode",
    is_flag=True,
    help='Output in JSON string',
)
def launch(
    model_name: str,
    model_family: str=None,
    model_variant: str=None,
    max_model_len: int=None,
    partition: str=None,
    num_nodes: int=None,
    num_gpus: int=None,
    qos: str=None,
    time: str=None,
    data_type: str=None,
    venv: str=None,
    log_dir: str=None,
    json_mode: bool=False
) -> None:
    """
    Launch a model on the cluster
    """
    launch_script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "launch_server.sh"
    )
    launch_cmd = f"bash {launch_script_path}" 

    models_df = load_models_df()

    if model_name not in models_df['model_name'].values:
        raise ValueError(f"Model name {model_name} not found in available models")

    default_args = load_default_args(models_df, model_name)

    for arg in default_args:
        if arg in locals() and locals()[arg] is not None:
            default_args[arg] = locals()[arg]
        renamed_arg = arg.replace("_", "-")
        launch_cmd += f" --{renamed_arg} {default_args[arg]}"    
    
    output = run_bash_command(launch_cmd)

    slurm_job_id = output.split(" ")[-1].strip().strip("\n")
    output_lines = output.split("\n")[:-2]

    table = create_table(key_title="Job Config", value_title="Value")
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
@click.argument(
    "slurm_job_id",
    type=int,
    nargs=1
)
@click.option(
    "--log-dir",
    type=str,
    help='Path to slurm log directory. This is required if it was set when launching the model'
)
@click.option(
    "--json-mode",
    is_flag=True,
    help='Output in JSON string',
)
def status(slurm_job_id: int, log_dir: str=None, json_mode: bool=False) -> None:
    """
    Get the status of a running model on the cluster
    """
    status_cmd = f"scontrol show job {slurm_job_id} --oneliner"
    output = run_bash_command(status_cmd)

    slurm_job_name = "UNAVAILABLE"
    status = "SHUTDOWN"
    base_url = "UNAVAILABLE"

    try:
        slurm_job_name = output.split(" ")[1].split("=")[1]
        slurm_job_state = output.split(" ")[9].split("=")[1]
    except IndexError:
        # Job ID not found
        slurm_job_state = "UNAVAILABLE"

    # If Slurm job is currently PENDING
    if slurm_job_state == "PENDING":
        slurm_job_pending_reason = output.split(" ")[10].split("=")[1]
        status = "PENDING"
    # If Slurm job is currently RUNNING
    elif slurm_job_state == "RUNNING":
        # Check whether the server is ready, if yes, run model health check to further determine status
        server_status = is_server_running(slurm_job_name, slurm_job_id, log_dir)
        if server_status == "RUNNING":
            status = model_health_check(slurm_job_name)
            if status == "READY":
                # Only set base_url if model is ready to serve requests
                base_url = get_base_url(slurm_job_name)
        else:
            status = server_status

    if json_mode:
        status_dict = {
            "model_name": slurm_job_name, 
            "model_status": status, 
            "base_url": base_url
        }
        if "slurm_job_pending_reason" in locals():
            status_dict["pending_reason"] = slurm_job_pending_reason
        click.echo(f'{status_dict}')
    else:
        table = create_table(key_title="Job Status", value_title="Value")
        table.add_row("Model Name", slurm_job_name)
        table.add_row("Model Status", status, style="blue")
        if "slurm_job_pending_reason" in locals():
            table.add_row("Reason", slurm_job_pending_reason)
        table.add_row("Base URL", base_url)
        CONSOLE.print(table)
        

@cli.command("shutdown")
@click.argument(
    "slurm_job_id",
    type=int,
    nargs=1
)
def shutdown(slurm_job_id: int) -> None:
    """
    Shutdown a running model on the cluster
    """
    shutdown_cmd = f"scancel {slurm_job_id}"
    run_bash_command(shutdown_cmd)
    click.echo(f"Shutting down model with Slurm Job ID: {slurm_job_id}")


@cli.command("list")
@click.option(
    "--json-mode",
    is_flag=True,
    help='Output in JSON string',
)
def list(json_mode: bool=False) -> None:
    """
    List all available models
    """
    models_df = load_models_df()
    if json_mode:
        click.echo(models_df['model_name'].to_json(orient='records'))
        return
    panels = []
    for _, row in models_df.iterrows():
        styled_text = f"[magenta]{row['model_family']}[/magenta]-{row['model_variant']}"
        panels.append(Panel(styled_text, expand=True))
    CONSOLE.print(Columns(panels, equal=True))


if __name__ == '__main__':
    cli()