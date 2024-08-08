import os

import click
from rich.console import Console

from ._utils import run_bash_command, is_server_running, model_health_check, get_base_url, create_table


console = Console()


@click.group()
def cli():
    """Vector Inference CLI"""
    pass

@cli.command("launch")
@click.argument(
    "model-family",
    type=str,
    nargs=1
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
    "--json-mode",
    is_flag=True,
    help='Output in JSON string',
)
def launch(
    model_family: str,
    model_variant: str=None,
    max_model_len: int=None,
    partition: str=None,
    num_nodes: int=None,
    num_gpus: int=None,
    qos: str=None,
    time: str=None,
    data_type: str=None,
    venv: str=None,
    image_input_type: str=None,
    image_token_id: str=None,
    image_input_shape: str=None,
    image_feature_size: str=None,
    json_mode: bool=False
) -> None:
    """
    Launch a model on the cluster
    """
    input_args_list = list(locals().keys())
    input_args_list.remove("json_mode")
    launch_script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "launch_server.sh"
    )
    launch_cmd = f"bash {launch_script_path}" 
    for arg in input_args_list:
        if locals()[arg] is not None:
            named_arg = arg.replace("_", "-")
            launch_cmd += f" --{named_arg} {locals()[arg]}"
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
        console.print(table)


@cli.command("status")
@click.argument(
    "slurm_job_id",
    type=int,
    nargs=1
)
@click.option(
    "--json-mode",
    is_flag=True,
    help='Output in JSON string',
)
def status(slurm_job_id: int, json_mode: bool=False) -> None:
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
        # If Slurm job is currently PENDING
        if slurm_job_state == "PENDING":
            status = "PENDING"
        # If Slurm job is currently RUNNING
        elif slurm_job_state == "RUNNING":
            # Check whether the server is ready, if yes, run model health check to further determine status
            server_status = is_server_running(slurm_job_name, slurm_job_id)
            if server_status == "RUNNING":
                status = model_health_check(slurm_job_name)
                if status == "READY":
                    # Only set base_url if model is ready to serve requests
                    base_url = get_base_url(slurm_job_name)
            else:
                status = server_status
    except:
        pass

    if json_mode:
        click.echo(f'{{"model_name": "{slurm_job_name}", "model_status": "{status}", "base_url": "{base_url}"}}')
    else:
        table = create_table(key_title="Job Status", value_title="Value")
        table.add_row("Model Name", slurm_job_name)
        table.add_row("Model Status", status, style="blue")
        table.add_row("Base URL", base_url)
        console.print(table)
        

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


if __name__ == '__main__':
    cli()