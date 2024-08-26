import subprocess
import os

import requests
from rich.table import Table
import pandas as pd


MODEL_READY_SIGNATURE = "INFO:     Uvicorn running on http://0.0.0.0:"


def run_bash_command(command: str) -> str:
    """
    Run a bash command and return the output
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, _ = process.communicate()
    return stdout


def get_model_dir(slurm_job_name: str, is_log_dir: bool=False) -> str:
    """
    Get the directory of a model
    """
    if is_log_dir:
        models_dir = os.path.join(os.path.expanduser("~"), ".vec-inf-logs")
    else:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "models"
        )
    model_dir = ""
    for dir in sorted(os.listdir(models_dir), key=len, reverse=True):
        if dir in slurm_job_name:
            model_dir = os.path.join(models_dir, dir)
            break

    return model_dir


def is_server_running(slurm_job_name: str, slurm_job_id: int, log_dir: str) -> str:
    """
    Check if a model is ready to serve requests
    """
    if not log_dir:
        log_dir = get_model_dir(slurm_job_name, is_log_dir=True)
    
    try:
        file_path = os.path.join(log_dir, f"{slurm_job_name}.{slurm_job_id}.err")
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Could not find file: {file_path}")
        return "LOG_FILE_NOT_FOUND"
    
    for line in lines:
        if MODEL_READY_SIGNATURE in line:
            return "RUNNING"
    return "LAUNCHING"


def get_base_url(slurm_job_name: str) -> str:
    """
    Get the base URL of a model
    """
    model_dir = get_model_dir(slurm_job_name)
    try:
        file_path = os.path.join(model_dir, f".{slurm_job_name}_url")
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return "UNAVAILABLE"
    return lines[0].strip().strip("\n")


def model_health_check(slurm_job_name: str) -> str:
    """
    Check the health of a running model on the cluster
    """
    base_url = get_base_url(slurm_job_name)
    health_check_url = base_url.replace("v1", "health")

    try:
        response = requests.get(health_check_url)
        # Check if the request was successful
        if response.status_code == 200:
            return "READY"
        else:
            return "FAILED"
    except requests.exceptions.RequestException as e:
        return "FAILED"
    

def create_table(key_title: str = "", value_title: str = "", show_header: bool = True) -> Table:
    """
    Create a table for displaying model status
    """
    table = Table(show_header=show_header, header_style="bold magenta")
    table.add_column(key_title, style="dim")
    table.add_column(value_title)
    return table


def load_models_df() -> pd.DataFrame:
    """
    Load the models dataframe
    """
    models_df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "models/models.csv"
        )
    )
    return models_df


def load_default_args(models_df: pd.DataFrame, model_name: str) -> dict:
    """
    Load the default arguments for a model
    """
    row_data = models_df.loc[models_df["model_name"] == model_name]
    default_args = row_data.iloc[0].to_dict()
    default_args.pop("model_name")
    return default_args