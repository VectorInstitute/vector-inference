# Vector Inference: Easy inference on Slurm clusters

----------------------------------------------------

[![PyPI](https://img.shields.io/pypi/v/vec-inf)](https://pypi.org/project/vec-inf)
[![downloads](https://img.shields.io/pypi/dm/vec-inf)](https://pypistats.org/packages/vec-inf)
[![code checks](https://github.com/VectorInstitute/vector-inference/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/vector-inference/actions/workflows/code_checks.yml)
[![docs](https://github.com/VectorInstitute/vector-inference/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/vector-inference/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/vector-inference/branch/main/graph/badge.svg?token=NI88QSIGAC)](https://app.codecov.io/github/VectorInstitute/vector-inference/tree/main)
[![vLLM](https://img.shields.io/badge/vLLM-0.10.1.1-blue)](https://docs.vllm.ai/en/v0.10.1.1/)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/vector-inference)

This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using [vLLM](https://docs.vllm.ai/en/latest/). **This package runs natively on the Vector Institute cluster environments**. To adapt to other environments, follow the instructions in [Installation](#installation).

**NOTE**: Supported models on Killarney are tracked [here](./MODEL_TRACKING.md)

## Installation
If you are using the Vector cluster environment, and you don't need any customization to the inference server environment, run the following to install package:

```bash
pip install vec-inf
```
Otherwise, we recommend using the provided [`Dockerfile`](Dockerfile) to set up your own environment with the package. The latest image has `vLLM` version `0.10.1.1`.

If you'd like to use `vec-inf` on your own Slurm cluster, you would need to update the configuration files, there are 3 ways to do it:
* Clone the repository and update the `environment.yaml` and the `models.yaml` file in [`vec_inf/config`](vec_inf/config/), then install from source by running `pip install .`.
* The package would try to look for cached configuration files in your environment before using the default configuration. The default cached configuration directory path points to `/model-weights/vec-inf-shared`, you would need to create an `environment.yaml` and a `models.yaml` following the format of these files in [`vec_inf/config`](vec_inf/config/).
* The package would also look for an enviroment variable `VEC_INF_CONFIG_DIR`. You can put your `environment.yaml` and `models.yaml` in a directory of your choice and set the enviroment variable `VEC_INF_CONFIG_DIR` to point to that location.

## Usage

Vector Inference provides 2 user interfaces, a CLI and an API

### CLI

The `launch` command allows users to deploy a model as a slurm job. If the job successfully launches, a URL endpoint is exposed for the user to send requests for inference.

We will use the Llama 3.1 model as example, to launch an OpenAI compatible inference server for Meta-Llama-3.1-8B-Instruct, run:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct
```
You should see an output like the following:

<img width="720" alt="launch_image" src="https://github.com/user-attachments/assets/c1e0c60c-cf7a-49ed-a426-fdb38ebf88ee" />

**NOTE**: On Vector Killarney Cluster environment, the following fields are required:
  * `--account`, `-A`: The Slurm account, this argument can be set to default by setting environment variable `VEC_INF_ACCOUNT`.
  * `--work-dir`, `-D`: A working directory other than your home directory, this argument can be set to default by seeting environment variable `VEC_INF_WORK_DIR`.

Models that are already supported by `vec-inf` would be launched using the cached configuration (set in [slurm_vars.py](vec_inf/client/slurm_vars.py)) or [default configuration](vec_inf/config/models.yaml). You can override these values by providing additional parameters. Use `vec-inf launch --help` to see the full list of parameters that can be overriden. You can also launch your own custom model as long as the model architecture is [supported by vLLM](https://docs.vllm.ai/en/stable/models/supported_models.html). For detailed instructions on how to customize your model launch, check out the [`launch` command section in User Guide](https://vectorinstitute.github.io/vector-inference/latest/user_guide/#launch-command)

#### Other commands

* `batch-launch`: Launch multiple model inference servers at once, currently ONLY single node models supported,
* `status`: Check the model status by providing its Slurm job ID.
* `metrics`: Streams performance metrics to the console.
* `shutdown`: Shutdown a model by providing its Slurm job ID.
* `list`: List all available model names, or view the default/cached configuration of a specific model.
* `cleanup`: Remove old log directories, use `--help` to see the supported filters. Use `--dry-run` to preview what would be deleted.

For more details on the usage of these commands, refer to the [User Guide](https://vectorinstitute.github.io/vector-inference/user_guide/)

### API

Example:

```python
>>> from vec_inf.api import VecInfClient
>>> client = VecInfClient()
>>> # Assume VEC_INF_ACCOUNT and VEC_INF_WORK_DIR is set
>>> response = client.launch_model("Meta-Llama-3.1-8B-Instruct")
>>> job_id = response.slurm_job_id
>>> status = client.get_status(job_id)
>>> if status.status == ModelStatus.READY:
...     print(f"Model is ready at {status.base_url}")
>>> client.shutdown_model(job_id)
```

For details on the usage of the API, refer to the [API Reference](https://vectorinstitute.github.io/vector-inference/api/)

## Check Job Configuration

With every model launch, a Slurm script will be generated dynamically based on the job and model configuration. Once the Slurm job is queued, the generated Slurm script will be moved to the log directory for reproducibility, located at `$log_dir/$model_family/$model_name.$slurm_job_id/$model_name.$slurm_job_id.slurm`. In the same directory you can also find a JSON file with the same name that captures the launch configuration, and will have an entry of server URL once the server is ready.

## Send inference requests

Once the inference server is ready, you can start sending in inference requests. We provide example scripts for sending inference requests in [`examples`](examples) folder. Make sure to update the model server URL and the model weights location in the scripts. For example, you can run `python examples/inference/llm/chat_completions.py`, and you should expect to see an output like the following:

```json
{
    "id":"chatcmpl-387c2579231948ffaf66cdda5439d3dc",
    "choices": [
        {
            "finish_reason":"stop",
            "index":0,
            "logprobs":null,
            "message": {
                "content":"Arrr, I be Captain Chatbeard, the scurviest chatbot on the seven seas! Ye be wantin' to know me identity, eh? Well, matey, I be a swashbucklin' AI, here to provide ye with answers and swappin' tales, savvy?",
                "role":"assistant",
                "function_call":null,
                "tool_calls":[],
                "reasoning_content":null
            },
            "stop_reason":null
        }
    ],
    "created":1742496683,
    "model":"Meta-Llama-3.1-8B-Instruct",
    "object":"chat.completion",
    "system_fingerprint":null,
    "usage": {
        "completion_tokens":66,
        "prompt_tokens":32,
        "total_tokens":98,
        "prompt_tokens_details":null
    },
    "prompt_logprobs":null
}

```
**NOTE**: Certain models don't adhere to OpenAI's chat template, e.g. Mistral family. For these models, you can either change your prompt to follow the model's default chat template or provide your own chat template via `--chat-template: TEMPLATE_PATH`.

## SSH tunnel from your local device
If you want to run inference from your local device, you can open a SSH tunnel to your cluster environment like the following:
```bash
ssh -L 8081:10.1.1.29:8081 username@v.vectorinstitute.ai -N
```
The example provided above is for the Vector Killarney cluster, change the variables accordingly for your environment. The IP address for the compute nodes on Killarney follow `10.1.1.XX` pattern, where `XX` is the GPU number (`kn029` -> `29` in this example).
