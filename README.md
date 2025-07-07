# Vector Inference: Easy inference on Slurm clusters

----------------------------------------------------

[![PyPI](https://img.shields.io/pypi/v/vec-inf)](https://pypi.org/project/vec-inf)
[![downloads](https://img.shields.io/pypi/dm/vec-inf)](https://pypistats.org/packages/vec-inf)
[![code checks](https://github.com/VectorInstitute/vector-inference/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/vector-inference/actions/workflows/code_checks.yml)
[![docs](https://github.com/VectorInstitute/vector-inference/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/vector-inference/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/vector-inference/branch/main/graph/badge.svg?token=NI88QSIGAC)](https://app.codecov.io/github/VectorInstitute/vector-inference/tree/main)
[![vLLM](https://img.shields.io/badge/vllm-0.8.5.post1-blue)](https://docs.vllm.ai/en/v0.8.5.post1/index.html)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/vector-inference)

This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using [vLLM](https://docs.vllm.ai/en/latest/). **All scripts in this repository runs natively on the Vector Institute cluster environment**. To adapt to other environments, update the environment variables in [`vec_inf/client/slurm_vars.py`](vec_inf/client/slurm_vars.py), and the model config for cached model weights in [`vec_inf/config/models.yaml`](vec_inf/config/models.yaml) accordingly.

## Installation
If you are using the Vector cluster environment, and you don't need any customization to the inference server environment, run the following to install package:

```bash
pip install vec-inf
```
Otherwise, we recommend using the provided [`Dockerfile`](Dockerfile) to set up your own environment with the package. The latest image has `vLLM` version `0.8.5.post1`.

## Usage

Vector Inference provides 2 user interfaces, a CLI and an API

### CLI

The `launch` command allows users to deploy a model as a slurm job. If the job successfully launches, a URL endpoint is exposed for the user to send requests for inference.

We will use the Llama 3.1 model as example, to launch an OpenAI compatible inference server for Meta-Llama-3.1-8B-Instruct, run:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct
```
You should see an output like the following:

<img width="600" alt="launch_image" src="https://github.com/user-attachments/assets/a72a99fd-4bf2-408e-8850-359761d96c4f">


#### Overrides

Models that are already supported by `vec-inf` would be launched using the cached configuration (set in [slurm_vars.py](vec_inf/client/slurm_vars.py)) or [default configuration](vec_inf/config/models.yaml). You can override these values by providing additional parameters. Use `vec-inf launch --help` to see the full list of parameters that can be
overriden. For example, if `qos` is to be overriden:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct --qos <new_qos>
```

To overwrite default vLLM engine arguments, you can specify the engine arguments in a comma separated string:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct --vllm-args '--max-model-len=65536,--compilation-config=3'
```

For the full list of vLLM engine arguments, you can find them [here](https://docs.vllm.ai/en/stable/serving/engine_args.html), make sure you select the correct vLLM version.

#### Custom models

You can also launch your own custom model as long as the model architecture is [supported by vLLM](https://docs.vllm.ai/en/stable/models/supported_models.html), and make sure to follow the instructions below:
* Your model weights directory naming convention should follow `$MODEL_FAMILY-$MODEL_VARIANT` ($MODEL_VARIANT is OPTIONAL).
* Your model weights directory should contain HuggingFace format weights.
* You should specify your model configuration by:
  * Creating a custom configuration file for your model and specify its path via setting the environment variable `VEC_INF_CONFIG`. Check the [default parameters](vec_inf/config/models.yaml) file for the format of the config file. All the parameters for the model should be specified in that config file.
  * Using launch command options to specify your model setup.
* For other model launch parameters you can reference the default values for similar models using the [`list` command ](#list-command).

Here is an example to deploy a custom [Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M) model which is not
supported in the default list of models using a user custom config. In this case, the model weights are assumed to be downloaded to
a `model-weights` directory inside the user's home directory. The weights directory of the model follows the naming convention so it
would be named `Qwen2.5-7B-Instruct-1M`. The following yaml file would need to be created, lets say it is named `/h/<username>/my-model-config.yaml`.

```yaml
models:
  Qwen2.5-7B-Instruct-1M:
    model_family: Qwen2.5
    model_variant: 7B-Instruct-1M
    model_type: LLM
    gpus_per_node: 1
    num_nodes: 1
    vocab_size: 152064
    qos: m2
    time: 08:00:00
    partition: a40
    model_weights_parent_dir: /h/<username>/model-weights
    vllm_args:
      --max-model-len: 1010000
      --max-num-seqs: 256
      --compilation-config: 3
```

You would then set the `VEC_INF_CONFIG` path using:

```bash
export VEC_INF_CONFIG=/h/<username>/my-model-config.yaml
```

**NOTE**
* There are other parameters that can also be added to the config but not shown in this example, check the [`ModelConfig`](vec_inf/client/config.py) for details.
* Check [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/serving/engine_args.html) for the full list of available vLLM engine arguments, the default parallel size for any parallelization is default to 1, so none of the sizes were set specifically in this example
* For GPU partitions with non-Ampere architectures, e.g. `rtx6000`, `t4v2`, BF16 isn't supported. For models that have BF16 as the default type, when using a non-Ampere GPU, use FP16 instead, i.e. `--dtype: float16`
* Setting `--compilation-config` to `3` currently breaks multi-node model launches, so we don't set them for models that require multiple nodes of GPUs.

#### Other commands

* `status`: Check the model status by providing its Slurm job ID, `--json-mode` supported.
* `metrics`: Streams performance metrics to the console.
* `shutdown`: Shutdown a model by providing its Slurm job ID.
* `list`: List all available model names, or view the default/cached configuration of a specific model, `--json-mode` supported.
* `cleanup`: Remove old log directories. You can filter by `--model-family`, `--model-name`, `--job-id`, and/or `--before-job-id`. Use `--dry-run` to preview what would be deleted.

For more details on the usage of these commands, refer to the [User Guide](https://vectorinstitute.github.io/vector-inference/user_guide/)

### API

Example:

```python
>>> from vec_inf.api import VecInfClient
>>> client = VecInfClient()
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
ssh -L 8081:172.17.8.29:8081 username@v.vectorinstitute.ai -N
```
Where the last number in the URL is the GPU number (gpu029 in this case). The example provided above is for the vector cluster, change the variables accordingly for your environment
