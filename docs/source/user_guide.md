# User Guide

## Usage

### `launch` command

The `launch` command allows users to deploy a model as a slurm job. If the job successfully launches, a URL endpoint is exposed for the user to send requests for inference.

We will use the Llama 3.1 model as example, to launch an OpenAI compatible inference server for Meta-Llama-3.1-8B-Instruct, run:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct
```
You should see an output like the following:

<img width="600" alt="launch_img" src="https://github.com/user-attachments/assets/883e6a5b-8016-4837-8fdf-39097dfb18bf">

#### Overrides

Models that are already supported by `vec-inf` would be launched using the [default parameters](vec_inf/config/models.yaml). You can override these values by providing additional parameters. Use `vec-inf launch --help` to see the full list of parameters that can be overriden. For example, if `qos` is to be overriden:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct --qos <new_qos>
```

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
    num_gpus: 1
    num_nodes: 1
    vocab_size: 152064
    max_model_len: 1010000
    max_num_seqs: 256
    pipeline_parallelism: true
    enforce_eager: false
    qos: m2
    time: 08:00:00
    partition: a40
    model_weights_parent_dir: /h/<username>/model-weights
```

You would then set the `VEC_INF_CONFIG` path using:

```bash
export VEC_INF_CONFIG=/h/<username>/my-model-config.yaml
```

Note that there are other parameters that can also be added to the config but not shown in this example, such as `data_type` and `log_dir`.

### `status` command

You can check the inference server status by providing the Slurm job ID to the `status` command:

```bash
vec-inf status 15373800
```

If the server is pending for resources, you should see an output like this:

<img width="400" alt="status_pending_img" src="https://github.com/user-attachments/assets/b659c302-eae1-4560-b7a9-14eb3a822a2f">

When the server is ready, you should see an output like this:

<img width="400" alt="status_ready_img" src="https://github.com/user-attachments/assets/672986c2-736c-41ce-ac7c-1fb585cdcb0d">

There are 5 possible states:

* **PENDING**: Job submitted to Slurm, but not executed yet. Job pending reason will be shown.
* **LAUNCHING**: Job is running but the server is not ready yet.
* **READY**: Inference server running and ready to take requests.
* **FAILED**: Inference server in an unhealthy state. Job failed reason will be shown.
* **SHUTDOWN**: Inference server is shutdown/cancelled.

Note that the base URL is only available when model is in `READY` state, and if you've changed the Slurm log directory path, you also need to specify it when using the `status` command.

### `metrics` command

Once your server is ready, you can check performance metrics by providing the Slurm job ID to the `metrics` command:
```bash
vec-inf metrics 15373800
```

And you will see the performance metrics streamed to your console, note that the metrics are updated with a 2-second interval.

<img width="400" alt="metrics_img" src="https://github.com/user-attachments/assets/3ee143d0-1a71-4944-bbd7-4c3299bf0339">

### `shutdown` command

Finally, when you're finished using a model, you can shut it down by providing the Slurm job ID:
```bash
vec-inf shutdown 15373800

> Shutting down model with Slurm Job ID: 15373800
```

(list-command)=
### `list` command

You call view the full list of available models by running the `list` command:

```bash
vec-inf list
```

<img width="940" alt="list_img" src="https://github.com/user-attachments/assets/8cf901c4-404c-4398-a52f-0486f00747a3">


You can also view the default setup for a specific supported model by providing the model name, for example `Meta-Llama-3.1-70B-Instruct`:

```bash
vec-inf list Meta-Llama-3.1-70B-Instruct
```

<img width="500" alt="list_model_img" src="https://github.com/user-attachments/assets/34e53937-2d86-443e-85f6-34e408653ddb">

`launch`, `list`, and `status` command supports `--json-mode`, where the command output would be structured as a JSON string.

## Send inference requests

Once the inference server is ready, you can start sending in inference requests. We provide example scripts for sending inference requests in [`examples`](https://github.com/VectorInstitute/vector-inference/blob/main/examples) folder. Make sure to update the model server URL and the model weights location in the scripts. For example, you can run `python examples/inference/llm/chat_completions.py`, and you should expect to see an output like the following:

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


**NOTE**: For multimodal models, currently only `ChatCompletion` is available, and only one image can be provided for each prompt.

## SSH tunnel from your local device

If you want to run inference from your local device, you can open a SSH tunnel to your cluster environment like the following:
```bash
ssh -L 8081:172.17.8.29:8081 username@v.vectorinstitute.ai -N
```
Where the last number in the URL is the GPU number (gpu029 in this case). The example provided above is for the vector cluster, change the variables accordingly for your environment
