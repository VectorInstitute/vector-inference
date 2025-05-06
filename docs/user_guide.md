# User Guide

## CLI Usage

### `launch` command

The `launch` command allows users to deploy a model as a slurm job. If the job successfully launches, a URL endpoint is exposed for the user to send requests for inference.

We will use the Llama 3.1 model as example, to launch an OpenAI compatible inference server for Meta-Llama-3.1-8B-Instruct, run:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct
```
You should see an output like the following:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Job Config              ┃ Value                                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Slurm Job ID            │ 16060964                                  │
│ Job Name                │ Meta-Llama-3.1-8B-Instruct                │
│ Model Type              │ LLM                                       │
│ Vocabulary Size         │ 128256                                    │
│ Partition               │ a40                                       │
│ QoS                     │ m2                                        │
│ Time Limit              │ 08:00:00                                  │
│ Num Nodes               │ 1                                         │
│ GPUs/Node               │ 1                                         │
│ CPUs/Task               │ 16                                        │
│ Memory/Node             │ 64G                                       │
│ Model Weights Directory │ /model-weights/Meta-Llama-3.1-8B-Instruct │
│ Log Directory           │ /h/vi_user/.vec-inf-logs/Meta-Llama-3.1   │
│ vLLM Arguments:         │                                           │
│   --max-model-len:      │ 131072                                    │
│   --max-num-seqs:       │ 256                                       │
└─────────────────────────┴───────────────────────────────────────────┘
```

#### Overrides

Models that are already supported by `vec-inf` would be launched using the cached configuration or [default configuration](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/config/models.yaml). You can override these values by providing additional parameters. Use `vec-inf launch --help` to see the full list of parameters that can be overriden. For example, if `qos` is to be overriden:

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
  * Creating a custom configuration file for your model and specify its path via setting the environment variable `VEC_INF_CONFIG`. Check the [default parameters](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/config/models.yaml) file for the format of the config file. All the parameters for the model should be specified in that config file.
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
```

You would then set the `VEC_INF_CONFIG` path using:

```bash
export VEC_INF_CONFIG=/h/<username>/my-model-config.yaml
```

Note that there are other parameters that can also be added to the config but not shown in this example, check the [`ModelConfig`](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/client/config.py) for details.

### `status` command

You can check the inference server status by providing the Slurm job ID to the `status` command:

```bash
vec-inf status 15373800
```

If the server is pending for resources, you should see an output like this:

```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Job Status     ┃ Value                      ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Model Name     │ Meta-Llama-3.1-8B-Instruct │
│ Model Status   │ PENDING                    │
│ Pending Reason │ Resources                  │
│ Base URL       │ UNAVAILABLE                │
└────────────────┴────────────────────────────┘
```

When the server is ready, you should see an output like this:

```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Job Status   ┃ Value                      ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Model Name   │ Meta-Llama-3.1-8B-Instruct │
│ Model Status │ READY                      │
│ Base URL     │ http://gpu042:8080/v1      │
└──────────────┴────────────────────────────┘
```

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

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Metric                  ┃ Value           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Prompt Throughput       │ 10.9 tokens/s   │
│ Generation Throughput   │ 34.2 tokens/s   │
│ Requests Running        │ 1 reqs          │
│ Requests Waiting        │ 0 reqs          │
│ Requests Swapped        │ 0 reqs          │
│ GPU Cache Usage         │ 0.1%            │
│ CPU Cache Usage         │ 0.0%            │
│ Avg Request Latency     │ 2.6 s           │
│ Total Prompt Tokens     │ 441 tokens      │
│ Total Generation Tokens │ 1748 tokens     │
│ Successful Requests     │ 14 reqs         │
└─────────────────────────┴─────────────────┘
```

### `shutdown` command

Finally, when you're finished using a model, you can shut it down by providing the Slurm job ID:
```bash
vec-inf shutdown 15373800

> Shutting down model with Slurm Job ID: 15373800
```

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

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model Config             ┃ Value                      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ model_name               │ Meta-Llama-3.1-8B-Instruct │
│ model_family             │ Meta-Llama-3.1             │
│ model_variant            │ 8B-Instruct                │
│ model_type               │ LLM                        │
│ gpus_per_node            │ 1                          │
│ num_nodes                │ 1                          │
│ cpus_per_task            │ 16                         │
│ mem_per_node             │ 64G                        │
│ vocab_size               │ 128256                     │
│ qos                      │ m2                         │
│ time                     │ 08:00:00                   │
│ partition                │ a40                        │
│ model_weights_parent_dir │ /model-weights             │
│ vLLM Arguments:          │                            │
│   --max-model-len:       │ 131072                     │
│   --max-num-seqs:        │ 256                        │
└──────────────────────────┴────────────────────────────┘
```

`launch`, `list`, and `status` command supports `--json-mode`, where the command output would be structured as a JSON string.

## Check Job Configuration

With every model launch, a Slurm script will be generated dynamically based on the job and model configuration. Once the Slurm job is queued, the generated Slurm script will be moved to the log directory for reproducibility, located at `$log_dir/$model_family/$model_name.$slurm_job_id/$model_name.$slurm_job_id.slurm`. In the same directory you can also find a JSON file with the same name that captures the launch configuration, and will have an entry of server URL once the server is ready.

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

## Python API Usage

You can also use the `vec_inf` Python API to launch and manage inference servers.

Check out the [Python API documentation](api.md) for more details. There
are also Python API usage examples in the [`examples/api`](https://github.com/VectorInstitute/vector-inference/blob/main/examples/api) folder.
