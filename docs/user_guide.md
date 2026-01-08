# User Guide

## CLI Usage

### `launch` command

The `launch` command allows users to launch a OpenAI-compatible model inference server as a slurm job. If the job successfully launches, a URL endpoint is exposed for the user to send requests for inference.

We will use the Meta Llama 3.1 model as example, to launch an OpenAI compatible inference server for Meta-Llama-3.1-8B-Instruct, run:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct
```
You should see an output like the following:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Job Config              ┃ Value                                                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Slurm Job ID            │ 1673493                                                        │
│ Job Name                │ Meta-Llama-3.1-8B-Instruct                                     │
│ Model Type              │ LLM                                                            │
│ Vocabulary Size         │ 128256                                                         │
│ Account                 │ aip-your-account                                               │
│ Working Directory       │ /your/working/directory                                        │
│ Resource Type           │ l40s                                                           │
│ Time Limit              │ 08:00:00                                                       │
│ Num Nodes               │ 1                                                              │
│ GPUs/Node               │ 1                                                              │
│ CPUs/Task               │ 16                                                             │
│ Memory/Node             │ 64G                                                            │
│ Virtual Environment     │ /model-weights/vec-inf-shared/vector-inference-vllm_latest.sif │
│ Model Weights Directory │ /model-weights/Meta-Llama-3.1-8B-Instruct                      │
│ Log Directory           │ /home/marshw/.vec-inf-logs/Meta-Llama-3.1                      │
│ Inference Engine        │ vLLM                                                           │
└─────────────────────────┴────────────────────────────────────────────────────────────────┘
```

**NOTE**: You can set the required fields in the environment configuration (`environment.yaml`), it's a mapping between required arguments and their corresponding environment variables. On the Vector **Killarney** Cluster environment, the required fields are:

  * `--account`, `-A`: The Slurm account, this argument can be set to default by setting environment variable `VEC_INF_ACCOUNT`.
  * `--work-dir`, `-D`: A working directory other than your home directory, this argument can be set to default by seeting environment variable `VEC_INF_WORK_DIR`.

#### Overrides

Models that are already supported by `vec-inf` would be launched using the cached configuration or [default configuration](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/config). You can override these values by providing additional parameters. Use `vec-inf launch --help` to see the full list of parameters that can be overriden. For example, if `resource-type` is to be overriden:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct --resource-type <new_resource_type>
```

To overwrite default inference engine choice, use `--engine`:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct --engine sglang
```

**NOTE**: Some models are only supported by default inference engine, check supported model architectures from inference engine documentations.

To overwrite default inference engine arguments, you can specify the arguments in a comma separated string using `--$ENGINE_NAME-args`:

```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct --vllm-args '--max-model-len=65536,--compilation-config=3'
```

For the full list of inference engine arguments, you can find them here:

* [vLLM: `vllm serve` Arguments](https://docs.vllm.ai/en/stable/serving/engine_args.html)
* [SGLang: Server Arguments](https://docs.sglang.io/advanced_features/server_arguments.html)

#### Custom models

You can also launch your own custom model as long as the model architecture is supported by the underlying inference engine, and make sure to follow the instructions below:

* Your model weights directory naming convention should follow `$MODEL_FAMILY-$MODEL_VARIANT` ($MODEL_VARIANT is OPTIONAL).
* Your model weights directory should contain HuggingFace format weights.
* You should specify your model configuration by:
  * Creating a custom configuration file for your model and specify its path via setting the environment variable `VEC_INF_MODEL_CONFIG` (This one will supersede `VEC_INF_CONFIG_DIR` if that is also set). Check the [default parameters](vec_inf/config/models.yaml) file for the format of the config file. All the parameters for the model should be specified in that config file.
  * Add your model configuration to the cached `models.yaml` in your cluster environment (if you have write access to the cached configuration directory).
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
    resource_type: l40s # You can also omit this field empty if your environment has a default type of resource to use
    time: 08:00:00
    model_weights_parent_dir: /h/<username>/model-weights
    vllm_args:
      --max-model-len: 1010000
      --max-num-seqs: 256
```

You would then set the `VEC_INF_MODEL_CONFIG` path using:

```bash
export VEC_INF_MODEL_CONFIG=/h/<username>/my-model-config.yaml
```

**NOTE**: There are other parameters that can also be added to the config but not shown in this example, check the [`ModelConfig`](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/client/config.py) for details.

### `batch-launch` command

The `batch-launch` command allows users to launch multiple inference servers at once, here is an example of launching 2 models:

```bash
vec-inf batch-launch Qwen2.5-1.5B-Instruct Qwen2.5-Math-PRM-7B
```

You should see an output like the following:

```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Job Config        ┃ Value                                                                ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Slurm Job ID      │ 599425                                                               │
│ Slurm Job Name    │ BATCH-Qwen2.5-1.5B-Instruct-Qwen2.5-Math-PRM-7B                      │
│ Account           │ aip-your-account                                                     │
│ Working Directory │ /your/working/directory                                              │
│ Log Directory     │ /home/marshw/.vec-inf-logs/BATCH-Qwen2.5-1.5B-Instruct-Qwen2.5-Math… │
│ Model Name        │ Qwen2.5-1.5B-Instruct                                                │
│ Resource Type     │   l40s                                                               │
│ Time Limit        │   08:00:00                                                           │
│ Num Nodes         │   1                                                                  │
│ GPUs/Node         │   1                                                                  │
│ CPUs/Task         │   16                                                                 │
│ Memory/Node       │   64G                                                                │
│ Model Name        │ Qwen2.5-Math-PRM-7B                                                  │
│ Resource Type     │   l40s                                                               │
│ Time Limit        │   08:00:00                                                           │
│ Num Nodes         │   1                                                                  │
│ GPUs/Node         │   1                                                                  │
│ CPUs/Task         │   16                                                                 │
│ Memory/Node       │   64G                                                                │
└───────────────────┴──────────────────────────────────────────────────────────────────────┘
```

The inference servers will begin launching only after all requested resources have been allocated, preventing resource waste. Unlike the `launch` command, `batch-launch` does not accept additional launch parameters from the command line. Users must either:

- Specify a batch launch configuration file using the `--batch-config` option, or
- Ensure model launch configurations are available at the default location (cached config or user-defined `VEC_INF_CONFIG`)

Since batch launches use heterogeneous jobs, users can request different partitions and resource amounts for each model. After launch, you can monitor individual servers using the standard commands (`status`, `metrics`, etc.) by providing the specific Slurm job ID for each server (e.g. 17480109+0, 17480109+1).

**NOTE**
* Currently only models that can fit on a single node (regardless of the node type) is supported, multi-node launches will be available in a future update.

### `status` command

You can check the status of all inference servers launched through `vec-inf` by running the `status` command:
```bash
vec-inf status
```

And you should see an output like this:
```
┏━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Job ID    ┃ Model Name ┃ Status  ┃ Base URL              ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1434429   │ Qwen3-8B   │ READY   │ http://gpu113:8080/v1 │
│ 1434584   │ Qwen3-14B  │ READY   │ http://gpu053:8080/v1 │
│ 1435035+0 │ Qwen3-32B  │ PENDING │ UNAVAILABLE           │
│ 1435035+1 │ Qwen3-14B  │ PENDING │ UNAVAILABLE           │
└───────────┴────────────┴─────────┴───────────────────────┘
```

If you want to check why a specific job is pending or failing, append the job ID to the status command:

```bash
vec-inf status 1435035+1
```

If the server is pending for resources, you should see an output like this:

```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Job Status     ┃ Value       ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Model Name     │ Qwen3-14B   │
│ Model Status   │ PENDING     │
│ Pending Reason │ Resources   │
│ Base URL       │ UNAVAILABLE │
└────────────────┴─────────────┘
```

When the server is ready, you should see an output like this:

```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Job Status   ┃ Value                 ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Model Name   │ Qwen3-14B             │
│ Model Status │ READY                 │
│ Base URL     │ http://gpu105:8080/v1 │
└──────────────┴───────────────────────┘
```

There are 5 possible states:

* **PENDING**: Job submitted to Slurm, but not executed yet. Job pending reason will be shown.
* **LAUNCHING**: Job is running but the server is not ready yet.
* **READY**: Inference server running and ready to take requests.
* **FAILED**: Inference server in an unhealthy state. Job failed reason will be shown.
* **SHUTDOWN**: Inference server is shutdown/cancelled.

**Note**
* The base URL is only available when model is in `READY` state.
* For servers launched with `batch-launch`, the job ID should follow the format of "MAIN_JOB_ID+OFFSET" (e.g. 1435035+0, 1435035+1).

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
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model Config              ┃ Value                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ model_name                │ Meta-Llama-3.1-70B-Instruct │
│ model_family              │ Meta-Llama-3.1              │
│ model_variant             │ 70B-Instruct                │
│ model_type                │ LLM                         │
│ gpus_per_node             │ 4                           │
│ num_nodes                 │ 1                           │
│ cpus_per_task             │ 16                          │
│ mem_per_node              │ 64G                         │
│ vocab_size                │ 128256                      │
│ time                      │ 08:00:00                    │
│ resource_type             │ l40s                        │
│ model_weights_parent_dir  │ /model-weights              │
│ vLLM Arguments:           │                             │
│   --tensor-parallel-size: │ 4                           │
│   --max-model-len:        │ 65536                       │
│   --max-num-seqs:         │ 256                         │
└───────────────────────────┴─────────────────────────────┘
```

`launch`, `list`, and `status` command supports `--json-mode`, where the command output would be structured as a JSON string.

### `cleanup` command

To avoid log build up and maintaining a clean working environment, you can use the `cleanup` command to remove old logs:
```bash
vec-inf cleanup [OPTIONS]
```

You can use the following filters to select which logs you would like to remove:
| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--log-dir` | - | string | Path to SLURM log directory (defaults to `~/.vec-inf-logs`) |
| `--model-family` | - | string | Filter logs by model family (e.g., "llama", "gpt") |
| `--model-name` | - | string | Filter logs by specific model name |
| `--job-id` | - | integer | Only remove logs with this exact SLURM job ID |
| `--before-job-id` | - | integer | Remove logs with job ID less than this value |
| `--dry-run` | - | flag | List matching logs without deleting them |


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

**NOTE**: Certain models don't adhere to OpenAI's chat template, e.g. Mistral family. For these models, you can either change your prompt to follow the model's default chat template or provide your own chat template via `--chat-template: TEMPLATE_PATH`.

## SSH tunnel from your local device

If you want to run inference from your local device, you can open a SSH tunnel to your cluster environment like the following:
```bash
ssh -L 8081:10.1.1.29:8081 username@v.vectorinstitute.ai -N
```
The example provided above is for the Vector Killarney cluster, change the variables accordingly for your environment. The IP address for the compute nodes on Killarney follow `10.1.1.XX` pattern, where `XX` is the GPU number (`kn029` -> `29` in this example). Similarly, for Bon Echo it's `172.17.8.XX`, where `XX` is from `gpuXX`.

## Python API Usage

You can also use the `vec_inf` Python API to launch and manage inference servers.

Check out the [Python API documentation](api.md) for more details. There
are also Python API usage examples in the [`examples/api`](https://github.com/VectorInstitute/vector-inference/blob/main/examples/api) folder.
