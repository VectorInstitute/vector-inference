# Vector Inference: Easy inference on Slurm clusters
This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using [vLLM](https://docs.vllm.ai/en/latest/). **All scripts in this repository runs natively on the Vector Institute cluster environment**. To adapt to other environments, update the config files in the `vec_inf/models` folder and the environment variables in the model launching scripts in `vec_inf` accordingly.  

## Installation
If you are using the Vector cluster environment, and you don't need any customization to the inference server environment, run the following to install package:
```bash
pip install vec-inf
```
Otherwise, we recommend using the provided [`Dockerfile`](Dockerfile) to set up your own environment with the package

## Launch an inference server
We will use the Llama 3 model as example, to launch an inference server for Llama 3 8B, run:
```bash
vec-inf launch llama-3
```
You should see an output like the following:

<img src="https://github.com/user-attachments/assets/c50646df-0991-4164-ad8f-6eb7e86b67e0" width="350">

There is a default variant for every model family, which is specified in `vec_inf/models/{MODEL_FAMILY_NAME}/README.md`, you can switch to other variants with the `--model-variant` option, and make sure to change the requested resource accordingly. More information about the available options can be found in the [`vec_inf/models`](vec_inf/models) folder. The inference server is compatible with the OpenAI `Completion` and `ChatCompletion` API. 

You can check the inference server status by providing the Slurm job ID to the `status` command:
```bash
vec-inf status 13014393
```

You should see an output like the following:

<img src="https://github.com/user-attachments/assets/310086fd-82ea-4bfc-8062-5c8e71c5650c" width="400">

There are 5 possible states:

* **PENDING**: Job submitted to Slurm, but not executed yet.
* **LAUNCHING**: Job is running but the server is not ready yet.
* **READY**: Inference server running and ready to take requests.
* **FAILED**: Inference server in an unhealthy state.
* **SHUTDOWN**: Inference server is shutdown/cancelled.

Note that the base URL is only available when model is in `READY` state. 
Both `launch` and `status` command supports `--json-mode`, where the output information would be structured as a JSON string.

Finally, when you're finished using a model, you can shut it down by providing the Slurm job ID:
```bash
vec-inf shutdown 13014393

> Shutting down model with Slurm Job ID: 13014393
```

Here is a more complicated example that launches a model variant using multiple nodes, say we want to launch Mixtral 8x22B, run
```bash
vec-inf launch mixtral --model-variant 8x22B-v0.1 --num-nodes 2 --num-gpus 4
```

And for launching a multimodal model, here is an example for launching LLaVa-NEXT Mistral 7B (default variant)
```bash
vec-inf launch llava-v1.6 --is-vlm 
```

## Send inference requests
Once the inference server is ready, you can start sending in inference requests. We provide example scripts for sending inference requests in [`examples`](examples) folder. Make sure to update the model server URL and the model weights location in the scripts. For example, you can run `python examples/inference/llm/completions.py`, and you should expect to see an output like the following:
> {"id":"cmpl-bdf43763adf242588af07af88b070b62","object":"text_completion","created":2983960,"model":"/model-weights/Llama-2-7b-hf","choices":[{"index":0,"text":"\nCanada is close to the actual continent of North America. Aside from the Arctic islands","logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":8,"total_tokens":28,"completion_tokens":20}}

**NOTE**: For multimodal models, currently only `ChatCompletion` is available, and only one image can be provided for each prompt.

## SSH tunnel from your local device
If you want to run inference from your local device, you can open a SSH tunnel to your cluster environment like the following:
```bash
ssh -L 8081:172.17.8.29:8081 username@v.vectorinstitute.ai -N
```
The example provided above is for the vector cluster, change the variables accordingly for your environment
