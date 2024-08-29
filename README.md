# Vector Inference: Easy inference on Slurm clusters
This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using [vLLM](https://docs.vllm.ai/en/latest/). **All scripts in this repository runs natively on the Vector Institute cluster environment**. To adapt to other environments, update [`launch_server.sh`](vec-inf/launch_server.sh), [`vllm.slurm`](vec-inf/vllm.slurm), [`multinode_vllm.slurm`](vec-inf/multinode_vllm.slurm) and [`models.csv`](vec-inf/models/models.csv) accordingly.  

## Installation
If you are using the Vector cluster environment, and you don't need any customization to the inference server environment, run the following to install package:
```bash
pip install vec-inf
```
Otherwise, we recommend using the provided [`Dockerfile`](Dockerfile) to set up your own environment with the package

## Launch an inference server
We will use the Llama 3.1 model as example, to launch an OpenAI compatible inference server for Meta-Llama-3.1-8B-Instruct, run:
```bash
vec-inf launch Meta-Llama-3.1-8B-Instruct
```
You should see an output like the following:

<img width="450" alt="launch_img" src="https://github.com/user-attachments/assets/557eb421-47db-4810-bccd-c49c526b1b43">

The model would be launched using the [default parameters](vec-inf/models/models.csv), you can override these values by providing additional options, use `--help` to see the full list.
If you'd like to see the Slurm logs, they are located in the `.vec-inf-logs` folder in your home directory. The log folder path can be modified by using the `--log-dir` option. 

You can check the inference server status by providing the Slurm job ID to the `status` command:
```bash
vec-inf status 13014393
```

You should see an output like the following:

<img width="450" alt="status_img" src="https://github.com/user-attachments/assets/7385b9ca-9159-4ca9-bae2-7e26d80d9747">

There are 5 possible states:

* **PENDING**: Job submitted to Slurm, but not executed yet. Job pending reason will be shown.
* **LAUNCHING**: Job is running but the server is not ready yet.
* **READY**: Inference server running and ready to take requests. 
* **FAILED**: Inference server in an unhealthy state. Job failed reason will be shown.
* **SHUTDOWN**: Inference server is shutdown/cancelled.

Note that the base URL is only available when model is in `READY` state, and if you've changed the Slurm log directory path, you also need to specify it when using the `status` command.

Finally, when you're finished using a model, you can shut it down by providing the Slurm job ID:
```bash
vec-inf shutdown 13014393

> Shutting down model with Slurm Job ID: 13014393
```

You call view the full list of available models by running the `list` command:
```bash
vec-inf list
```
<img width="1200" alt="list_img" src="https://github.com/user-attachments/assets/a4f0d896-989d-43bf-82a2-6a6e5d0d288f">

`launch`, `list`, and `status` command supports `--json-mode`, where the command output would be structured as a JSON string.

## Send inference requests
Once the inference server is ready, you can start sending in inference requests. We provide example scripts for sending inference requests in [`examples`](examples) folder. Make sure to update the model server URL and the model weights location in the scripts. For example, you can run `python examples/inference/llm/completions.py`, and you should expect to see an output like the following:
> {"id":"cmpl-bdf43763adf242588af07af88b070b62","object":"text_completion","created":2983960,"model":"/model-weights/Llama-2-7b-hf","choices":[{"index":0,"text":"\nCanada is close to the actual continent of North America. Aside from the Arctic islands","logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":8,"total_tokens":28,"completion_tokens":20}}

**NOTE**: For multimodal models, currently only `ChatCompletion` is available, and only one image can be provided for each prompt.

## SSH tunnel from your local device
If you want to run inference from your local device, you can open a SSH tunnel to your cluster environment like the following:
```bash
ssh -L 8081:172.17.8.29:8081 username@v.vectorinstitute.ai -N
```
Where the last number in the URL is the GPU number (gpu029 in this case). The example provided above is for the vector cluster, change the variables accordingly for your environment
