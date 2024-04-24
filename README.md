# Vector Inference: Easy inference on Slurm clusters
This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using [vLLM](https://docs.vllm.ai/en/latest/). All scripts in this repository runs natively on the Vector Institute cluster environment, and can be easily adapted to other environments.  

## Installation
If you are using the Vector cluster environment, and you don't need any customization to the inference server environment, you can skip this step and go to the next section. Otherwise, you might need up to 10GB of storage to setup your own virtual environment. The following steps needs to be run only once for each user.

1. Setup the virtual environment for running inference servers, run 
```bash
bash venv.sh
```
More details can be found in [venv.sh](venv.sh), make sure to adjust the commands to your environment if you're not using the Vector cluster.

2. Locate your virtual environment by running
```bash
poetry env info --path
```

1. OPTIONAL: It is recommended to enable [FlashAttention](https://github.com/Dao-AILab/flash-attention) backend for better performance, run the following commands inside your environment to install:
```bash
pip install wheel

# Change the path according to your environment, this is an example for the Vector cluster
export CUDA_HOME=/pkgs/cuda-12.3

pip install flash-attn --no-build-isolation
```

## Launch an inference server
We will use the Llama 3 model as example, to launch an inference server for Llama 3 8B, run
```bash
bash models/llama3/launch_server.sh
```
You should see an output like the following:
> Job Name: vLLM/Meta-Llama-3-8B
> 
> Partition: a40
> 
> Generic Resource Scheduling: gpu:1
> 
> Data Type: auto
> 
> Submitted batch job 12217446

If you want to use your own virtual environment, you can run this instead:
```bash
bash models/llama3/launch_server.sh -e $(poetry env info --path)
```
By default, the `launch_server.sh` script in Llama 3 folder uses the 8B variant, you can switch to other variants with the `-v` flag, and make sure to change the requested resource accordingly. More information about the flags and customizations can be found in the [`models`](models) folder. The inference server is compatible with the OpenAI `Completion` and `ChatCompletion` API. You can inspect the Slurm output files to check the inference server status. 

Here is a more complicated example that launches a model variant using multiple nodes, say we want to launch Mixtral 8x22B, run
```bash
bash models/mixtral/launch_server.sh -v 8x22B-v0.1 -N 2 -n 4
```
The default partition for Mixtral models is a40, and we need 8 a40 GPUs to load Mixtral 8x22B, so we requested 2 a40 nodes with 4 GPUs per node. You should see an output like the following:
> Number of nodes set to: 2
>
> Number of GPUs set to: 4
>
> Model variant set to: 8x22B-v0.1
> 
> Job Name: vLLM/Mixtral-8x22B-v0.1
> 
> Partition: a40
> 
> Generic Resource Scheduling: gpu:8
> 
> Data Type: auto
> 
> Submitted batch job 12430232

## Send inference requests
Once the inference server is ready, you can start sending in inference requests. We provide example [Python](examples/inference.py) and [Bash](examples/inference.sh) scripts for sending inference requests in [`examples`](examples) folder. Make sure to update the model server URL and the model weights location in the scripts. You can run either `python examples/inference.py` or `bash examples/inference.sh`, and you should expect to see an output like the following:
> {"id":"cmpl-bdf43763adf242588af07af88b070b62","object":"text_completion","created":2983960,"model":"/model-weights/Llama-2-7b-hf","choices":[{"index":0,"text":"\nCanada is close to the actual continent of North America. Aside from the Arctic islands","logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":8,"total_tokens":28,"completion_tokens":20}}

## SSH tunnel from your local device
If you want to run inference from your local device, you can open a SSH tunnel to your cluster environment like the following:
```bash
ssh -L 8081:172.17.8.29:8081 username@v.vectorinstitute.ai -N
```
The example provided above is for the vector cluster, change the variables accordingly for your environment
