# VLLM for the Vector Cluster

## Installation

On the Vector Cluster, we've provided a read-only reference virtual environment with vLLM pre-installed at `/ssd005/projects/llm/vllm-ray-env`. Unless you plan to install custom packages in the same virtual environment or set up the environment on a different cluster, you may skip ahead to the "Launching" section.

If you want to setup the vLLM virtual environment on your own, you might need up to 10GB of storage. The following steps needs to be run only once for each user.

### Virtual Env

```bash
module load python/3.10
python3 -m venv env
echo "module unload python" >> env/bin/activate
source env/bin/activate

```

After creating a virtual environment, be sure to un-load the python module to install to the virtual environment and not user default path.

Verify that the virtual environment is actually selected using the following one-line command:

```bash
python3 -c 'import sys; print(sys.prefix)'  # should print virtual-env path
python3 -c 'import sys; print(sys.base_prefix)' # /pkgs/python-3.10.12
pip3 install -U --require-virtualenv pip
```

### Nvidia Packages and Libraries

VLLM tensor parallelism invokes NCCL (the Nvidia CUDA communication library) via cupy.
More info here: [link](https://github.com/vllm-project/vllm/blob/22de45235/vllm/model_executor/parallel_utils/cupy_utils.py#L78-L80)

Overall steps:

- Module-Load CUDA shared objects (from the cluster).
- Pip-Install cupy wheel, pinned to a specific version where setattr is allowed.
- Install NCCL libraries with cupy

Module-Load CUDA shared objects (libcudart.so.11.0, etc.) from the cluster. Add these to the virtualenv activate script so the shared objects are added to PATH automatically next time.

```bash
deactivate
# TODO: Update this line when the cluster adds a cuda-12 shared object package.
echo "module load cuda-11.8" >> env/bin/activate
source env/bin/activate
```

Install cupy wheel. Because vLLM requires bfloat16 while cupy does not support bfloat16, vLLM leveraged some workarounds that require `setattr` access. This workaround works only for certain cupy versions.
[cupy bfloat16](https://github.com/cupy/cupy/blob/2fd0b819b/cupy/_core/dlpack.pyx#L319)
[vllm cupy bfloat16 workarounds](https://github.com/vllm-project/vllm/blob/22de45235/vllm/model_executor/parallel_utils/cupy_utils.py#L104-L110)

For entertainment purposes, here's the particular commit which seems to break the vLLM workaround. ([link](https://github.com/cupy/cupy/commit/80dade7b33ded1f50fb5297ac466d00dfcf3f2c5), [blame](https://github.com/cupy/cupy/blame/main/cupy/_core/core.pyx#L126), [issue](https://github.com/cupy/cupy/issues/7883) requesting that this workaround should be broken).

To install the wheel for a cupy version where this workaround is not intentionally broken, run the following:

```bash
pip install -U --require-virtualenv "cupy-cuda11x==12.1.0"
```

Install NVIDIA packages via cupy:

```bash
python -m cupyx.tools.install_library --cuda "11.x" --library nccl
python -m cupyx.tools.install_library --cuda "11.x" --library cutensor
python -m cupyx.tools.install_library --cuda "11.x" --library cudnn
```

### Packages

Install vLLM and dependency packages:

- vLLM, via pre-built wheels
- ray, for multi-gpu tensor parallelism.

```bash
pip3 install -U --require-virtualenv ray vllm
```

## Launching and Testing

The `openai_entrypoint.sh` script provides a convenient way to launch a multi-GPU vLLM server as a SLURM job. The inference server is compatible with the OpenAI `Completion` and `ChatCompletion` API.

Note that there is no direct way to obtain the address of the worker node until SLURM assigns the job. To communicate API base URL back to you, the script provides to option to write that information to a file specified in `VLLM_BASE_URL_FILENAME` in the config. Refer to [configs.md](configs.md#VLLM_BASE_URL_FILENAME) for more info.

Edit configurations in slurm.env as needed. We provide example configurations for the following clusters:

- Vector (Vaughan)
- Mila
- Compute Canada (Narval)

Instead of invoking the SLURM script directly with sbatch, you might want to invoke openai_entrypoint.sh to apply your custom configurations.

```bash
source slurm.env
bash openai_entrypoint.sh
```

After the job has started, invoke test.sh to the API status.

```bash
source slurm.env  # load VLLM_BASE_URL_FILENAME
bash test.sh
```

Example output:

> API_BASE_URL: http://gpu105:19568/v1
>
> {"id":"cmpl-e01cb03c3f05436a9f2e6a000420b461","object":"chat.completion","created":1072630,"model":"google/gemma-2b-it","choices":[{"index":0,"message":{"role":"assistant","content":"The sky appears blue due to Rayleigh scattering. Rayleigh scattering is the scattering of light by molecules in the Earth's atmosphere. Blue light has a shorter wavelength than other colors of light, so it is scattered more strongly. This is why the sky appears blue to us."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":17,"total_tokens":72,"completion_tokens":55}}
