# Environment Variables
The following environment variables all have default values that's suitable for the Vector cluster environment. You can use flags to modify certain environment variable values.

* **MODEL_NAME**: Name of model family.
* **MODEL_VARIANT**: Variant of the model, the variants available are listed in respective model folders. Default variant is bolded in the corresponding README.md file.
* **MODEL_DIR**: Path to model's directory in vector-inference repo.
* **VLLM_BASE_URL_FILENAME**: The file to store the inference server URL, this file would be generated after launching an inference server, and it would be located in the corresponding model folder with the name `.vllm_{model-name}-{model-variant}_url`.
* **VENV_BASE**: Location of the virtual environment.
* **VLLM_MODEL_WEIGHTS**: Location of the model weights.
* **VLLM_DATA_TYPE**: Model data type.
* **LD_LIBRARY_PATH**: Include custom locations for dynamically linked library files in a Unix-like operating system. In the script, we tell the dynamic linker to also look at the CUDA and cuDNN directories.
* **JOB_NAME**: Slurm job name.
* **NUM_NODES**: Numeber of nodes scheduled. Default to suggested resource allocation.
* **NUM_GPUS**: Number of GPUs scheduled. Default to suggested resource allocation.
* **JOB_PARTITION**: Type of compute partition. Default to suggested resource allocation.
* **QOS**: Quality of Service

# Flags
* `-p`: Overrides **JOB_PARTITION**.
* `-N`: Overrides **NUM_NODES**.
* `-n`: Overrides **NUM_GPUS**.
* `-q`: Overrides **QOS**.
* `-d`: Overrides **VLLM_DATA_TYPE**.
* `-e`: Overrides **VENV_BASE**.
* `-v`: Overrides **MODEL_VARIANT**
