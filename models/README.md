# Environment Variables
The following environment variables all have default values that's suitable for the Vector cluster environment. You can use flags to modify certain environment variable values.

* **MODEL_FAMILY**: Directory name of the model family.
* **SRC_DIR**: Relative path for the `[src](../src/)` folder.
* **CONFIG_FILE**: Config file containing default values for some environment variables in the **MODEL_FAMILY** diretory.
* **MODEL_NAME**: Name of model family according to the actual model weights.
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
* **QOS**: Quality of Service.
* **TIME**: Max Walltime.

The following environment variables are only for Vision Language Models

* **CHAT_TEMPLATE**: The relative path to the chat template if no default chat template is available.
* **IMAGE_INPUT_TYPE**: Possible choices: `pixel_values`, `image_features`. The image input type passed into vLLM, default to `pixel_values`.
* **IMAGE_TOKEN_ID**: Input ID for image token. Default to HF Config value. Default value set according to model.
* **IMAGE_INPUT_SHAPE**: The biggest image input shape (worst for memory footprint) given an input type. Only used for vLLM’s profile_run. Default value set according to model.
* **IMAGE_FEATURE_SIZE**: The image feature size along the context dimension. Default value set according to model.

# Named Arguments
NOTE: Arguments like `--num-nodes` or `model-variant` might not be available to certain model families because they should fit inside a single node or there is no variant availble in `/model-weights` yet. You can manually add these options in launch scripts if you need, or make a request to download weights for other variants.
* `--model-family`: Sets **MODEL_FAMILY**, the available options are the names of each sub-directory in this directory. **This argument MUST be set.**  
* `--model-variant`: Overrides **MODEL_VARIANT**
* `--partition`: Overrides **JOB_PARTITION**.
* `--num-nodes`: Overrides **NUM_NODES**.
* `--num-gpus`: Overrides **NUM_GPUS**.
* `--qos`: Overrides **QOS**.
* `--time`: Overrides **TIME**.
* `--data-type`: Overrides **VLLM_DATA_TYPE**.
* `--venv`: Overrides **VENV_BASE**.
* `--is-vlm`: Specifies this is a Vision Language Model, no value needed.

The following flags are only available to Vision Language Models

* `--image-input-type`: Overrides **IMAGE_INPUT_TYPE**
* `--image-token-id`: Overrides **IMAGE_TOKEN_ID**
* `--image-input-shape`: Overrides **IMAGE_INPUT_SHAPE**
* `--image-feature-size`: Overrides **IMAGE_FEATURE_SIZE**
