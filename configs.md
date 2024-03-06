# Configurations

## VENV_BASE

Root folder of the virtual environment to activate. This folder should contain subfolders including `bin` and `lib`.

For example, to reuse the pre-configured vLLM venv on the vector cluster, set:

```bash
export VENV_BASE=/ssd005/projects/llm/vllm-ray-venv/
```

## VLLM_BASE_URL_FILENAME

After the job starts, the API Base URL to be set in `OPENAI_API_BASE` will be written to the file specified in `VLLM_BASE_URL_FILENAME`.

## VLLM_MODEL_NAME

Name of model to load.

- Local model: point to a folder including both model weights/config and tokenizer.
- Model from the HuggingFace

# JOB_GRES_TYPE

Some clusters (Mila, Narval, but not Vector Vaughan) specify the type of GPU to use via a separate `gres` parameter. If `JOB_GRES_TYPE` is not blank, the specified value will be inserted into the string used for the `gres` flag.