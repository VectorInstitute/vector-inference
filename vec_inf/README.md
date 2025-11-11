## `vec-inf` CLI Commands

* `launch`: Specify a model family and other optional parameters to launch an OpenAI compatible inference server.
* `batch-launch`: Specify a list of models to launch multiple OpenAI compatible inference servers at the same time.
* `status`: Check the status of all `vec-inf` jobs, or a specific job by providing its job ID.
* `metrics`: Streams performance metrics to the console.
* `shutdown`: Shutdown a model by providing its Slurm job ID.
* `list`: List all available model names, or view the default/cached configuration of a specific model.
* `cleanup`: Remove old log directories. You can filter by `--model-family`, `--model-name`, `--job-id`, and/or `--before-job-id`. Use `--dry-run` to preview what would be deleted.

Use `--help` to see all available options

## `VecInfClient` API

* `launch_model`: Launch an OpenAI compatible inference server.
* `batch_launch_models`: Launch multiple OpenAI compatible inference servers.
* `fetch_running_jobs`: Get the running `vec-inf` job IDs.
* `get_status`: Get the status of a running model.
* `get_metrics`: Get the performance metrics of a running model.
* `shutdown_model`: Shutdown a running model.
* `list_models`" List all available models.
* `get_model_config`: Get the configuration for a specific model.
* `wait_until_ready`: Wait until a model is ready or fails.
* `cleanup_logs`: Remove logs from the log directory.
