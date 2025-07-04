# `vec-inf` Commands

* `launch`: Specify a model family and other optional parameters to launch an OpenAI compatible inference server, `--json-mode` supported.
* `status`: Check the model status by providing its Slurm job ID, `--json-mode` supported.
* `metrics`: Streams performance metrics to the console.
* `shutdown`: Shutdown a model by providing its Slurm job ID.
* `list`: List all available model names, or view the default/cached configuration of a specific model, `--json-mode` supported.
* `cleanup`: Remove old log directories. You can filter by `--model-family`, `--model-name`, `--job-id`, and/or `--before-job-id`. Use `--dry-run` to preview what would be deleted.

Use `--help` to see all available options
