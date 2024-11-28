# `vec-inf` Commands

* `launch`: Specify a model family and other optional parameters to launch an OpenAI compatible inference server, `--json-mode` supported. Check [`here`](./models/README.md) for complete list of available options.
* `list`: List all available model names, or append a supported model name to view the default configuration, `--json-mode` supported.
* `metrics`: Streams performance metrics to the console.
* `status`: Check the model status by providing its Slurm job ID, `--json-mode` supported.
* `shutdown`: Shutdown a model by providing its Slurm job ID.

Use `--help` to see all available options
