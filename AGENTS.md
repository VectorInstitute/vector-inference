# AGENTS.md

Background- `vec-inf` provides a set of command-line utilities for launching vLLM servers on SLURM clusters.

## Testing

- Run `uv run vec_inf/client/oai_compatibility.py` and make sure the output shows the response from the LLM assistant. This script might take a while to complete without printing any output, and you should set a timeout of 5 minutes.
- Cleanup:
  - Run `date` to get the time when the tests started.
  - Keep track of the SLURM job id of jobs you launched.
  - Run `squeue --me` to identify currently-running jobs.
  - Run `scancel [job_id]` to stop all jobs that you created. If you think some jobs were launched incorrectly (e.g., you didn't receive a job_id in response,) try to stop these jobs as well by comparing START_TIME.

## Code Style

- Lint wiith ruff `uv run ruff check [filenames]`
- Lint with basedpyright `uv run basedpyright [filenames]`
  - Pyright Warnings are okay but not nice- you don't have to fix existing ones, but try not to introduce additional ones.

## Special notes to agents

- To run a command, you should run `source .envrc` and then use `uv run` to activate the right venv.
- You are expected to run all testing commands and to follow the clean-up instructions accordingly.
- Do not stop the job whose name starts with "vscode"- the codex agent process is running within that job.
  - However, you may run `scancel --me --qos scavenger` since only the jobs that you invoked will be under that specific SLURM qos.
- Recommended: use `timeout --signal=SIGINT [timeout] ...`.