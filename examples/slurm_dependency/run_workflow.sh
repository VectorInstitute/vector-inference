#!/bin/bash

# ---- Config ----
MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
LAUNCH_ARGS="$MODEL_NAME"

# ---- Step 1: Launch the server
RAW_JSON=$(vec-inf launch $LAUNCH_ARGS --json-mode)
SERVER_JOB_ID=$(echo "$RAW_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['slurm_job_id'])")
echo "Launched server as job $SERVER_JOB_ID"
echo "$RAW_JSON"

# ---- Step 2: Submit downstream job
sbatch --dependency=after:$SERVER_JOB_ID --export=SERVER_JOB_ID=$SERVER_JOB_ID downstream_job.sbatch
