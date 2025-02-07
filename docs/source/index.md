---
hide-toc: true
---

# Vector Inference: Easy inference on Slurm clusters

```{toctree}
:hidden:

user_guide

```

This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using [vLLM](https://docs.vllm.ai/en/latest/). **All scripts in this repository runs natively on the Vector Institute cluster environment**. To adapt to other environments, update [`launch_server.sh`](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/launch_server.sh), [`vllm.slurm`](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/vllm.slurm), [`multinode_vllm.slurm`](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/multinode_vllm.slurm) and [`models.csv`](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/models/models.csv) accordingly.

## Installation

If you are using the Vector cluster environment, and you don't need any customization to the inference server environment, run the following to install package:

```bash
pip install vec-inf
```

Otherwise, we recommend using the provided [`Dockerfile`](https://github.com/VectorInstitute/vector-inference/blob/main/Dockerfile) to set up your own environment with the package.
