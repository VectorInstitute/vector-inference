# Vector Inference: Easy inference on Slurm clusters

This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using [vLLM](https://docs.vllm.ai/en/latest/). **All scripts in this repository runs natively on the Vector Institute cluster environment**. To adapt to other environments, follow the instructions in [Installation](#installation).

## Installation

If you are using the Vector cluster environment, and you don't need any customization to the inference server environment, run the following to install package:

```bash
pip install vec-inf
```

Otherwise, we recommend using the provided [`Dockerfile`](https://github.com/VectorInstitute/vector-inference/blob/main/Dockerfile) to set up your own environment with the package. The latest image has `vLLM` version `0.10.1.1`.

If you'd like to use `vec-inf` on your own Slurm cluster, you would need to update the configuration files, there are 3 ways to do it:
* Clone the repository and update the `environment.yaml` and the `models.yaml` file in [`vec_inf/config`](vec_inf/config/), then install from source by running `pip install .`.
* The package would try to look for cached configuration files in your environment before using the default configuration. The default cached configuration directory path points to `/model-weights/vec-inf-shared`, you would need to create an `environment.yaml` and a `models.yaml` following the format of these files in [`vec_inf/config`](vec_inf/config/).
* The package would also look for an enviroment variable `VEC_INF_CONFIG_DIR`. You can put your `environment.yaml` and `models.yaml` in a directory of your choice and set the enviroment variable `VEC_INF_CONFIG_DIR` to point to that location.
