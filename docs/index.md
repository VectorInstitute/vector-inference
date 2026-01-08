# Vector Inference: Easy inference on Slurm clusters

This repository provides an easy-to-use solution to run inference servers on [Slurm](https://slurm.schedmd.com/overview.html)-managed computing clusters using open-source inference engines ([vLLM](https://docs.vllm.ai/en/v0.12.0/), [SGLang](https://docs.sglang.io/index.html)). **This package runs natively on the Vector Institute cluster environments**. To adapt to other environments, follow the instructions in [Installation](#installation).


**NOTE**: Supported models on Killarney are tracked [here](https://github.com/VectorInstitute/vector-inference/blob/main/MODEL_TRACKING.md)

## Installation

If you are using the Vector cluster environment, and you don't need any customization to the inference server environment, run the following to install package:

```bash
pip install vec-inf
```

Otherwise, we recommend using the provided [`vllm.Dockerfile`](https://github.com/VectorInstitute/vector-inference/blob/main/vllm.Dockerfile) and [`sglang.Dockerfile`](https://github.com/VectorInstitute/vector-inference/blob/main/sglang.Dockerfile) to set up your own environment with the package. The built images are available through [Docker Hub](https://hub.docker.com/orgs/vectorinstitute/repositories)

If you'd like to use `vec-inf` on your own Slurm cluster, you would need to update the configuration files, there are 3 ways to do it:

* Clone the repository and update the `environment.yaml` and the `models.yaml` file in [`vec_inf/config`](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/config), then install from source by running `pip install .`.
* The package would try to look for cached configuration files in your environment before using the default configuration. The default cached configuration directory path points to `/model-weights/vec-inf-shared`, you would need to create an `environment.yaml` and a `models.yaml` following the format of these files in [`vec_inf/config`](https://github.com/VectorInstitute/vector-inference/blob/main/vec_inf/config).
* [OPTIONAL] The package would also look for an enviroment variable `VEC_INF_CONFIG_DIR`. You can put your `environment.yaml` and `models.yaml` in a directory of your choice and set the enviroment variable `VEC_INF_CONFIG_DIR` to point to that location.
