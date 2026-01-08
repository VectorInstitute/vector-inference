# Configs

* [`environment.yaml`](environment.yaml): Configuration for the Slurm cluster environment, including image paths, resource availabilities, default value, and etc.
* [`models.yaml`](models.yaml): Configuration for launching model inference servers, including Slurm parameters as well as inference engine arguments.

**NOTE**: These configs acts as last resort fallbacks in the `vec-inf` package, they will be updated to match the latest cached config on the Vector Killarney cluster with each new package version release.
