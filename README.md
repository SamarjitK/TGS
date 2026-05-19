# TGS
## 1. Introduction

This repository contains a fork of the source code for the NSDI'23 paper "Transparent GPU Sharing in Container Clouds for Deep Learning Workloads" [[Paper]](https://www.usenix.org/conference/nsdi23/presentation/wu).

## 2. Updates

An overview of updates/changes to this repository:
- Created a new `README.md` (hello!) to provide concise instructions for minimal setup.
- Created a new `requirements.txt` file to be compatible with newer setups (uses python 3.10)
- Bugfixes:
    - `build.sh` - related to known (issue)[https://github.com/pkusys/TGS/issues/5]. Instead of mounting shared object (`.so`) files directly, they are now built within the container to avoid `glibc` version mismatch.
    - `hijack.h` - replaced outdated `__sync_bool_compare_and_swap_8` with `__sync_bool_compare_and_swap` for better compatibility with newer compilers.
    - `task.py` - removed extra quotes when specifying GPU devices.
- Enhancements (may be required based on your setup):
    - `loader.c` - added the CUDA/NVML fallback loading so the hijack layer could find the real driver libs and not abort on startup.
    - `worker.py` - `check_tasks()` exits gracefully in both success and failure cases.
    - `trainer_client.py` - made report RPCs time-bounded and failure-tolerant so training did not hang behind scheduler connectivity problems.
    - `build.sh` - added a check to build the custom docker image if it doesn't exist locally, to avoid runtime errors.
- Additions:
    - `trainer.py` - you can now customize the `REPORT_INTERVAL` environment variable to specify how often you want the trainer to report stats to the scheduler. `TGS_REPORT_INTERVAL_SEC` is also supported for SLO-focused runs. Increased it from the default 10 seconds to 2 seconds.
    - `worker.py` - can write plain-text SLO/report metrics to `TGS_SLO_METRICS_PATH`, separate from Python logging.
    - `plot_tgs_throughput.py` - a simple script to plot the throughput results from the test run.
    - `scripts/`, `config/`, `workloads/` - added some new test scripts and associated workloads (see Run section below) to test our changes to TGS.

## 3. Python setup

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed! You'll want to create a venv with Python 3.10, and then install the dependencies:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt --no-build-isolation
```

You'll need to turn off build isolation because grpcio-tools depends on "pkg_resources" which is provided by setuptools, but setuptools is not available in the environment when build isolation is enabled.

## 4. Build

Run the following commands:

```bash
cd TGS
docker build -f docker/fixed.Dockerfile -t tf_torch:fixed .  # Build the custom docker image
make rpc
./download.sh
cd hijack
./build.sh
```

The build script used to rely on an image built on top of `bingyangwu2000/tf_torch`, which has some extra dependencies pre-installed. There is now some custome images (to avoid downloading at runtime) in the `docker/` directory, and the `build.sh` script now uses `tf_torch:fixed` as the base iamge, which you should build as shown above.

There is also a `make clean` command to clean up the rpc artifacts if you want to start fresh.

## 5. Docker setup

There are docker images for specific workloads in the `/docker` directory that are required for some of the test scripts (check the associated `config/` files for details). The table below lists the image names, descriptions, and running instructions:

| Image Name | Run Instructions | Description |
|------------|------------------|-------------|
| `tf_torch:latest` | `docker pull bingyangwu2000/tf_torch` | This is what the original TGS project uses. No longer necessary |
| `tf_torch:base` | `docker build -f docker/base.Dockerfile -t tf_torch:base .` | Builds on this original image, implements speedup and also supports text inference. |
| `tf_torch:fixed` | `docker build -f docker/fixed.Dockerfile -t tf_torch:fixed .` | The new base image, upgrades versioning to support modern inference engines |
| `tf_torch:vllm` | `docker build -f docker/vllm.Dockerfile -t tf_torch:vllm .` | Builds on the fixed image, installs vllm. Not yet used for anything. |
| `tf_torch:llamacpp` | `docker build -f docker/llamacpp.Dockerfile -t tf_torch:llamacpp .` | Builds on the fixed image, installs LLaMA.cpp. Used for text inference tests. |
| `tf_torch:llamacpp-model` | `bash docker/build_llamacpp_model.sh` | See script for more details. Builds on the `llamacpp` image and installs a small LLaMA model (which can be specified). Place huggingface token in `docker/hf_token.txt` if private. |

## 6. Run

This script will run the TGS system with the test configuration provided in `config/test_tgs.csv`. You can modify this file to test different configurations (start times, iterations, etc.)

```
./scripts/test_tgs.sh
```

If you run into docker issues between runs, make sure to remove any existing containers that will have the same name as the ones being created by the script. You can do this with `docker ps -a` to list all containers and `docker rm <container_name>` to remove any that are causing issues.

We've started building out support for inference workloads, with scripts and description listed in the table below:

| Script | Description |
|--------|-------------|
| `test_inference_tgs.sh` | Image inference |
| `test_text_inference_tgs.sh` | Text generation |
| `test_text_inference_llamacpp.sh` | HF LLM inference with LLaMA.cpp |
| `test_text_inference_vllm.sh` | Work in progress |

You can also plot the results by running `uv run scripts/plot_tgs_throughput.py`, which will dump resulting plots in the `results/` directory. You can modify this script to plot different metrics or configurations as needed. If you've run many types of tests, you will need to specify which files in `job_logs/` to read from by using a `--job` argument based on the model: for example `--job resnet50` or `--job distilgpt2`.

For SLO experiments, set `TGS_SLO_MODE=1` to enable shared-job leader/lagger comparison. Set `TGS_SLO_METRICS_PATH=/path/to/file.txt` to append comma-separated `report` and `slo` rows to a plain text file, which is easier to parse than the Python log stream. Lower `TGS_REPORT_INTERVAL_SEC` or `REPORT_INTERVAL` if you need trainers to send reports more often.

Trace CSVs can also pass startup SLO hints into the hijack layer with optional `slo_group`, `slo_role`, `slo_target_tpot_ms`, and `slo_initial_rate_limit` columns. See `config/test_tgs_slo_shared.csv` for two identical `shared` jobs that use the same hijack mounts while passing `slow` and `fast` roles down as `TGS_SLO_*` environment variables.
