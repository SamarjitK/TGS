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
docker build -t tf_torch_fixed .  # Build the custom docker image
make rpc
./download.sh
cd hijack
./build.sh
```

The build script now relies on an image built on top of `bingyangwu2000/tf_torch`, which has some extra dependencies pre-installed. In the future, we will create a custom image per type of workload (to avoid downloading at runtime), but for now these live in a unified image that we refer to as `tf_torch_fixed`. The modified `build.sh` *should* check if the image exists locally and build it if not.

There is also a `make clean` command to clean up the rpc artifacts if you want to start fresh.

## 5. Run

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

You can also plot the results by running `uv run scripts/plot_tgs_throughput.py`, which will dump resulting plots in the `results/` directory. You can modify this script to plot different metrics or configurations as needed. If you've run many types of tests, you will need to specify which files in `job_logs/` to read from by using a `--job` argument based on the model: for example `--job resnet50` or `--job distilgpt2`.

For SLO experiments, set `TGS_SLO_MODE=1` to enable shared-job leader/lagger comparison. Set `TGS_SLO_METRICS_PATH=/path/to/file.txt` to append comma-separated `report` and `slo` rows to a plain text file, which is easier to parse than the Python log stream. Lower `TGS_REPORT_INTERVAL_SEC` or `REPORT_INTERVAL` if you need trainers to send reports more often.

Trace CSVs can also pass startup SLO hints into the hijack layer with optional `slo_group`, `slo_role`, `slo_target_tpot_ms`, and `slo_initial_rate_limit` columns. See `config/test_tgs_slo_shared.csv` for two identical `shared` jobs that use the same hijack mounts while passing `slow` and `fast` roles down as `TGS_SLO_*` environment variables.
