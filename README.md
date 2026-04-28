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
- Additions:
    - `trainer.py` - you can now customize the `REPORT_INTERVAL` environment variable to specify how often you want the trainer to report stats to the scheduler. Increased it from the default 10 seconds to 2 seconds.
    - `plot_tgs_throughput.py` - a simple script to plot the throughput results from the test run.

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
make rpc
./download.sh
cd hijack
./build.sh
```

The build script should also automatically pull in the docker image we need. If it doesn't, you can pull it manually with `docker pull bingyangwu2000/tf_torch`. There is also a `make clean` command to clean up the rpc artifacts if you want to start fresh.

## 4. Run

This script will run the TGS system with the test configuration provided in `config/test_tgs.csv`. You can modify this file to test different configurations (start times, iterations, etc.)

```
./scripts/test_tgs.sh
```

If you run into docker issues between runs, make sure to remove any existing containers that will have the same name as the ones being created by the script. You can do this with `docker ps -a` to list all containers and `docker rm <container_name>` to remove any that are causing issues.

You can also plot the results by running `uv run scripts/plot_tgs_throughput.py`, which will dump resulting plots in the `results/` directory. You can modify this script to plot different metrics or configurations as needed.
