# SPI-Active + Isaac Gym — uv Setup and Training Guide

This guide shows how to set up a Python 3.8 environment using uv, install NVIDIA Isaac Gym, install this repository, and run a training job. It mirrors the workflow used to reproduce GO2 locomotion experiments.

If you run into issues, see Common Issues at the end.

## Prerequisites

- NVIDIA GPU with recent drivers (`nvidia-smi` works).
- CUDA/cuDNN versions compatible with your Isaac Gym build.
- Linux (Ubuntu 20.04/22.04 tested) with Python 3.8.
- NVIDIA Developer account to download Isaac Gym.
- uv installed: https://docs.astral.sh/uv/getting-started/
  - Quick install (Linux/macOS): `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 1) Create a uv virtual environment (Python 3.8)

Create and activate a venv. Example uses a custom name to match the workflow:

```bash
uv venv -p 3.8 .venv-sysid_isaacgym
source .venv-sysid_isaacgym/bin/activate
```

Alternatively, to use a project-local `.venv`:

```bash
uv venv -p 3.8
source .venv/bin/activate
```

## 2) Install Isaac Gym (Python API)

Download and extract Isaac Gym (e.g., “Preview 4”) from NVIDIA. Then install the Python API in editable mode inside the activated venv.

```bash
# Adjust the path below to where you extracted Isaac Gym
uv pip install -e /path/to/isaac-gym/python
```

Quick checks:

```bash
python -c "import isaacgym; print('Isaac Gym OK')"
```

If you hit libpython errors at runtime, ensure your venv’s `lib/` is on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$(python -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
```

## 3) Install this repository with uv

From the repo root:

```bash
# Installs the workspace in editable mode and resolves dependencies
uv sync --dev
```

Notes
- This repo is a uv workspace; subpackages (e.g., `isaac_utils/`) are installed automatically in editable mode by `uv sync`.
- You can also use `uv pip install -e .`, but `uv sync` is preferred because it respects the lockfile and workspace config.

## 4) Run training with uv

Use `uv run --active` to run with the currently activated venv. Example command (GO2 Walk-These-Ways):

```bash
uv run --active python spigym/train_agent.py \
  +simulator=isaacgym \
  +exp=go2_wtw \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_go2_wtw \
  +robot=go2/go2 \
  +terrain=terrain_locomotion_plane \
  +obs=loco/go2_wtw \
  num_envs=4096 \
  project_name=spi-active-exps \
  experiment_name=go2_wtw_test_shorter_obs \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_initial_penalty_scale=1.0 \
  +opt=wandb
```

About the flags
- `+key=value` syntax extends/overrides Hydra configs in this repo.
- `num_envs=4096` is GPU-memory heavy; reduce if you see OOM (e.g., `num_envs=1024`).
- `+opt=wandb` enables Weights & Biases logging; see next section.

## 5) (Optional) W&B setup

If you enable W&B (`+opt=wandb`):

```bash
wandb login
# or
export WANDB_API_KEY=...  # your key
```

## Verifying your setup

- GPU visible: `python -c "import torch; print(torch.cuda.is_available())"`
- Isaac Gym import: `python -c "import isaacgym; print('ok')"`
- Dry-run smaller job first: set `num_envs=128` and run for a few minutes.

## Common Issues

- ImportError: cannot import name isaacgym
  - Ensure the venv is active and `uv pip install -e /path/to/isaac-gym/python` was run in the same venv.
- libpython or GL errors when launching examples/viewer
  - Ensure your venv/lib is on `LD_LIBRARY_PATH` (see above) and that NVIDIA drivers are recent. For headless servers, disable the viewer in configs or run without viewer.
- Python version mismatch
  - Isaac Gym typically targets Python 3.8. Use 3.8 for the venv as shown above.
- CUDA / driver mismatch
  - Match your NVIDIA driver and CUDA version to the Isaac Gym binary you installed. Verify with `nvidia-smi`.

## Handy uv commands

- Create venv: `uv venv -p 3.8`
- Activate: `source .venv/bin/activate`
- Sync deps: `uv sync --dev`
- Run (active venv): `uv run --active python -m spi.train_agent`

## Repro checklist

- [ ] `uv --version` prints a recent version
- [ ] `nvidia-smi` shows your GPU
- [ ] Isaac Gym installed into the same venv
- [ ] `uv sync --dev` completed without errors
- [ ] Training command runs (or with smaller `num_envs`)

---

If anything is unclear or you need a Dockerfile-based setup, please open an issue and we can add a containerized workflow.

