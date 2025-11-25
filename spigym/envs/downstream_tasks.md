Downstream Tasks: Training Recipes
=================================

This document describes how to train downstream locomotion tasks after (or independent of) Active SysID. It provides runnable commands, explains the common Hydra flags, and points to relevant configs.


Common Setup
------------
- Ensure your environment is installed per the root README.
- Activate your venv and prefer `uv run --active` to keep execution inside it.
- Start with a modest `+num_envs` (e.g., 512 or 1024) if you are unsure about GPU memory.

Hydra basics
- `+simulator=isaacgym` selects the Isaac Gym simulator preset.
- `+exp=...` picks the task experiment config under `spigym/config/exp/`.
- `+rewards=...` sets the reward shaping.
- `+robot=...`, `+terrain=...`, `+obs=...` choose robot, terrain, and observation configs.
- `project_name`, `experiment_name` are logging identifiers (W&B if `+opt=wandb`).


Velocity Tracking (Go2)
-----------------------
Example command:

```
uv run --active python spigym/train_agent.py \
  +simulator=isaacgym \
  +exp=go2_locomotion \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_go2_agile_locomotion \
  +robot=go2/go2 \
  +terrain=terrain_locomotion_plane \
  +obs=loco/go2_locomotion \
  num_envs=2048 \
  project_name=spi-active-exps \
  experiment_name=go2_locomotion_agile \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_initial_penalty_scale=0.1 \
  rewards.reward_penalty_degree=0.0003 \
  rewards.reward_scales.penalty_orientation=-0.75 \
  +opt=wandb
```


Forward Block Jump (Go2)
------------------------
Example command:

```
uv run --active python spigym/train_agent.py \
  +simulator=isaacgym \
  +exp=go2_block_jump \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_go2_block_jump \
  +robot=go2/go2 \
  +terrain=terrain_locomotion_plane \
  +obs=loco/go2_block_jump \
  num_envs=2048 \
  project_name=spi-active-exps \
  experiment_name=go2_block_jump \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_initial_penalty_scale=1.0 \
  rewards.reward_scales.penalty_orientation=-3.0 \
  rewards.reward_scales.penalty_action_rate=-0.1 \
  rewards.reward_scales.penalty_slippage=-1.5 \
  +opt=wandb
```


Yaw Jump (Go2)
--------------
Example command:

```
uv run --active python spigym/train_agent.py \
  +simulator=isaacgym \
  +exp=go2_block_jump \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_go2_yaw_jump \
  +robot=go2/go2 \
  +terrain=terrain_locomotion_plane \
  +obs=loco/go2_block_jump \
  num_envs=2048 \
  project_name=Go2_sim2real_yaw_jump \
  experiment_name=go2_yaw_jump \
  rewards.reward_penalty_curriculum=False \
  rewards.reward_scales.penalty_action_rate=-0.5 \
  rewards.reward_scales.penalty_slippage=-1.5 \
  rewards.reward_scales.yaw_orientation_control=3.0 \
  rewards.reward_scales.lin_vel_z=5.0 \
  rewards.reward_scales.feet_x=-1.0 \
  +opt=wandb
```


Roll-Pitch Tracking (Go2)
-------------------------
Example command:

```
uv run --active python spigym/train_agent.py \
  +simulator=isaacgym \
  +exp=go2_rp_track \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_go2_rp_track \
  +robot=go2/go2 \
  +terrain=terrain_locomotion_plane \
  +obs=loco/go2_rp_track \
  num_envs=2048 \
  project_name=spi-active-exps \
  experiment_name=go2_rp_track \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_initial_penalty_scale=0.5 \
  +opt=wandb
```


Tips and Customization
----------------------
- Start small: reduce `num_envs` if you see GPU OOM.
- Domain randomization: swap `+domain_rand` configs to stress-test robustness.
- Rewards: all reward components live under `+rewards=...`; you can override individual scales via `rewards.reward_scales.*`.
- Logging: add `+opt=wandb` and ensure `wandb login` is configured.
- Checkpoints: by default, training scripts save under `logs/` with timestamped run names.


Troubleshooting
---------------
- Viewer issues on headless servers: set `env.config.headless=True` in your Hydra override, or use configs that disable viewers by default.
- Isaac Gym import errors: double-check it is installed in the same venv (`uv pip install -e /path/to/isaac-gym/python`).
- Performance: ensure you run with a recent NVIDIA driver and that `torch.cuda.is_available()` is true.

