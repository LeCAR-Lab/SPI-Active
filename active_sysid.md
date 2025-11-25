# Active SysID Guide

Active SysID is a framework for identifying physical parameters of legged robots through active exploration. The process typically involves four main steps:

1. **Training a multi-behavioral omni locomotion controller** - A versatile controller capable of executing diverse locomotion behaviors
2. **Optimizing input command sequence by maximizing FIM** - Finding command sequences that maximize Fisher Information Matrix (FIM) for better parameter identification
3. **Collecting data by executing the omni controller in sim/real using the best commands** - Running the optimized commands to gather informative trajectory data
4. **Following the sysid pipeline in the main README to get the parameters** - Using the collected data with the system identification tools to extract physical parameters

---

## 1. Training the Omni Locomotion Controller

The omni locomotion controller is inspired by the multi-behavioral locomotion approach from [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways/tree/master). This controller learns to execute diverse locomotion behaviors including different gaits, speeds, and body configurations.

### Training Command

To train the omni locomotion controller:

```bash
uv run --active  python spigym/train_agent.py \
+simulator=isaacgym \
+exp=go2_omni \
+domain_rand=NO_domain_rand  \
+rewards=loco/reward_go2_omni \
+robot=go2/go2 \
+terrain=terrain_locomotion_plane \
+obs=loco/go2_omni \
num_envs=4096 \
project_name=spi-active-exps \
experiment_name=omni-controller  \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=1.0 
```

**Important:** Train the controller for **10,000 iterations** to ensure it learns diverse locomotion behaviors. The trained checkpoint will be saved in the logs folder  and can be used for the next step.

---

## 2. Optimizing Command Sequences

After training the omni controller, the next step is to find command sequences that maximize the Fisher Information Matrix (FIM) for better parameter identification. This optimization process uses CMA-ES to search for informative command trajectories.

### Configuration Parameters

You can customize the optimization process by modifying two key configuration files:

#### `spigym/config/algo/active_sysid.yaml`

This file controls the optimization algorithm and command sampling:

- **`optimize.iterations`**: Number of Optuna optimization iterations (default: 5)
sequence in seconds (default: 25s)
- **`command.horizon_length`**: Length of each command segment in seconds (default: 5s)
- **`command.command_sampling_idxs`**: Which command dimensions to optimize (default: `[0, 2, 5]` - lin_vel_x, ang_vel_yaw, gait_phase)
- **`command.command_sampling_mode`**: How to sample commands over time:
  - `"constant"` (default) - Constant commands per horizon
  - `"polynomial"` - Polynomial interpolation (set `poly_degree`)
  - `"bezier"` - Bezier curve interpolation (set `num_bezier_points`)

#### `spigym/config/env/active_sysid_openloop.yaml`

This file controls the environment setup for active exploration:

- **`delta_param`**: Finite-difference perturbation size for auxiliary environments (default: 0.1)
- **`ksync_steps`**: How often to synchronize auxiliary env states to main env (default: 5)
- **`motor_model`**: Motor dynamics model type (default: `"act2tau_vec3_tanh"`)
- **`default_param`**: Default parameter values for the robot(Stage 1 parameters can be used):
  - `mass`: Base mass in kg
  - `comx`, `comy`, `comz`: Center of mass offsets
  - `inertiax`, `inertiay`, `inertiaz`: Inertia values
  - `motor_model_hip_a`, `motor_model_thigh_a`, `motor_model_calf_a`: Motor model parameters
- **`exploration_params`**: List of parameters to that are of interest or that need to be excited during data collection (e.g., `[mass]` to identify mass)

### Running Active SysID Optimization

To run the command sequence optimization:

```bash
HYDRA_FULL_ERROR=1 \
uv run --active python spigym/run_active_sysid.py \
  +exp=active_sysid \
  +simulator=isaacgym_active_sysid \
  +checkpoint=/path/to/your/trained_model_XXXX.pt \
  +num_envs=1024 \
  env.config.headless=True \
  +project_name=active_sysid_command_gen \
  experiment_name=constant
```

Replace `/path/to/your/trained_model_XXXX.pt` with the path to your trained omni controller checkpoint from Step 1.

### Output and Best Commands

After optimization completes, the best command sequence is saved to:

```
logs/{project_name}/{experiment_name}/best_commands.npz
```

This file contains:
- **`best_commands`**: A numpy array of shape `[num_steps, command_dim]` containing the optimized command sequence

The experiment directory also contains:
- **`config.yaml`**: Full configuration used for the optimization
- **`study/study.pkl`**: Optuna study object for resuming optimization
- Analysis plots for FIM and command trajectories

You can load the best commands in Python:

```python
import numpy as np
data = np.load('path/to/experiment_dir/best_commands.npz')
best_commands = data['best_commands']
```

---

## 3. Data Collection

**Stay tuned for the next release!** Instructions for collecting data using the optimized commands will be available soon.

---

## 4. System Identification Pipeline

**Stay tuned for the next release!** Instructions for using the collected data with the system identification tools will be available soon.

---

## Tips and Best Practices

- **Number of environments**: Use at least 1024 environments for stable FIM estimation
- **Optimization iterations**: Start with 5-10 iterations and increase if needed for better results
- **Command sampling**: The default `command_sampling_idxs=[0, 2, 5]` optimizes forward velocity, yaw rate, and gait phase, which are typically most informative for mass identification
- **Delta parameter**: A `delta_param` of 0.1 (10% perturbation) works well for most cases, but you may need to adjust for very sensitive parameters

