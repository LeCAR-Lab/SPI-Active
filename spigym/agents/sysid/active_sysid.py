"""
Active System Identification (ActiveSysId) agent.

This agent wraps PPO infrastructure to actively search for command sequences
that maximize an identification objective (e.g., Fisher Information Matrix).
It uses Optuna to sample command parameters under Hydra-configured ranges and
sampling modes, evaluates them in batched simulation, and logs the best result.

Key concepts
- Command sampling: You choose which command dimensions to optimize via
  `command.command_sampling_idxs`, their ranges via `command.command_ranges`,
  and the sampling strategy via `command.command_sampling_mode`.
- Evaluation: A batch of environments is partitioned into groups of
  (1 main + N auxiliary) envs. The environment computes the identification
  reward using differences between main and auxiliary envs.
- Outputs: The best command sequence is saved to `best_commands.npz`, and
  analysis plots for FIM and Optuna are saved under the run directory.
"""

from typing import Any
import torch
import torch.optim as optim
from omegaconf import OmegaConf
from spigym.envs.base_task.base_task import BaseTask
from spigym.agents.ppo.ppo import PPO


import os
from hydra.utils import instantiate
from loguru import logger
from rich.console import Console
from spigym.utils.average_meters import TensorAverageMeterDict
from spigym.agents.callbacks.base_callback import RL_EvalCallback
import pickle as pkl
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from rich.progress import Progress

import optuna




console = Console()



class ActiveSysId(PPO):
    """Active SysID optimizer built on PPO scaffolding.

    This class does not perform policy learning; it reuses PPO's rollout and
    evaluation utilities to repeatedly evaluate sampled command sequences and
    drive an Optuna study.
    """
    def __init__(self, env: BaseTask, config:OmegaConf, log_dir=None, device="cpu"):
        
        self.device= device
        self.env = env
        self.config = config
        self.log_dir = log_dir

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Book keeping
        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.eval_callbacks: list[RL_EvalCallback] = []
        self.episode_env_tensors = TensorAverageMeterDict()

        self.sysid_logger = []
        self.sysid_logger_path = os.path.join(self.log_dir, "sysid_logger.pkl")
        self.total_rew = torch.zeros(self.num_envs, device=self.device)
                
        self.rew_coeff = self.get_rew_coeff()
        
        
        

    def _init_config(self):
        """Load Hydra/OmegaConf configuration specific to Active SysID.

        Expects the following sub-configs (see `config/algo/active_sysid.yaml`):
        - `command`: command ranges, default command vector, which indices to
          sample, sampling mode and its params, and rollout/horizon lengths.
        - `optimize`: Optuna sampler and iterations.
        """
        super()._init_config()
        # Env related Config
        self.command_config = self.config.command
        
        self.command_sampling_idxs = self.command_config.command_sampling_idxs
        self.command_sampling_mode = self.command_config.get("command_sampling_mode", "constant")
        self.default_command = np.array(self.command_config.default_command, dtype=np.float32)
        
        # Only load mode-specific params if that mode is being used
        if self.command_sampling_mode == "polynomial":
            self.poly_degree = self.command_config.get("poly_degree", 3)
        if self.command_sampling_mode == "bezier":
            self.num_bezier_points = self.command_config.get("num_bezier_points", 4)
        
        # Calculate number of steps (not updates)
        dt = self.env.dt
        self.total_steps = int(self.command_config.rollout_length / dt)
        self.num_command_updates = self.command_config.rollout_length // self.command_config.horizon_length
        self.num_steps_per_update = int(self.command_config.horizon_length / dt)
        self.num_rollouts_per_command = self.command_config.get("num_rollouts_per_command", 1)
        
        self.optimize_config = self.config.optimize
        self._load_command_ranges()
        self.optimize_iterations = self.optimize_config.iterations
        self.study_name = self.config.project_name
        self.prev_study_path = self.optimize_config.prev_study_path
        
        self.study_dir = os.path.join(self.log_dir, "study")
        self.study_save_path = "study.pkl"
        
        
       
    
        
        self.num_envs: int = self.env.config.num_envs
        self.num_main_envs: int = self.num_envs // (len(self.env.config.exploration_params) + 1)
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.env.config.robot.actions_dim
        self.load_optimizer = self.config.load_optimizer
        
        
        
    

    

    def setup(self):
        # import ipdb; ipdb.set_trace()
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer()
        

    

    

   
 
    def _load_command_ranges(self):
        """Load and cache command ranges for sampling."""
        self.command_ranges = np.array(self.command_config.command_ranges, dtype=np.float32)
        logger.info(f"Command ranges: {self.command_ranges}")
    
    
    def optimize(self):
        """Run the active optimization loop with Optuna.

        For each iteration, this will:
        1) Ask the sampler for a batch of trials (one per main env)
        2) Turn trial params into command sequences per the sampling mode
        3) Evaluate all commands in parallel in the environment
        4) Tell the study the (negative) objective to minimize
        5) Save plots and the best command sequence
        """
        os.makedirs(self.study_dir, exist_ok=True)
        
        logger.info(f"Optimizer: {self.optimize_config.sampler}")
        sampler: optuna.samplers.BaseSampler = instantiate(self.optimize_config.sampler)
        logger.info(f"Sampler: {type(sampler)}")
        
        if self.prev_study_path:
            logger.info(f"Loding study from {self.prev_study_path}")
            self.opt_study:optuna.Study = pkl.load(open(self.prev_study_path, "rb"))
            self.opt_study.sampler = sampler
        
        else:
            self.opt_study = optuna.create_study(direction="minimize",
                                                sampler=sampler, 
                                                study_name=self.study_name, 
                                                load_if_exists=True)
        
        iter_num = self.optimize_iterations
        batch_size = self.num_main_envs
        
        self.batch_results = []
        for i in range(iter_num):
            logger.info(f"Iteration {i}")
            trails, commands = [], []
            # print("sampling")
                
            
            with Progress() as progress:
                task = progress.add_task("Sampling", total=batch_size)
                for j in range(batch_size):
                    trail = self.opt_study.ask()
                    trails.append(trail)
                    commands.append(self.sample_commands(trail, self.command_ranges, self.num_command_updates))                 
                    progress.update(task, advance=1)

            # merge params list of dicts into one dict with tensor
            command_sampled = torch.tensor(np.array(commands), device=self.device, dtype=torch.float32)
            
            reward_dict = self.evaluate_policy(command_sampled, log=False)
            for j in range(batch_size):
                env_idx = j * int(len(self.env.config.exploration_params) + 1)
                self.opt_study.tell(trails[j], -reward_dict["total_reward"][env_idx])
                self.batch_results.append(reward_dict["total_reward"][env_idx])
            
            print(f"Iteration {i}")
            print(f"Best value: {self.opt_study.best_value}")
            # print(f"Best params: {self.opt_study.best_params}")

        best_params = self.opt_study.best_params
        logger.info(f"Best params: {best_params}")
        
        # Generate best command sequence from params
        best_commands = self.generate_commands_from_params(best_params)
        logger.info(f"Best commands shape: {best_commands.shape}")
        
        # Save best commands for later use
        best_commands_path = os.path.join(self.log_dir, "best_commands.npz")
        np.savez(best_commands_path, best_commands=best_commands)
        logger.info(f"Best commands saved to {best_commands_path}")
        
        # Plot and save best commands
        self.plot_best_commands(best_commands)
        
        # Plot FIM landscape
        self.plot_FIM()
        
        # Generate Optuna visualizations
        self.plot_optuna_results()

            
        
        
        # yaml.dump(best_params, open(os.path.join(self.log_dir, "best_params.yaml"), "w"))
        # yaml.dump(self.opt_study.best_value, open(os.path.join(self.log_dir, "best_value.yaml"), "w"))
        
        # # self.plot_optimization()
        
        # #save the study to the study_dir
        # trails_df :pd.DataFrame = self.opt_study.trials_dataframe()
        # trails_df.to_csv(os.path.join(self.study_dir, "trails.csv"))
        # trails_df.to_pickle(os.path.join(self.study_dir, "trails.pkl"))
        
        # pkl.dump(self.opt_study, open(os.path.join(self.study_dir, self.study_save_path), "wb"))
        
    def sample_commands(self, trial, command_ranges, num_command_updates: int):
        """
        Sample commands based on the configured mode.
        Only samples from command_sampling_idxs, rest use default_command.
        
        Args:
            trial: Optuna trial object (None for constant sampling mode)
            command_ranges: Array of [min, max] ranges for each command dimension
            num_command_updates: Number of discrete command update points
            
        Returns:
            Array of shape (total_steps, len(default_command)) with commands for each timestep
        """
        num_dims = len(self.default_command)
        sampled_dims = len(self.command_sampling_idxs)
        
        # Sample only the specified dimensions
        if self.command_sampling_mode == "constant":
            sampled_commands = self._sample_constant(trial, command_ranges, num_command_updates, sampled_dims)
        elif self.command_sampling_mode == "polynomial":
            sampled_commands = self._sample_polynomial(trial, command_ranges, num_command_updates, sampled_dims)
        elif self.command_sampling_mode == "bezier":
            sampled_commands = self._sample_bezier(trial, command_ranges, num_command_updates, sampled_dims)
        else:
            raise ValueError(f"Unknown command_sampling_mode: {self.command_sampling_mode}")
        
        # Expand to full dimensions using default_command for non-sampled dims
        full_commands = np.zeros((sampled_commands.shape[0], num_dims), dtype=np.float32)
        full_commands[:, :] = self.default_command  # Start with defaults
        
        # Fill in sampled dimensions
        for idx, sampled_idx in enumerate(self.command_sampling_idxs):
            full_commands[:, sampled_idx] = sampled_commands[:, idx]
        
        return full_commands.astype(np.float32)
    
    def _sample_constant(self, trial, command_ranges, num_command_updates, num_dims):
        """Sample constant commands for each horizon.

        Each sampled dimension gets one scalar per update window; values are
        repeated for all timesteps inside the window.
        """
        # Sample one value per horizon for each dimension
        commands_updates = np.zeros((num_command_updates, num_dims), dtype=np.float32)
        for dim_idx in range(num_dims):
            actual_dim = self.command_sampling_idxs[dim_idx]
            for update in range(num_command_updates):
                commands_updates[update, dim_idx] = trial.suggest_float(
                    f"dim_{actual_dim}_update_{update}", 
                    command_ranges[actual_dim][0], 
                    command_ranges[actual_dim][1]
                )
        
        # Expand to all timesteps (repeat each update for num_steps_per_update)
        commands = np.repeat(commands_updates, self.num_steps_per_update, axis=0)
        return commands.astype(np.float32)
    
    def _sample_polynomial(self, trial, command_ranges, num_command_updates, num_dims):
        """Sample piecewise polynomial curves per update window.

        Coefficients are sampled in a bounded space for stability; the curve is
        squashed with tanh to [-1, 1] then linearly mapped to the target range.
        """
        num_coeffs = self.poly_degree + 1
        coeff_bound = 2.0

        t = np.linspace(0.0, 1.0, self.num_steps_per_update, dtype=np.float32)
        commands_list = []

        for update in range(num_command_updates):
            # Sample coefficients per dim in normalized space
            coeffs = []
            for dim_idx in range(num_dims):
                actual_dim = self.command_sampling_idxs[dim_idx]
                dim_coeffs = [
                    trial.suggest_float(
                        f"dim_{actual_dim}_update_{update}_coeff_{i}",
                        -coeff_bound,
                        coeff_bound,
                    )
                    for i in range(num_coeffs)
                ]
                coeffs.append(dim_coeffs)

            # Evaluate and map to command ranges
            segment = np.zeros((self.num_steps_per_update, num_dims), dtype=np.float32)
            for dim_idx in range(num_dims):
                actual_dim = self.command_sampling_idxs[dim_idx]
                # raw polynomial in R
                poly = np.zeros_like(t)
                for i, c in enumerate(coeffs[dim_idx]):
                    poly += c * (t ** i)
                # squash to [-1, 1]
                poly = np.tanh(poly)
                # map to [min, max]
                vmin, vmax = command_ranges[actual_dim]
                segment[:, dim_idx] = vmin + (poly + 1.0) * 0.5 * (vmax - vmin)

            commands_list.append(segment.astype(np.float32))

        commands = np.vstack(commands_list)
        return commands.astype(np.float32)
    
    def _sample_bezier(self, trial, command_ranges, num_command_updates, num_dims):
        """Sample Bezier spline control points."""
        # Sample control points for each dimension
        commands_list = []
        
        for update in range(num_command_updates):
            # Sample control points for this segment
            control_points = np.zeros((self.num_bezier_points, num_dims), dtype=np.float32)
            for dim_idx in range(num_dims):
                actual_dim = self.command_sampling_idxs[dim_idx]
                for point in range(self.num_bezier_points):
                    control_points[point, dim_idx] = trial.suggest_float(
                        f"dim_{actual_dim}_update_{update}_bezier_{point}",
                        command_ranges[actual_dim][0], 
                        command_ranges[actual_dim][1]
                    )
            
            # Generate Bezier curve for this horizon
            t = np.linspace(0, 1, self.num_steps_per_update, dtype=np.float32)
            segment = np.zeros((self.num_steps_per_update, num_dims), dtype=np.float32)
            
            for dim_idx in range(num_dims):
                actual_dim = self.command_sampling_idxs[dim_idx]
                for i in range(self.num_steps_per_update):
                    # Bezier curve evaluation using Bernstein basis
                    bezier_value = 0.0
                    for j in range(self.num_bezier_points):
                        # Binomial coefficient
                        n = self.num_bezier_points - 1
                        binom_coeff = self._binomial_coefficient(n, j)
                        # Bernstein polynomial
                        bernstein = binom_coeff * (t[i] ** j) * ((1 - t[i]) ** (n - j))
                        bezier_value += bernstein * control_points[j, dim_idx]
                    segment[i, dim_idx] = np.clip(bezier_value, command_ranges[actual_dim][0], command_ranges[actual_dim][1])
            
            commands_list.append(segment.astype(np.float32))
        
        commands = np.vstack(commands_list)
        return commands.astype(np.float32)
    
    def _binomial_coefficient(self, n, k):
        """Calculate binomial coefficient C(n, k)."""
        from math import factorial
        if k > n - k:
            k = n - k  # Take advantage of symmetry
        result = 1
        for i in range(k):
            result *= (n - i) / (i + 1)
        return int(result)

    def generate_commands_from_params(self, best_params_dict):
        """
        Generate command sequences from Optuna best_params dict.
        This reconstructs the full command trajectory from sampled parameters.
        
        Args:
            best_params_dict: Dict from optuna study.best_params
            
        Returns:
            Array of shape (total_steps, len(default_command)) with commands
        """
        num_dims = len(self.default_command)
        sampled_dims = len(self.command_sampling_idxs)
        
        if self.command_sampling_mode == "constant":
            sampled_commands = self._generate_constant_from_params(best_params_dict)
        elif self.command_sampling_mode == "polynomial":
            sampled_commands = self._generate_polynomial_from_params(best_params_dict)
        elif self.command_sampling_mode == "bezier":
            sampled_commands = self._generate_bezier_from_params(best_params_dict)
        else:
            raise ValueError(f"Unknown command_sampling_mode: {self.command_sampling_mode}")
        
        # Expand to full dimensions using default_command for non-sampled dims
        full_commands = np.zeros((sampled_commands.shape[0], num_dims), dtype=np.float32)
        full_commands[:, :] = self.default_command  # Start with defaults
        
        # Fill in sampled dimensions
        for idx, sampled_idx in enumerate(self.command_sampling_idxs):
            full_commands[:, sampled_idx] = sampled_commands[:, idx]
        
        return full_commands.astype(np.float32)
    
    def _generate_constant_from_params(self, best_params_dict):
        """Reconstruct constant commands from params dict."""
        commands_updates = np.zeros((self.num_command_updates, len(self.command_sampling_idxs)), dtype=np.float32)
        for dim_idx, actual_dim in enumerate(self.command_sampling_idxs):
            for update in range(self.num_command_updates):
                key = f"dim_{actual_dim}_update_{update}"
                commands_updates[update, dim_idx] = best_params_dict[key]
        
        # Expand to all timesteps
        commands = np.repeat(commands_updates, self.num_steps_per_update, axis=0)
        return commands.astype(np.float32)
    
    def _generate_polynomial_from_params(self, best_params_dict):
        """Reconstruct polynomial commands consistent with normalized/tanh mapping."""
        num_coeffs = self.poly_degree + 1
        t = np.linspace(0.0, 1.0, self.num_steps_per_update, dtype=np.float32)
        commands_list = []

        for update in range(self.num_command_updates):
            # Reconstruct coefficients
            coeffs = []
            for dim_idx, actual_dim in enumerate(self.command_sampling_idxs):
                dim_coeffs = [
                    best_params_dict[f"dim_{actual_dim}_update_{update}_coeff_{i}"]
                    for i in range(num_coeffs)
                ]
                coeffs.append(dim_coeffs)

            # Evaluate and map to command ranges
            segment = np.zeros((self.num_steps_per_update, len(self.command_sampling_idxs)), dtype=np.float32)
            for dim_idx, actual_dim in enumerate(self.command_sampling_idxs):
                poly = np.zeros_like(t)
                for i, c in enumerate(coeffs[dim_idx]):
                    poly += c * (t ** i)
                poly = np.tanh(poly)
                vmin, vmax = self.command_ranges[actual_dim]
                segment[:, dim_idx] = vmin + (poly + 1.0) * 0.5 * (vmax - vmin)

            commands_list.append(segment.astype(np.float32))

        commands = np.vstack(commands_list)
        return commands.astype(np.float32)
    
    def _generate_bezier_from_params(self, best_params_dict):
        """Reconstruct Bezier commands from params dict."""
        commands_list = []
        
        for update in range(self.num_command_updates):
            # Reconstruct control points
            control_points = np.zeros((self.num_bezier_points, len(self.command_sampling_idxs)), dtype=np.float32)
            for dim_idx, actual_dim in enumerate(self.command_sampling_idxs):
                for point in range(self.num_bezier_points):
                    key = f"dim_{actual_dim}_update_{update}_bezier_{point}"
                    control_points[point, dim_idx] = best_params_dict[key]
            
            # Generate Bezier curve
            t = np.linspace(0, 1, self.num_steps_per_update, dtype=np.float32)
            segment = np.zeros((self.num_steps_per_update, len(self.command_sampling_idxs)), dtype=np.float32)
            
            for dim_idx, actual_dim in enumerate(self.command_sampling_idxs):
                for i in range(self.num_steps_per_update):
                    bezier_value = 0.0
                    for j in range(self.num_bezier_points):
                        n = self.num_bezier_points - 1
                        binom_coeff = self._binomial_coefficient(n, j)
                        bernstein = binom_coeff * (t[i] ** j) * ((1 - t[i]) ** (n - j))
                        bezier_value += bernstein * control_points[j, dim_idx]
                    segment[i, dim_idx] = np.clip(bezier_value,
                                                  self.command_ranges[actual_dim][0],
                                                  self.command_ranges[actual_dim][1])
            
            commands_list.append(segment.astype(np.float32))
        
        commands = np.vstack(commands_list)
        return commands.astype(np.float32)
    
    
    
    ##########################################################################################
    # Code for Policy Evaluation
    ##########################################################################################

    @torch.no_grad()
    def evaluate_policy(self, commands, log=False):
        """
        Evaluate policy with multiple rollouts for expectation estimation.
        Runs num_rollouts_per_command times and averages results.
        """

        self._create_eval_callbacks()
        actor_state = self._create_actor_state()
        self.eval_policy = self._get_inference_policy()
        
        all_rollout_rewards = []
        
        for rollout in range(self.num_rollouts_per_command):
            
            self.reward_dict = {"total_reward": torch.zeros(self.env.num_envs, device=self.device)}
            obs_dict = self._pre_evaluate_policy(reset_env=True, commands=commands)
            
            step = 1
            
            init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device, dtype=torch.float32)
            init_dones = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float32)
            actor_state.update({"obs": obs_dict, "actions": init_actions, "dones": init_dones})
            
            terminated_envs = init_dones.nonzero(as_tuple=True)
            logger.info(f"total steps: {self.total_steps}")

            with Progress() as progress:
                task = progress.add_task(f"Evaluating rollout {rollout+1}/{self.num_rollouts_per_command}", total=self.total_steps - 1)
                for i in range(1,self.total_steps-1):
                    actor_state["step"] = step
                    actor_state = self._pre_eval_env_step(actor_state)
                    # Zero actions for terminated env indices correctly
                    term_idx = terminated_envs[0]
                    if term_idx.numel() > 0:
                        actor_state['actions'][term_idx] = 0.0
                    actor_state = self.env_step(actor_state)
                    actor_state = self._post_eval_env_step(actor_state)
                    terminated_envs = actor_state["dones"].nonzero(as_tuple=True)
                        
                    to_log = actor_state["extras"]["to_log"]
                    for key in to_log.keys():
                        if 'rew' in key:
                            reward_scale = self.rew_coeff[key]
                            to_log[key][terminated_envs] = self.command_config.termination_rew
                            if key not in self.reward_dict.keys():
            
                                self.reward_dict[key] = to_log[key] * reward_scale
                            else:
                                self.reward_dict[key] += to_log[key] * reward_scale
                            self.reward_dict["total_reward"] += to_log[key] * reward_scale
                    
                    step += 1
                    
                    
                    
                    progress.update(task, advance=1)
                    
            logger.info(f"Finished rollout {rollout+1}/{self.num_rollouts_per_command}")
            
            # Average by step and convert to numpy
            rollout_reward_dict = {}
            for key in self.reward_dict.keys():
                rollout_reward_dict[key] = (self.reward_dict[key] / step).cpu().numpy()
            
            all_rollout_rewards.append(rollout_reward_dict)
        
        # Average across all rollouts (expectation estimation)
        avg_reward_dict = {}
        for key in all_rollout_rewards[0].keys():
            avg_reward_dict[key] = np.mean([rr[key] for rr in all_rollout_rewards], axis=0)
        
        self._post_evaluate_policy()
        return avg_reward_dict
    
    def _pre_evaluate_policy(self, reset_env=True, commands=None):
        self._eval_mode()
        self.env.set_is_evaluating()
        obs_dict = None
        if reset_env and commands is not None:
            obs_dict = self.env.reset_all(commands)
        elif reset_env:
            obs_dict = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()
        
        return obs_dict

    

    def get_rew_coeff(self):
        rew_coeff = self.env.config.rewards.reward_scales
        new_rew_coeff = {}
        for key, val in rew_coeff.items():
            new_rew_coeff["rew_" + key] = val
        return new_rew_coeff



    def plot_FIM(self):
        """Plot FIM optimization landscape and convergence."""
        # Convert batch_results to numpy for easier analysis
        fim_values = np.array(self.batch_results)
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Raw FIM values over all trials
        axes[0, 0].plot(fim_values, alpha=0.5, linewidth=1)
        axes[0, 0].set_title("Raw FIM Values Over Trials")
        axes[0, 0].set_xlabel("Trial Number")
        axes[0, 0].set_ylabel("FIM Value")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Best FIM value over time (cumulative minimum)
        best_so_far = np.minimum.accumulate(fim_values)
        axes[0, 1].plot(best_so_far, 'r-', linewidth=2)
        axes[0, 1].set_title("Best FIM Value (Cumulative Minimum)")
        axes[0, 1].set_xlabel("Trial Number")
        axes[0, 1].set_ylabel("Best FIM Value")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: FIM distribution histogram
        axes[1, 0].hist(fim_values, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title("FIM Value Distribution")
        axes[1, 0].set_xlabel("FIM Value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: FIM statistics per iteration (if we know batch size)
        batch_size = len(fim_values) // self.optimize_iterations
        if batch_size > 0:
            fim_per_iteration = fim_values.reshape(self.optimize_iterations, -1)
            means = np.mean(fim_per_iteration, axis=1)
            stds = np.std(fim_per_iteration, axis=1)
            
            x_iter = np.arange(len(means)) + 1
            axes[1, 1].errorbar(x_iter, means, yerr=stds, fmt='o-', 
                              capsize=5, capthick=2, linewidth=2)
            axes[1, 1].set_title("FIM Statistics Per Iteration")
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("FIM Value")
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.log_dir, "fim_landscape.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"FIM landscape plot saved to {plot_path}")
        
        plt.close(fig)
    
    def plot_optuna_results(self):
        """Generate Optuna visualization plots for the optimization study."""
        try:
            # Get the study dataframe with all trials
            df = self.opt_study.trials_dataframe(attrs=('number', 'value', 'state', 'datetime_start', 'datetime_complete'))
            
            # Save dataframe to CSV
            df.to_csv(os.path.join(self.log_dir, "optuna_trials.csv"))
            logger.info("Optuna trials dataframe saved to optuna_trials.csv")
            
            # Create comprehensive matplotlib plots from the dataframe
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Optimization history
            if 'value' in df.columns:
                df_plot = df[df['state'] == 'COMPLETE'].copy()
                df_plot = df_plot.sort_values('number')
                
                axes[0, 0].scatter(df_plot['number'], df_plot['value'], alpha=0.5, s=20, label='All trials')
                
                # Plot best value so far
                best_so_far = []
                best_val = float('inf')
                for idx, val in enumerate(df_plot['value']):
                    if val < best_val:
                        best_val = val
                    best_so_far.append(best_val)
                
                axes[0, 0].plot(df_plot['number'], best_so_far, 'r-', linewidth=2, label='Best so far')
                axes[0, 0].set_xlabel('Trial Number')
                axes[0, 0].set_ylabel('Objective Value')
                axes[0, 0].set_title('Optimization History')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Value distribution
            if 'value' in df.columns:
                df_complete = df[df['state'] == 'COMPLETE']
                axes[0, 1].hist(df_complete['value'], bins=50, alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(df_complete['value'].min(), color='r', linestyle='--', 
                                   linewidth=2, label=f'Best: {df_complete["value"].min():.4f}')
                axes[0, 1].axvline(df_complete['value'].mean(), color='b', linestyle='--', 
                                   linewidth=2, label=f'Mean: {df_complete["value"].mean():.4f}')
                axes[0, 1].set_xlabel('Objective Value')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Objective Value Distribution')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Parameter plots for each dimension (if sampled dimensions are small enough)
            sampled_dims = len(self.command_sampling_idxs)
            if sampled_dims <= 3:  # Only plot if manageable number of params
                param_cols = [col for col in df.columns if col.startswith('params_dim_')]
                for idx, param_col in enumerate(param_cols[:min(3, len(param_cols))]):
                    df_param = df[df['state'] == 'COMPLETE'].copy()
                    axes[1, idx].scatter(df_param[param_col], df_param['value'], alpha=0.5, s=20)
                    axes[1, idx].set_xlabel(param_col.replace('params_', ''))
                    axes[1, idx].set_ylabel('Objective Value')
                    axes[1, idx].set_title(f'Parameter: {param_col.replace("params_", "")}')
                    axes[1, idx].grid(True, alpha=0.3)
            
            # Hide unused subplot
            if sampled_dims < 3:
                for idx in range(sampled_dims, 3):
                    axes[1, idx].axis('off')
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.log_dir, "optuna_analysis.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Optuna analysis plot saved to {plot_path}")
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error generating Optuna plots: {e}")
    
    def plot_best_commands(self, best_commands):
        """Plot the best command sequences for visualization."""
        num_dims = best_commands.shape[1]
        num_steps = best_commands.shape[0]
        
        # Create subplots - arrange in a grid
        n_cols = 4
        n_rows = int(np.ceil(num_dims / n_cols))
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axs = axs.reshape(1, -1)
        axs = axs.flatten()
        
        # Time axis
        time_steps = np.arange(num_steps, dtype=np.float32) * self.env.dt
        
        # Plot each dimension
        for dim in range(num_dims):
            ax = axs[dim]
            ax.plot(time_steps, best_commands[:, dim], linewidth=2)
            ax.set_title(f"Command Dim {dim}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for dim in range(num_dims, len(axs)):
            axs[dim].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.log_dir, "best_commands.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Best commands plot saved to {plot_path}")
        
        plt.close(fig)
