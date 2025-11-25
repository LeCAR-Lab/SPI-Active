from time import time
from warnings import WarningMessage
import numpy as np
import os
from datetime import datetime
from spigym.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress
from collections import defaultdict
from isaac_utils.rotations import quat_apply_yaw
from spigym.envs.locomotion.go2_locomotion import go2_locomotion



EVAL_after_real = False

class go2_omni_interface(go2_locomotion):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        self.non_feet_indices = [i for i in torch.arange(self.simulator.contact_forces.shape[1]) if i not in self.feet_indices]
        
        
                
        # walking gait
        self.clock_inputs = torch.zeros((self.num_envs, 4), device=self.device)
        self.doubletime_clock_inputs = torch.zeros((self.num_envs, 4), device=self.device)
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.foot_indices_aux = torch.zeros((self.num_envs, 4), device=self.device)
        self.desired_contact_states = torch.zeros((self.num_envs, 4), device=self.device)
        self.halftime_clock_inputs = torch.zeros((self.num_envs, 4), device=self.device)
        self.desired_footswing_height = torch.zeros((self.num_envs, 4), device=self.device)

        self.desired_foot_traj = torch.zeros((self.num_envs, 4, 3), device=self.device)
        self.curriculum_thresholds = self.config.obs.curriculum_thresholds
        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))
        
    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros((self.num_envs, self.config.obs.commands.num_commands), dtype=torch.float32, device=self.device)
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device)
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device)
        
    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.commands = torch.zeros((self.num_envs, self.config.obs.commands.num_commands), dtype=torch.float32, device=self.device)
        # TODO: haotian: adding command configuration
        self.commands[:,0] = 0.0
        self.commands[:,1] = 0.0
        self.commands[:,2] = 0.0
        self.commands[:,3] = 0.0
        self.commands[:,4] = 0.5
        self.commands[:,5] = 0.0
        self.commands[:,6] = 0.5
        self.commands[:,7] = 0.0
        self.commands[:,8] = 0.5
        self.commands[:,9] = 0.08
        self.commands[:,10] = 0.0
        self.commands[:,11] = 0.0
        self.commands[:,12] = 0.25
        self.commands[:,13] = 0.40


    def _compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            try:
                assert rew.shape[0] == self.num_envs
            except:
                import ipdb; ipdb.set_trace()
            # penalty curriculum
            if name in self.config.rewards.reward_penalty_reward_names:
                if self.config.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            else:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
            

        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.config.rewards.omni_style:
            self.rew_buf[:] = self.rew_buf_pos[:]*torch.exp(self.rew_buf_neg[:]/self.config.rewards.sigma_rew_neg)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew
        
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1


        if self.use_reward_penalty_curriculum:
            self.log_dict["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float)
            self.log_dict["average_episode_length"] = self.average_episode_length
    
    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        super().reset_envs_idx(env_ids, target_states, target_buf)
        self.gait_indices[env_ids] = 0
    
    

    def clock(self):
        return torch.vstack((torch.sin(2 * torch.pi *  self.gait_indices), torch.cos(2 * torch.pi *  self.gait_indices))).T
    
    def _update_tasks_callback(self):
        # super(LeggedRobotLocomotion,self).update_tasks_callback()
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        
    def _init_command_distribution(self,env_ids):
        self.category_names = ['nominal']
        if self.config.obs.commands.gaitwise_curricula:
            self.category_names = ['pronk','trot','pace','bound']
        
        if self.config.obs.commands.curriculum_type == "RewardThresholdCurriculum":
            from spigym.envs.env_utils.curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula+= [CurriculumClass(seed=self.config.obs.commands.curriculum_seed,
                                               x_vel=(self.config.obs.commands.limit_vel_x[0],
                                                      self.config.obs.commands.limit_vel_x[1],
                                                      self.config.obs.commands.num_bins_vel_x),
                                               y_vel=(self.config.obs.commands.limit_vel_y[0],
                                                      self.config.obs.commands.limit_vel_y[1],
                                                      self.config.obs.commands.num_bins_vel_y),
                                               yaw_vel=(self.config.obs.commands.limit_vel_yaw[0],
                                                        self.config.obs.commands.limit_vel_yaw[1],
                                                        self.config.obs.commands.num_bins_vel_yaw),
                                               body_height=(self.config.obs.commands.limit_body_height[0],
                                                            self.config.obs.commands.limit_body_height[1],
                                                            self.config.obs.commands.num_bins_body_height),
                                               gait_frequency=(self.config.obs.commands.limit_gait_frequency[0],
                                                               self.config.obs.commands.limit_gait_frequency[1],
                                                               self.config.obs.commands.num_bins_gait_frequency),
                                               gait_phase=(self.config.obs.commands.limit_gait_phase[0],
                                                           self.config.obs.commands.limit_gait_phase[1],
                                                           self.config.obs.commands.num_bins_gait_phase),
                                               gait_offset=(self.config.obs.commands.limit_gait_offset[0],
                                                            self.config.obs.commands.limit_gait_offset[1],
                                                            self.config.obs.commands.num_bins_gait_offset),
                                               gait_bounds=(self.config.obs.commands.limit_gait_bound[0],
                                                            self.config.obs.commands.limit_gait_bound[1],
                                                            self.config.obs.commands.num_bins_gait_bound),
                                               gait_duration=(self.config.obs.commands.limit_gait_duration[0],
                                                              self.config.obs.commands.limit_gait_duration[1],
                                                              self.config.obs.commands.num_bins_gait_duration),
                                               footswing_height=(self.config.obs.commands.limit_footswing_height[0],
                                                                 self.config.obs.commands.limit_footswing_height[1],
                                                                 self.config.obs.commands.num_bins_footswing_height),
                                               body_pitch=(self.config.obs.commands.limit_body_pitch[0],
                                                           self.config.obs.commands.limit_body_pitch[1],
                                                           self.config.obs.commands.num_bins_body_pitch),
                                               body_roll=(self.config.obs.commands.limit_body_roll[0],
                                                          self.config.obs.commands.limit_body_roll[1],
                                                          self.config.obs.commands.num_bins_body_roll),
                                               stance_width=(self.config.obs.commands.limit_stance_width[0],
                                                             self.config.obs.commands.limit_stance_width[1],
                                                             self.config.obs.commands.num_bins_stance_width),
                                               stance_length=(self.config.obs.commands.limit_stance_length[0],
                                                                self.config.obs.commands.limit_stance_length[1],
                                                                self.config.obs.commands.num_bins_stance_length)
                                               )]
        if self.config.obs.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.config.obs.commands.lipschitz_threshold,
                                      binary_phases=self.config.obs.commands.binary_phases)
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int64)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int64)
        low = np.array(
            [self.config.obs.commands.lin_vel_x[0], self.config.obs.commands.lin_vel_y[0],
             self.config.obs.commands.ang_vel_yaw[0], self.config.obs.commands.body_height_cmd[0],
             self.config.obs.commands.gait_frequency_cmd_range[0],
             self.config.obs.commands.gait_phase_cmd_range[0], self.config.obs.commands.gait_offset_cmd_range[0],
             self.config.obs.commands.gait_bound_cmd_range[0], self.config.obs.commands.gait_duration_cmd_range[0],
             self.config.obs.commands.footswing_height_range[0], self.config.obs.commands.body_pitch_range[0],
             self.config.obs.commands.body_roll_range[0],self.config.obs.commands.stance_width_range[0],
             self.config.obs.commands.stance_length_range[0], ])
        high = np.array(
            [self.config.obs.commands.lin_vel_x[1], self.config.obs.commands.lin_vel_y[1],
             self.config.obs.commands.ang_vel_yaw[1], self.config.obs.commands.body_height_cmd[1],
             self.config.obs.commands.gait_frequency_cmd_range[1],
             self.config.obs.commands.gait_phase_cmd_range[1], self.config.obs.commands.gait_offset_cmd_range[1],
             self.config.obs.commands.gait_bound_cmd_range[1], self.config.obs.commands.gait_duration_cmd_range[1],
             self.config.obs.commands.footswing_height_range[1], self.config.obs.commands.body_pitch_range[1],
             self.config.obs.commands.body_roll_range[1],self.config.obs.commands.stance_width_range[1],
             self.config.obs.commands.stance_length_range[1], ])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def step(self, actions):
        if EVAL_after_real:
            self._step_contact_targets()
            self._pre_physics_step(actions)
            self._physics_step()
            self._post_physics_step()
            # if self.episode_length_buf[0] == 1:
            #     import ipdb; ipdb.set_trace()
            
            return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras
        else:
            self._step_contact_targets()
            return super().step(actions)
    
    def _resample_commands(self, env_ids):
        timesteps = int(self.config.locomotion_command_resampling_time / self.dt)
        ep_len = min(timesteps,self.max_episode_length)
        
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue
            
            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0]))
        
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]
        
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i
            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.config.obs.commands.num_commands]).to(
                self.device)
        
        if self.config.obs.commands.num_commands > 5:
            if self.config.obs.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.config.obs.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.config.obs.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.config.obs.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

    def _step_contact_targets(self):
        frequencies = self.commands[:,4]
        phases = self.commands[:,5]
        offsets = self.commands[:,6]
        bounds = self.commands[:,7]
        durations = self.commands[:,8]

        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)


        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices_aux = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

        self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.config.obs.commands.num_commands > 9:
            self.desired_footswing_height =self.commands[:,9]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # actions *= 0.
        # print("self.simulator.dof_vel", self.simulator.dof_vel)
        # print("actions", actions)
        actions_scaled = actions * self.config.robot.control.action_scale
        actions_scaled[:,[0,3,6,9]]*=0.5
        control_type = self.config.robot.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if hasattr(self.config.domain_rand, 'randomize_motor_model') and self.config.domain_rand.randomize_motor_model:
            # randomize motor strength between range given by config
            A_hip = torch_rand_float(self.config.domain_rand.motor_model_range.hip[0], self.config.domain_rand.motor_model_range.hip[1], (self.num_envs, 1), device=self.device)
            A_thigh = torch_rand_float(self.config.domain_rand.motor_model_range.thigh[0], self.config.domain_rand.motor_model_range.thigh[1], (self.num_envs, 1), device=self.device)
            A_calf = torch_rand_float(self.config.domain_rand.motor_model_range.calf[0], self.config.domain_rand.motor_model_range.calf[1], (self.num_envs, 1), device=self.device)
            A = torch.cat([A_hip, A_thigh, A_calf, A_hip, A_thigh, A_calf, A_hip, A_thigh, A_calf, A_hip, A_thigh, A_calf], dim=1)
            # B = torch_rand_float(self.config.domain_rand.motor_model_range.B[0], self.config.domain_rand.motor_model_range.B[1], (self.num_envs, self.num_dofs), device=self.device)
            
            torques = A*torch.tanh(1*torques/A)

        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques

    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            
            # self.simulator.draw_sphere(self.simulator)

            for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
                if self.config.robot.motion.visualization.customize_color:
                    color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                else:
                    color_inner = (0.3, 0.3, 0.3)
                color_inner = tuple(color_inner)

                self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id)

                if pos_id in self.motion_tracking_id:
                    color_schems = (0.851, 0.144, 0.07)
                    start_point = self._rigid_body_pos_extend[env_id, pos_id]
                    end_point = pos_joint
                    line_width = 0.03
                    for _ in range(50):
                        self.simulator.draw_line(Point(start_point +torch.rand(3, device=self.device) * line_width),
                                            Point(end_point + torch.rand(3, device=self.device) * line_width),
                                            Point(color_schems),
                                            env_id)

    '''REWARDS'''

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.simulator.contact_forces[:, self.penalised_contact_indices, :], dim=-1)>0.1), dim=1)
    
    def _reward_jump(self):
        body_height = self.simulator.robot_root_states[:, 2]
        jump_height_target = self.commands[:, 3]+ self.config.rewards.base_height_target
        return -torch.square(body_height - jump_height_target)
    
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = self.simulator.contact_forces[:, self.feet_indices, :]
        foot_forces = torch.norm(foot_forces, dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        for i in range(4):
            reward += (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.config.rewards.gait_force_sigma))
        return reward / 4
    
    def _reward_tracking_contacts_shaped_vel(self):
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        foot_vel = torch.norm(foot_vel, dim=-1).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_vel[:, i] ** 2 / self.config.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_dof_pos(self):
        #penalize dof positions 
        return torch.sum(torch.square(self.simulator.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.simulator._rigid_body_pos[:, self.feet_indices, 2] - reference_heights<0.03
        foot_vel = torch.square(torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices], dim=2).view(self.num_envs, -1))
        return torch.sum(near_ground*foot_vel, dim=1)
    
    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices_aux * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_pos_z = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        foot_height = (foot_pos_z).view(self.num_envs, -1)# - reference_heights
        target_height = self.commands[:,9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)

        return torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)
    
    def _reward_raibert_heuristic(self):
        foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices, :3]
        base_pos = self.simulator.robot_root_states[:, :3]
        cur_footsteps_translated = foot_pos - base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat), 
              cur_footsteps_translated[:, i, :], w_last=True)
        
        if self.config.obs.commands.num_commands >=13:   
            desired_stance_width = self.commands[:,12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = self.config.rewards.desired_stance_width
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

        if self.config.obs.commands.num_commands >=14:
            desired_stance_length = self.commands[:,13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = self.config.rewards.desired_stance_length
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)
        
        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices_aux * 2.0)) * 1.0 - 0.5
        frequencies = self.commands[:,4]
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
    ### MISC feet rewards####
    def _reward_penalty_non_feet_contact(self):
        non_feet_contact = torch.sum(torch.sum(torch.square(self.simulator.contact_forces[:, self.non_feet_indices,:]), dim=2),dim=1)
        return non_feet_contact

    def _reward_feet_no_contact(self):
        # returns 1 if all feet are not in contact with the ground
        feet_forces = self.simulator.contact_forces[:, self.feet_indices, :]
        feet_contact = torch.norm(feet_forces, dim=2)
        all_feet_on_ground = torch.all(feet_contact >0, dim=1)        
        return (all_feet_on_ground)*1.0*(torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_penalty_close_feet_xy(self):
        # returns 1 if two feet are too close
        foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices, :2]
        fl, fr, rl, rr = foot_pos[:, 0], foot_pos[:, 1], foot_pos[:, 2], foot_pos[:, 3]
        f_feet_dist = torch.norm(fl - fr, dim=1)
        r_feet_dist = torch.norm(rl - rr, dim=1)
        # cost = torch.sum(feet_distance_xy < self.config.rewards.close_feet_threshold, dim=1)
        cost_l = f_feet_dist < self.config.rewards.close_feet_threshold
        cost_r = r_feet_dist < self.config.rewards.close_feet_threshold
        return (cost_l.float() + cost_r.float())*1.0
    

    '''observations'''

    def _get_obs_command_body_height(self):
        # During eval, drive the height command with a sinusoid
        # centered at the default eval command (index 3), clamped to limits.
        if getattr(self, "is_evaluating", False):
            center = self.commands[:, 3:4]
            # Use gait phase as the oscillator; amplitude kept conservative
            phase = self.gait_indices.unsqueeze(1)  # [N, 1], in [0, 1)
            amplitude = 0.20  # meters of offset around center
            height_cmd = center + amplitude * torch.sin(2 * np.pi * phase)

            # Clamp within configured limits to be safe
            low, high = self.config.obs.commands.limit_body_height[0], self.config.obs.commands.limit_body_height[1]
            # print("height_cmd", height_cmd)
            return torch.clamp(height_cmd, min=low, max=high)
        # Training or non-eval: pass through command as-is
        return self.commands[:, 3:4]
    
    def _get_obs_command_gait_freq(self):
        return self.commands[:,4:5]
    
    def _get_obs_command_gait_phase(self):
        return self.commands[:,5:9]
    
    def _get_obs_command_footswing_height(self):
        return self.commands[:,9:10]
    
    def _get_obs_command_body_attitude(self):
        return self.commands[:,10:12]
    
    def _get_obs_command_stance(self):
        return self.commands[:,12:14]
    
    
    
    def _get_obs_last_action(self):
        return self.last_actions

    def _get_obs_timing_param(self):
        return self.gait_indices
    
    def _get_obs_clock_inputs(self):
        return self.clock_inputs
    
    def _get_obs_yaw(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        return heading

    def _get_obs_contact_states(self):
        return (self.simulator.contact_forces[:, self.feet_indices, 2]>1.0).view(self.num_envs, -1)
    
    
    def _get_obs_desired_contact_states(self):
        return self.desired_contact_states
