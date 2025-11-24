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
from spigym.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from spigym.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from spigym.envs.locomotion.locomotion import LeggedRobotLocomotion
# from spigym.envs.env_utils.command_generator import CommandGenerator


EVAL_after_real = False

class go2_tracking_interface(LeggedRobotLocomotion):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        self.vel_cmd_coutner = 0
        self.state_log = defaultdict(list)
        self.data_path = "spigym/data/performance_bags/10kg.npz"
        self.data_sysid = np.load(self.data_path, allow_pickle=True)
        self.non_feet_indices = [i for i in torch.arange(self.simulator.contact_forces.shape[1]) if i not in self.feet_indices]

        self.static_f = torch.tensor(self.config.static_f, device=self.device)
        self.dynamic_f = torch.tensor(self.config.dynamic_f, device=self.device)
        self.activ = torch.tensor(self.config.activ, device=self.device)



        # walking gait
        # self.clock_inputs = torch.zeros((self.num_envs, 4), device=self.device)
        # self.doubletime_clock_inputs = torch.zeros((self.num_envs, 4), device=self.device)
        # self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        # self.foot_indices_aux = torch.zeros((self.num_envs, 4), device=self.device)
        # self.desired_contact_states = torch.zeros((self.num_envs, 4), device=self.device)
        # self.halftime_clock_inputs = torch.zeros((self.num_envs, 4), device=self.device)
        # self.desired_footswing_height = torch.zeros((self.num_envs, 4), device=self.device)

        # self.desired_foot_traj = torch.zeros((self.num_envs, 4, 3), device=self.device)

        self.debug_viz = False

    def log_state(self,key, value):
        self.state_log[key].append(value)
    
    def log_states(self, dicy):
        for key, value in dicy.items():
            self.log_state(key, value)
    
    def save_state_log(self):
        for key, value in self.state_log.items():
            self.state_log[key] = np.array(value)
        data  = dict(self.data_sysid)
        print(data.keys())
        data.update(self.state_log)
        np.savez(self.data_path, **data)
        print(f"state log saved to {self.data_path}")

    def step(self, actions):
        if EVAL_after_real:
            self._step_contact_targets()
            self._pre_physics_step(actions)
            self._physics_step()
            self.commands[:,0] = self.data_sysid['cmd_vel'][self.vel_cmd_coutner][0]
            self.commands[:,1] = self.data_sysid['cmd_vel'][self.vel_cmd_coutner][1]
            self.commands[:,2] = self.data_sysid['cmd_vel'][self.vel_cmd_coutner][2]
            self._post_physics_step()
            self.vel_cmd_coutner += 2
            # if self.episode_length_buf[0] == 1:
            #     import ipdb; ipdb.set_trace()
            self.log_states({
                    "cmd_vel_sim": self.commands[0].cpu().numpy(),
                    "base_lin_vel": self.base_lin_vel[0].cpu().numpy(),
                    "base_ang_vel": self.base_ang_vel[0].cpu().numpy(),
                    "base_quat": self.base_quat[0].cpu().numpy(),
                    "cmd_vel_sim_timestamps": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                })
            
            if self.vel_cmd_coutner >= len(self.data_sysid['cmd_vel']):
                self.save_state_log()
                exit()
            return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras
        else:
            # self._step_contact_targets()
            return super().step(actions)


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
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        if self.config.use_motor_model:
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            fric_torques = self.static_f*torch.tanh(self.simulator.dof_vel/self.activ) + self.dynamic_f*self.simulator.dof_vel
            return torques - fric_torques
        
        else:
            return torques
    # def _reset_root_states(self, env_ids, target_root_states=None):
    #     """ Resets ROOT states position and velocities of selected environmments
    #         if target_root_states is not None, reset to target_root_states
    #     Args:
    #         env_ids (List[int]): Environemnt ids
    #         target_root_states (Tensor): Target root states
    #     """
    #     if target_root_states is not None:
    #         self.simulator.robot_root_states[env_ids] = target_root_states
    #         self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]

    #     else:
    #         # base position
    #         if self.custom_origins:
    #             self.simulator.robot_root_states[env_ids] = self.base_init_state
    #             self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
    #             self.simulator.robot_root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=str(self.device)) # xy position within 1m of the center
    #         else:
    #             self.simulator.robot_root_states[env_ids] = self.base_init_state
    #             self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
    #         # base velocities
    #         self.simulator.robot_root_states[env_ids, 2] = 0.34 + 0.18*torch.rand(len(env_ids), device=str(self.device)) # z position between 0.3 and 0.7
    #         self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel
    #         # self.simulator.robot_root_states[env_ids, 9] = 0.0 + 0.5*torch.rand(len(env_ids), device=str(self.device)) # z position between 0.3 and 0.7





    def _draw_debug_vis(self):

        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            
            desired_foot_traj = self.desired_foot_traj[env_id]
            
            color = (1.0, 0.0, 0.0)
            for i in range(4):
                self.simulator.draw_sphere(desired_foot_traj[i], 0.02, color, env_id)
            



    def _step_contact_targets(self):
        frequencies = self.config.rewards.frequencies
        phases = self.config.rewards.phases
        offsets = self.config.rewards.offsets
        bounds = self.config.rewards.bounds
        durations = self.config.rewards.durations
        # print("frequencies", frequencies)
        # print("phases", phases)
        # print("offsets", offsets)
        # print("bounds", bounds)
        # print("durations", durations)
        
        
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)


        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices_aux = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)


        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (
                        0.5 / (1 - durations))

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

        # if self.cfg.commands.num_commands > 9:
        self.desired_footswing_height =0.12

    

    ### rewards for feet

    def _reward_custom_feet_air_time(self):

        return (torch.norm(self.commands[:, :2], dim=1) > 0.2)*self._reward_feet_air_time()
    
    def _reward_penalty_non_feet_contact(self):
        non_feet_contact = torch.sum(torch.sum(torch.square(self.simulator.contact_forces[:, self.non_feet_indices,:] > 1.0), dim=2),dim=1)
        return non_feet_contact

    def _reward_feet_no_contact(self):
        # returns 1 if all feet are not in contact with the ground
        feet_forces = self.simulator.contact_forces[:, self.feet_indices, :]
        feet_contact = torch.norm(feet_forces, dim=2)
        all_feet_on_ground = torch.all(feet_contact >0, dim=1)        
        return (all_feet_on_ground)*1.0*(torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_base_height(self):
        # Penalize base height away from target

        base_height = self.simulator.robot_root_states[:, 2]
        error = torch.square(base_height - self.config.rewards.desired_base_height)
        return torch.exp(-error/self.config.rewards.reward_tracking_sigma.height)*1.0
        # return torch.exp(-error/self.config.rewards.reward_tracking_sigma.height)*1.0*(torch.norm(self.commands[:, :2], dim=1) < 0.25)


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
    

    def _reward_raibert_heuristic(self):
        foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices, :3]
        base_pos = self.simulator.robot_root_states[:, :3]
        cur_footsteps_translated = foot_pos - base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat), 
              cur_footsteps_translated[:, i, :], w_last=True)
            
        desired_stance_width = self.config.rewards.desired_stance_width
        desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)
        desired_stance_length = self.config.rewards.desired_stance_length
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices_aux * 2.0)) * 1.0 - 0.5
        frequencies = self.config.rewards.frequencies
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies)
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies)

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))


        desired_footsteps_world_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            desired_footsteps_world_frame[:, i, :2] = desired_footsteps_body_frame[:, i, :2]
            desired_footsteps_world_frame[:, i, :] = quat_apply_yaw(self.base_quat, desired_footsteps_world_frame[:, i, :], w_last=True)

        self.desired_foot_traj = desired_footsteps_world_frame + base_pos.unsqueeze(1)

        reward = torch.exp(-reward/self.config.rewards.reward_tracking_sigma.height)*1.0

        return reward
    


    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices_aux * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_pos_z = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        foot_height = (foot_pos_z).view(self.num_envs, -1)# - reference_heights
        target_height = self.desired_footswing_height * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) #* (1 - self.desired_contact_states)

        self.desired_foot_traj[:, :, 2] = target_height
        
        # return torch.sum(rew_foot_clearance, dim=1)

        reward = torch.sum(rew_foot_clearance, dim=1)
        # reward = torch.exp(-reward/self.config.rewards.reward_tracking_sigma.height)*1.0
        return reward
    


    def _reward_penalty_contacts_shaped_force(self):
        foot_forces = self.simulator.contact_forces[:, self.feet_indices, :]
        foot_forces = torch.norm(foot_forces, dim=-1)
        desired_contact = self.desired_contact_states
        foot_forces_bool = (foot_forces > 0.0).float()
        reward = 0
        for i in range(4):
            reward += (1-desired_contact[:, i]) * (
                        torch.exp(-1 * foot_forces_bool[:, i] ** 2 / self.config.rewards.gait_force_sigma))
        return reward / 4
    
    def _reward_penalty_contacts_shaped_vel(self):
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        foot_vel = torch.norm(foot_vel, dim=-1).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_vel[:, i] ** 2 / self.config.rewards.gait_vel_sigma)))
        return reward / 4
    

    def _get_obs_clock(self):
        return self.clock()
    

    def clock(self):
        return torch.vstack((torch.sin(2 * torch.pi *  self.gait_indices), torch.cos(2 * torch.pi *  self.gait_indices))).T
