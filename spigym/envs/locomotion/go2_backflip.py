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
from spigym.utils.torch_utils import *

EVAL_after_real = False

def torch_rand_choice(list_of_phase_lens,shape,device):
    values_tensor = torch.tensor(list_of_phase_lens, device=device,dtype=torch.float32)
    return values_tensor[torch.randint(0, len(list_of_phase_lens), shape)]

class go2_backflip(LeggedRobotLocomotion):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        self.state_log = defaultdict(list)
        self.data_path = "spigym/data/performance_bags/side_flip.npz"
        self.phase = 0*self.episode_length_buf

        # self.data_sysid = np.load(self.data_path, allow_pickle=True)
        # self.phase_len = self.config.rewards.phase_length*torch.ones_like(self.episode_length_buf.to(torch.float32))
        self.non_feet_indices = [i for i in torch.arange(self.simulator.contact_forces.shape[1]) if i not in self.feet_indices]
        

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
    #         # self.simulator.robot_root_states[env_ids, 2] = 0.3 + 0.1*torch.rand(len(env_ids), device=str(self.device)) # z position between 0.3 and 0.7
    #         # self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.3, 0.3, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel
    #         # self.simulator.robot_root_states[env_ids, 9] = 0.0 + 0.5*torch.rand(len(env_ids), device=str(self.device)) # z position between 0.3 and 0.7


    
    def log_state(self,key, value):
        self.state_log[key].append(value)
    
    def log_states(self, dicy):
        for key, value in dicy.items():
            self.log_state(key, value)
    
    def save_state_log(self):
        for key, value in self.state_log.items():
            self.state_log[key] = np.array(value)
        data  = dict()
        print(data.keys())
        data.update(self.state_log)
        np.savez(self.data_path, **data)
        print(f"state log saved to {self.data_path}")


    def step(self, actions):
        
        if EVAL_after_real:
            
            # if self.episode_length_buf[0] == 1:
            #     import ipdb; ipdb.set_trace()
            self.log_states({
                    'base_pos': self._get_obs_base_pos().cpu().numpy(),
                    # 'base_pos_ref': self.x_ref_buffer[self.episode_length_buf].cpu().numpy(),
                    # 'phase': self.phase.cpu().numpy(),
                    'base_height': self.simulator.robot_root_states[:, 2].cpu().numpy(),
                    'base_vel': self.simulator.robot_root_states[:, 7:10].cpu().numpy(),
                    'contact_force': torch.norm(torch.norm(self.simulator.contact_forces[:, self.feet_indices,:],dim=2),dim=1).cpu().numpy(),
                    'feet_pos': self.simulator._rigid_body_pos[:, self.feet_indices, :].cpu().numpy(),
                    'dof_pos': self.simulator.dof_pos.cpu().numpy(),
                    'actions': self.last_actions.cpu().numpy(),
                    'dof_vel': self.simulator.dof_vel.cpu().numpy(),
                    'torques': self.torques.cpu().numpy(),
                    'quat': self.base_quat.cpu().numpy(),
                })
            
            # print(self.simulator.dof_pos)
            # exit()
            if self.phase >= 1*self.max_episode_length:
                self.save_state_log()
                exit()
        self.phase += 1
        return super().step(actions)   

   
           
    ##rewards for backflip##
    
    def _reward_orientation_control(self):
        current_time  = self.episode_length_buf*self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.5)
        quat_pitch = quat_from_angle_axis(4*torch.pi*phase, torch.tensor([0., 1., 0.], device=self.device))
        desired_base_quat = quat_mul(quat_pitch, self.base_init_state[3:7].reshape(1,-1).repeat(self.num_envs,1))
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)
        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity), dim=1)

        return orientation_diff
    
    def _reward_roll_orientation_control(self):
        current_time  = self.episode_length_buf*self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.5)
        quat_roll = quat_from_angle_axis(4*torch.pi*phase, torch.tensor([1.0, 0., 0.], device=self.device))
        desired_base_quat = quat_mul(quat_roll, self.base_init_state[3:7].reshape(1,-1).repeat(self.num_envs,1))
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)
        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity), dim=1)
        return orientation_diff


    def _update_reset_buf(self):
        if self.config.termination.terminate_by_contact:
            # print("self.termination_contact_indices", self.termination_contact_indices)
            # print("self.simulator.contact_forces[:, self.termination_contact_indices, :]", self.simulator.contact_forces[:, self.termination_contact_indices, :])
            # import ipdb; ipdb.set_trace()
            # print("feet contact forces", self.simulator.contact_forces[:, self.termination_contact_indices, :])
            self.reset_buf |= torch.any(torch.norm(self.simulator.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

            # the name of the contact indiecs can be found by self.simulator.dof_names[self.termination_contact_indices]
                
            # Step 1: Find which contact indices caused the reset condition
            # exceeding_contact_indices = self.termination_contact_indices[
            #     torch.any(torch.norm(self.simulator.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=0)
            # ]

            # # Step 2: Map these indices to their corresponding names
            # exceeding_contact_names = [self.simulator.body_names[idx] for idx in exceeding_contact_indices]

            # Print or log the names of the contact indices that caused the reset
            # import ipdb; ipdb.set_trace()
            # print("Contact indices causing reset:", exceeding_contact_names)
                 
        if self.config.termination.terminate_by_gravity:
            # print(self.projected_gravity)
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > self.config.termination_scales.termination_gravity_x, dim=1)
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > self.config.termination_scales.termination_gravity_y, dim=1)
        if self.config.termination.terminate_by_low_height:
            # import ipdb; ipdb.set_trace()
            self.reset_buf |= torch.any(self.simulator.robot_root_states[:, 2:3] < self.config.termination_scales.termination_min_base_height, dim=1)

        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.) # lower limit
            out_of_dof_pos_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.)
            
            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)
            # get random number between 0 and 1, if it is smaller than self.config.termination_probality.terminate_when_close_to_dof_pos_limit, apply the termination
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_pos_limit:
                self.reset_buf |= out_of_dof_pos_limits > 0.
        
        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.config.termination_scales.termination_close_to_dof_vel_limit).clip(min=0., max=1.), dim=1)
            
            

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_vel_limit:
                self.reset_buf |= out_of_dof_vel_limits > 0.
        
        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.termination_scales.termination_close_to_torque_limit).clip(min=0., max=1.), dim=1)
            
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_torque_limit:
                self.reset_buf |= out_of_torque_limits > 0.

    

    def _reward_ang_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = -self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2)
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)
    
    def _reward_ang_vel_x(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = self.base_ang_vel[:, 0].clamp(max=9.2, min=-9.2)
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_lin_vel_z(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.base_lin_vel[:, 2].clamp(max=3.5)
        return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.8) #- lin_vel * torch.logical_and(current_time > 1.5, current_time < 2.0)
    
    def _reward_lin_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.base_lin_vel[:, 1].clamp(max=3.5)
        return torch.abs(lin_vel) * (current_time<1.0)
    
    def _reward_reach_y_target(self):
        current_time = self.episode_length_buf * self.dt
        target_y = -1.5
        x_diff_end = torch.abs(self._get_obs_base_pos()[:, 1] - target_y)
        x_diff_start = torch.abs(self._get_obs_base_pos()[:, 1])
        # return x_diff * (current_time > 1.5)
        return torch.exp(-x_diff_end/self.config.rewards.reward_tracking_sigma.height) * (current_time > 1.4) + torch.exp(-x_diff_start/self.config.rewards.reward_tracking_sigma.height) * (current_time < 0.5)

    def _reward_reach_x_target(self):
        current_time = self.episode_length_buf * self.dt
        target_x = 0.0
        x_diff = torch.abs(self._get_obs_base_pos()[:, 0] - target_x)
        # return x_diff * (current_time > 1.5)
        return torch.exp(-x_diff/self.config.rewards.reward_tracking_sigma.height) 

    def _reward_height_control(self):
        # Penalize non flat base orientation
        current_time = self.episode_length_buf * self.dt
        target_height = 0.35
        height_diff = torch.square(target_height - self.simulator.robot_root_states[:, 2])# * torch.logical_or(current_time < 0.4, current_time > 1.4)
        return torch.exp(-height_diff/self.config.rewards.reward_tracking_sigma.height) * torch.logical_or(current_time < 0.4, current_time > 1.2)

    def _reward_actions_symmetry(self):
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        return actions_diff
    
    def _reward_gravity_y(self):
        return torch.square(self.projected_gravity[:, 1])
    
    def _reward_gravity_x(self):
        return torch.square(self.projected_gravity[:, 0])
    
    def _reward_feet_distance(self):
        current_time = self.episode_length_buf * self.dt
        feet_pos = self.simulator._rigid_body_pos[:, self.feet_indices, :]
        cur_footsteps_translated = feet_pos - self.simulator.robot_root_states[:, :3].unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply(quat_conjugate(self.base_quat),
                                                                 cur_footsteps_translated[:, i, :])

        stance_width = 0.3 * torch.ones([self.num_envs, 1,], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)
        
        return stance_diff*(current_time>1.4)
    
    def _reward_feet_distance_x(self):
        current_time = self.episode_length_buf * self.dt
        feet_pos = self.simulator._rigid_body_pos[:, self.feet_indices, :]
        cur_footsteps_translated = feet_pos - self.simulator.robot_root_states[:, :3].unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply(quat_conjugate(self.base_quat),
                                                                 cur_footsteps_translated[:, i, :])

        stance_width = 0.1923 * torch.ones([self.num_envs, 1,], device=self.device)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 0]).sum(dim=1)
        
        return stance_diff
    
    def _reward_penalty_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        current_time = self.episode_length_buf * self.dt
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)*torch.logical_or(current_time < 0.5, current_time > 1.4)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return (1.0 * (torch.norm(self.simulator.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1)).sum(dim=1)


    def _reward_feet_height_before_backflip(self):
        current_time = self.episode_length_buf * self.dt
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2].view(self.num_envs, -1) - 0.02
        return torch.exp(-feet_height.clamp(min=0).sum(dim=1)/self.config.rewards.reward_tracking_sigma.height) * torch.logical_or(current_time <0.5, current_time >1.4)

        return feet_height.clamp(min=0).sum(dim=1) * (current_time < 0.5)
    
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
        
        if self.config.domain_rand.randomize_motor_model:
            # randomize motor strength between range given by config
            A_hip = torch_rand_float(self.config.domain_rand.motor_model_range.hip[0], self.config.domain_rand.motor_model_range.hip[1], (self.num_envs, 1), device=self.device)
            A_thigh = torch_rand_float(self.config.domain_rand.motor_model_range.thigh[0], self.config.domain_rand.motor_model_range.thigh[1], (self.num_envs, 1), device=self.device)
            A_calf = torch_rand_float(self.config.domain_rand.motor_model_range.calf[0], self.config.domain_rand.motor_model_range.calf[1], (self.num_envs, 1), device=self.device)
            A = torch.cat([A_hip, A_thigh, A_calf, A_hip, A_thigh, A_calf, A_hip, A_thigh, A_calf, A_hip, A_thigh, A_calf], dim=1)
            # B = torch_rand_float(self.config.domain_rand.motor_model_range.B[0], self.config.domain_rand.motor_model_range.B[1], (self.num_envs, self.num_dofs), device=self.device)
            
            # A_hip = torch_rand_float_normal(self.config.domain_rand.motor_model_range.hip[0], self.config.domain_rand.motor_model_range.hip[1], (self.num_envs, 1), device=self.device)
            # A_thigh = torch_rand_float_normal(self.config.domain_rand.motor_model_range.thigh[0], self.config.domain_rand.motor_model_range.thigh[1], (self.num_envs, 1), device=self.device)
            # A_calf = torch_rand_float_normal(self.config.domain_rand.motor_model_range.calf[0], self.config.domain_rand.motor_model_range.calf[1], (self.num_envs, 1), device=self.device)
            # A = torch.cat([A_hip, A_thigh, A_calf, A_hip, A_thigh, A_calf, A_hip, A_thigh, A_calf, A_hip, A_thigh, A_calf], dim=1)

            torques = A*torch.tanh(1*torques/A)

        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques


    
    ###observation for backflip ###
    def _get_obs_clock(self):
        return self.clock()
    
    def _get_obs_command_phase_len(self):
        # self.phase_len = self.commands[:, 4]
        return self.phase_len.unsqueeze(1)

    def _get_obs_base_pos(self):
        init_pos = self.env_origins
        return self.simulator.robot_root_states[:, :3] - init_pos

    def _get_obs_base_quat(self):
        return self.base_quat
    
    def _get_obs_last_actions(self):
        return self.last_actions
    
    def _get_obs_phase(self):
        phase  = torch.pi * self.episode_length_buf[:,None] * self.dt/2
        return torch.cat([torch.sin(phase), torch.cos(phase),torch.sin(phase/2),torch.cos(phase/2),torch.sin(phase/4),torch.cos(phase/4)], dim =-1,)
        
    def _get_obs_base_height(self):
        return self.simulator.robot_root_states[:, 2].unsqueeze(1)