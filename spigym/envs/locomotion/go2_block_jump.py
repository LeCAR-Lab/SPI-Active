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
from spigym.envs.locomotion.go2_locomotion import go2_locomotion

# from spigym.envs.env_utils.command_generator import CommandGenerator




def torch_rand_choice(list_of_phase_lens,shape,device):
    values_tensor = torch.tensor(list_of_phase_lens, device=device,dtype=torch.float32)
    return values_tensor[torch.randint(0, len(list_of_phase_lens), shape)]

class go2_block_jump(go2_locomotion):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        
        # self.phase_len = self.config.rewards.phase_length*torch.ones_like(self.episode_length_buf.to(torch.float32))
        self.non_feet_indices = [i for i in torch.arange(self.simulator.contact_forces.shape[1]) if i not in self.feet_indices]
        self.reference_generator()
        self.init_dof_pos = torch.tensor([0.29609138,0.86028969,-1.6451211,-0.27495769,0.7985841,-1.69473946,0.27923194,0.99235493,-1.57032239,-0.29827648,0.99607897,-1.61637068],dtype=torch.float, device=self.device, requires_grad=False)
        self.init_dof_pos = self.init_dof_pos.unsqueeze(0)
        print("non feet indices", self.non_feet_indices)



    
    def gen_parabol(self,N,max_value=0.4):
        x = torch.arange(N)
        h = (N-1)/2
        a = max_value/(h**2)
        y = -a*(x-h)**2 + max_value
        return y
    
    def reference_generator(self):
        # x and z tensor same size as max episode length
        self.x_ref_buffer = torch.zeros(int(1.5*self.max_episode_length), device=self.device)
        self.z_ref_buffer = torch.zeros(int(1.5*self.max_episode_length), device=self.device)

        self.x_ref_buffer[:int(self.max_episode_length/4)] = 0.0
        self.x_ref_buffer[int(self.max_episode_length/4):int(1*self.max_episode_length/2)] = torch.linspace(0.0, self.config.rewards.x_target , int(1*self.max_episode_length/4)+1)
        self.x_ref_buffer[int(1*self.max_episode_length/2):] = self.config.rewards.x_target

        self.z_ref_buffer[:int(self.max_episode_length/4)] = 0.34
        self.z_ref_buffer[int(self.max_episode_length/4):int(3*self.max_episode_length/8)] = torch.linspace(0.34, self.config.rewards.z_target, int(1*self.max_episode_length/8)+1)
        self.z_ref_buffer[int(3*self.max_episode_length/8):int(1*self.max_episode_length/2)] = torch.linspace(self.config.rewards.z_target, 0.3, int(1*self.max_episode_length/8)+1)
        self.z_ref_buffer[int(1*self.max_episode_length/2):] = 0.34  
    
    
    
    def _reset_dofs(self, env_ids, target_state=None):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        If target_state is not None, reset to target_state

        Args:
            env_ids (List[int]): Environemnt ids
            target_state (Tensor): Target state
        """
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device))
            # self.simulator.dof_pos[env_ids] = self.init_dof_pos
            # import ipdb; ipdb.set_trace()
            
            self.simulator.dof_vel[env_ids] = 0.

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                     gymtorch.unwrap_tensor(self.simulator.dof_state),
        #                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _reset_root_states(self, env_ids, target_root_states=None):
        """ Resets ROOT states position and velocities of selected environmments
            if target_root_states is not None, reset to target_root_states
        Args:
            env_ids (List[int]): Environemnt ids
            target_root_states (Tensor): Target root states
        """
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]

        else:
            # base position
            if self.custom_origins:
                self.simulator.robot_root_states[env_ids] = self.base_init_state
                self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
                self.simulator.robot_root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=str(self.device)) # xy position within 1m of the center
            else:
                self.simulator.robot_root_states[env_ids] = self.base_init_state
                self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            # base velocities
            self.simulator.robot_root_states[env_ids, 2] = 0.3 + 0.1*torch.rand(len(env_ids), device=str(self.device)) # z position between 0.3 and 0.7
            # self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.3, 0.3, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel
            # self.simulator.robot_root_states[env_ids, 9] = 0.0 + 0.5*torch.rand(len(env_ids), device=str(self.device)) # z position between 0.3 and 0.7

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
            # torques = A*torques
            # A = torch_rand_float(self.config.domain_rand.motor_model_range[0], self.config.domain_rand.motor_model_range[1], (self.num_envs, 1), device=self.device)
            # A_all = torch.cat([A]*self.num_dofs, dim=1)
            # torques = A_all*torch.tanh(1*torques/A_all)


        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques


 
    def _update_reset_buf(self):
        super()._update_reset_buf()
        ## if the base height is away from the reference more than the threshold, reset the episode
        if self.config.termination.terminate_by_ref_error_threshold:
            
            base_height_error = torch.abs(self.simulator.robot_root_states[:, 2] - self.z_ref_buffer[self.episode_length_buf])
            base_x_error = torch.abs(self._get_obs_base_pos()[:,0] - self.x_ref_buffer[self.episode_length_buf])
            base_y_error = torch.abs(self._get_obs_base_pos()[:,1])
            # self.reset_buf|= base_height_error > self.config.termination_scales.base_height_error_threshold
            # self.reset_buf|= base_x_error > self.config.termination_scales.base_x_error_threshold
            # self.reset_buf|= base_y_error > self.config.termination_scales.base_y_error_threshold
            #check if any foot is having contact
            contact = torch.norm(torch.norm(self.simulator.contact_forces[:, self.feet_indices,:],dim=2),dim=1)>0
            # self.reset_buf|= contact!=self.contact_ref_buffer[self.phase]
        
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.config.terrain.mesh_type in ["heightfield", "trimesh"]:
            # import ipdb; ipdb.set_trace()
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.config.terrain.max_init_terrain_level
            if not self.config.terrain.curriculum: max_init_level = self.config.terrain.num_rows - 1
            # self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_levels = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.config.terrain.num_rows), rounding_mode='floor').to(torch.long)

            # self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.config.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.terrain_types = torch.arange(self.num_envs, device=self.device) % self.config.terrain.num_cols

            
            self.max_terrain_level = self.config.terrain.num_rows
            if isinstance(self.simulator.terrain.env_origins, np.ndarray):
                self.terrain_origins = torch.from_numpy(self.simulator.terrain.env_origins).to(self.device).to(torch.float)
            else:
                self.terrain_origins = self.simulator.terrain.env_origins.to(self.device).to(torch.float)   
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            # import ipdb; ipdb.set_trace()
            
            
            # exit()
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.config.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
   

    ###REWARDS
    
    def _reward_track_height_ref(self):
        height_error = torch.square(self.simulator.robot_root_states[:, 2] - self.z_ref_buffer[self.episode_length_buf])
        return torch.exp(-height_error/self.config.rewards.reward_tracking_sigma.height)
    
    def _reward_track_x_ref(self):
        x_ref_error = torch.square(self._get_obs_base_pos()[:,0] - self.x_ref_buffer[self.episode_length_buf])
        return torch.exp(-x_ref_error/self.config.rewards.reward_tracking_sigma.height)
    
    
        
    def _reward_enforce_same_foot_xy(self):
        
        base_rot = self.base_quat
        base_pos = self.simulator.robot_root_states[:, :3]
        foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices, :3]
        foot_pos = foot_pos - base_pos[:, None, :3]
        fl_pos, fr_pos, rl_pos, rr_pos = foot_pos[:, 0], foot_pos[:, 1], foot_pos[:, 2], foot_pos[:, 3]
        
        fl = quat_rotate_inverse(base_rot, fl_pos)
        fr = quat_rotate_inverse(base_rot, fr_pos)
        rl = quat_rotate_inverse(base_rot, rl_pos)
        rr = quat_rotate_inverse(base_rot, rr_pos)
        
        fl = fl[:, :2]
        fr_flip_y = torch.stack([fr[:, 0], -fr[:, 1]], dim=1)
        foot_f_xy_diff = torch.norm(fl - fr_flip_y, dim=1)
        rl = rl[:, :2]
        rr_flip_y = torch.stack([rr[:, 0], -rr[:, 1]], dim=1)
        foot_r_xy_diff = torch.norm(rl - rr_flip_y, dim=1)
        return foot_f_xy_diff + foot_r_xy_diff

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
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return (1.0 * (torch.norm(self.simulator.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1)).sum(dim=1)

    def _reward_lin_vel_z(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.base_lin_vel[:, 2].clamp(max=1.75)
        # return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75) #- lin_vel * torch.logical_and(current_time > 1.5, current_time < 2.0)
        if self.config.rewards.task == "yj":
            
            return lin_vel * torch.logical_and(current_time > 0.75, current_time < 1.3)  
        else: 
            return lin_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)
        
    def _reward_lin_vel_x(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.base_lin_vel[:, 0].clamp(max=0.9)
        return lin_vel * torch.logical_and(current_time > 0.75, current_time < 1.35) #- lin_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

    def _reward_lin_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = torch.abs(self.base_lin_vel[:, 1].clamp(max=3))
        return lin_vel 

    def _reward_height_control(self):
        # Penalize non flat base orientation
        current_time = self.episode_length_buf * self.dt
        target_height = 0.35
        height_diff = torch.square(target_height - self.simulator.robot_root_states[:, 2]) #* torch.logical_or(current_time < 0.75, current_time > 1.5)
        return height_diff*(current_time > 1.5)
    


    def _reward_actions_symmetry(self):
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        return torch.exp(-actions_diff/self.config.rewards.reward_tracking_sigma.height)

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
        
        return stance_diff
    
    def _reward_feet_x(self):
        current_time = self.episode_length_buf * self.dt
        feet_pos = self.simulator._rigid_body_pos[:, self.feet_indices, :]
        cur_footsteps_translated = feet_pos - self.simulator.robot_root_states[:, :3].unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply(quat_conjugate(self.base_quat),
                                                                 cur_footsteps_translated[:, i, :])

        stance_width = 0.3 * torch.ones([self.num_envs, 1,], device=self.device)
        default_stance_length = 0.1934*torch.ones([self.num_envs, 1,], device=self.device)
        extra_stance_length = self.config.rewards.extra_stance_length
        desired_ys = torch.cat([default_stance_length+extra_stance_length,default_stance_length+extra_stance_length,-default_stance_length+extra_stance_length,-default_stance_length+extra_stance_length], dim=1)
        stance_diff = torch.abs(desired_ys - footsteps_in_body_frame[:, :, 0]).sum(dim=1)
        
        return stance_diff#*torch.logical_and(current_time > 0.9, current_time < 1.25)
    
    
    def _reward_feet_height_before_jump(self):
        current_time = self.episode_length_buf * self.dt
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2].view(self.num_envs, -1) - 0.02
        return torch.exp(-feet_height.clamp(min=0).sum(dim=1)/self.config.rewards.reward_tracking_sigma.height) * torch.logical_or(current_time <0.75, current_time >1.25)
        return feet_height.clamp(min=0).sum(dim=1) * torch.logical_or(current_time <0.75, current_time >1.5)
    
    def _reward_reach_x_target(self):
        current_time = self.episode_length_buf * self.dt
        target_x = self.config.rewards.x_target
        x_diff = torch.abs(self._get_obs_base_pos()[:, 0] - target_x)
        # return x_diff * (current_time > 1.5)
        return torch.exp(-x_diff/self.config.rewards.reward_tracking_sigma.height) #* (current_time > 1.5)


    def _reward_reach_z_target(self):
        current_time = self.episode_length_buf * self.dt
        target_z = self.config.rewards.z_target
        z_diff = torch.abs(self.simulator.robot_root_states[:, 2] - target_z)
        return z_diff * torch.logical_and(current_time > 0.75, current_time < 1.3)
    
    def _reward_ang_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = -self.base_ang_vel[:, 1].clamp(max=3.0, min=-3.0)
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)
    
    
    ## penalty rewards 
    def _reward_penalty_orientation(self):
        current_time  = self.episode_length_buf*self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.5)
        quat_yaw = quat_from_angle_axis(0*phase, torch.tensor([0., 0., 1.], device=self.device))
        desired_base_quat = quat_mul(quat_yaw, self.base_init_state[3:7].reshape(1,-1).repeat(self.num_envs,1))
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)
        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity), dim=1)

        return orientation_diff#*(current_time > 1.0)
    
    def _reward_yaw_orientation_control(self):
        current_time  = self.episode_length_buf*self.dt
        phase = ((current_time - 0.75)*1.5/3.0).clamp(min=0, max=0.375)
        final_orientation  = 0.25*torch.ones_like(phase,device=self.device)
        quat_yaw = quat_from_angle_axis(2*torch.pi*phase, torch.tensor([0., 0., 1.0], device=self.device))
        desired_base_quat = quat_mul(quat_yaw, self.base_init_state[3:7].reshape(1,-1).repeat(self.num_envs,1))
        desired_projected_vector = quat_rotate_inverse(desired_base_quat, self.forward_vec)
        current_projected_vector = quat_rotate_inverse(self.base_quat, self.forward_vec)
        orientation_diff = torch.sum(torch.square(current_projected_vector-desired_projected_vector), dim=1)
        # print("proejcted gravity", current_projected_vector[0])
        # print('desired_projected_gravity', desired_projected_vector[0])
        # exit()
        return torch.exp(-orientation_diff/self.config.rewards.reward_tracking_sigma.height)
        
    
    def _reward_penalty_ang_vel_x(self):
        return torch.abs(self.base_ang_vel[:, 0])
    
    def _reward_ang_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = -self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2)
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

    def _reward_penalty_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        current_time = self.episode_length_buf * self.dt
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        # return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)*(current_time > 1.0)
        # return torch.sum(torch.norm(foot_vel, dim=-1) , dim=1)*(current_time > 1.2)

        return torch.sum(torch.norm(foot_vel, dim=-1) , dim=1)*torch.logical_or(current_time > 1.2, current_time<0.75)
    
    def _reward_penalty_contact_during_air(self):
        current_time = self.episode_length_buf * self.dt
        contact = torch.norm(torch.norm(self.simulator.contact_forces[:, self.feet_indices,:],dim=2),dim=1)>0
        return contact*torch.logical_and(current_time > 0.75, current_time < 1.4)

    def _reward_penalty_contact_landing(self):
        current_time = self.episode_length_buf * self.dt
        contact = torch.norm(torch.norm(self.simulator.contact_forces[:, self.feet_indices,:],dim=2),dim=1)
        return contact*torch.logical_and(current_time > 0.9, current_time < 1.25)
    
    def _reward_non_feet_contact(self):
        non_feet_contact = torch.sum(torch.sum(torch.square(self.simulator.contact_forces[:, self.non_feet_indices,:] > 1.0), dim=2),dim=1)
        return non_feet_contact

    ###observation for jump ###
    
    
    

    def _get_obs_base_pos(self):
        init_pos = self.env_origins
        return self.simulator.robot_root_states[:, :3] - init_pos

    def _get_obs_base_quat(self):
        return self.base_quat
    
    def _get_obs_last_actions(self):
        return self.last_actions
    
    def _get_obs_phase(self):
        phase  = 2*torch.pi * self.episode_length_buf[:,None]/self.max_episode_length
        # return torch.cat([torch.sin(phase), torch.cos(phase),torch.sin(phase/2),torch.cos(phase/2),torch.sin(phase/4),torch.cos(phase/4)], dim =-1,)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim =-1,)
    

    def _get_obs_base_height(self):
        return self.simulator.robot_root_states[:, 2].unsqueeze(1)