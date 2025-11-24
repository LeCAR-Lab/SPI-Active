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

def torch_rand_choice(list_of_phase_lens,shape,device):
    values_tensor = torch.tensor(list_of_phase_lens, device=device,dtype=torch.float32)
    return values_tensor[torch.randint(0, len(list_of_phase_lens), shape)]

class go2_single_jump(LeggedRobotLocomotion):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.init_done = True
        self.state_log = defaultdict(list)
        self.data_path = "spigym/data/performance_bags/single_jump.npz"
        # self.data_sysid = np.load(self.data_path, allow_pickle=True)
        self.phase = 0*self.episode_length_buf
        # self.phase_len = self.config.rewards.phase_length*torch.ones_like(self.episode_length_buf.to(torch.float32))
        self.non_feet_indices = [i for i in torch.arange(self.simulator.contact_forces.shape[1]) if i not in self.feet_indices]
        self.reference_generator()
        ##plot x_ref and z_ref
        import matplotlib.pyplot as plt
        
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
        self.contact_ref_buffer = torch.zeros(int(1.5*self.max_episode_length), device=self.device)
        self.x_ref_buffer[:int(self.max_episode_length/3)] = 0.0
        self.x_ref_buffer[int(self.max_episode_length/3):int(13*self.max_episode_length/24)] = torch.linspace(0.0, 0.75, int(13*self.max_episode_length/24)-int(self.max_episode_length/3))
        self.x_ref_buffer[int(13*self.max_episode_length/24):] = 0.75
        self.z_ref_buffer[:int(self.max_episode_length/6)] = 0.34
        self.z_ref_buffer[int(self.max_episode_length/6):int(self.max_episode_length/3)] = self.config.min_base_height
        self.z_ref_buffer[int(13*self.max_episode_length/24):] = self.config.nominal_base_height
        # self.z_ref_buffer[int(self.max_episode_length/3):int(19*self.max_episode_length/48)] = torch.linspace(self.config.min_base_height, self.config.max_base_height, int(19*self.max_episode_length/48)-int(self.max_episode_length/3))
        # self.z_ref_buffer[int(19*self.max_episode_length/48):int(13*self.max_episode_length/24)] = torch.linspace(self.config.max_base_height, self.config.min_base_height, int(13*self.max_episode_length/24)-int(19*self.max_episode_length/48))
        self.z_ref_buffer[int(self.max_episode_length/3):int(13*self.max_episode_length/24)] = self.config.min_base_height + self.gen_parabol(int(13*self.max_episode_length/24)-int(self.max_episode_length/3),self.config.max_base_height-self.config.min_base_height)
        self.contact_ref_buffer[:int(self.max_episode_length/3)] = 1.0
        self.contact_ref_buffer[int(self.max_episode_length/3):int(13*self.max_episode_length/24)] = 0.0
        self.contact_ref_buffer[int(13*self.max_episode_length/24):] = 1.0
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
                    'phase': self.phase.cpu().numpy(),
                    'base_height': self.simulator.robot_root_states[:, 2].cpu().numpy(),
                    'base_vel': self.simulator.robot_root_states[:, 7:10].cpu().numpy(),
                })
            
            if self.phase >= self.max_episode_length:
                self.save_state_log()
                exit()
        self.phase += 1
        return super().step(actions)    
            
    
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
            self.simulator.robot_root_states[env_ids, 2] = 0.34 + 0.18*torch.rand(len(env_ids), device=str(self.device)) # z position between 0.3 and 0.7
            self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel
            # self.simulator.robot_root_states[env_ids, 9] = 0.0 + 0.5*torch.rand(len(env_ids), device=str(self.device)) # z position between 0.3 and 0.7

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        self.phase[env_ids] = 0
        return super().reset_envs_idx(env_ids, target_states, target_buf)
    
    def _update_reset_buf(self):
        super()._update_reset_buf()
        ## if the base height is away from the reference more than the threshold, reset the episode
        if self.config.termination.terminate_by_ref_error_threshold:
            
            base_height_error = torch.abs(self.simulator.robot_root_states[:, 2] - self.z_ref_buffer[self.phase])
            base_x_error = torch.abs(self._get_obs_base_pos()[:,0] - self.x_ref_buffer[self.phase])
            base_y_error = torch.abs(self._get_obs_base_pos()[:,1])
            # self.reset_buf|= base_height_error > self.config.termination_scales.base_height_error_threshold
            # self.reset_buf|= base_x_error > self.config.termination_scales.base_x_error_threshold
            # self.reset_buf|= base_y_error > self.config.termination_scales.base_y_error_threshold
            #check if any foot is having contact
            contact = torch.norm(torch.norm(self.simulator.contact_forces[:, self.feet_indices,:],dim=2),dim=1)>0
            # self.reset_buf|= contact!=self.contact_ref_buffer[self.phase]
    def clock(self):
        return torch.vstack((torch.sin(2 * torch.pi *  self.phase / self.max_episode_length), torch.cos(2 * torch.pi *  self.phase / self.max_episode_length))).T
    
    def height_target(self):
        return self.config.rewards.base_height + 0.1 * (self.clock()[:,0]<0).float() + 0.14*(self.clock()[:,0]>0).float()

    def _reward_com_height_reach(self):
        height_error = abs(self.simulator.robot_root_states[:, 2]-self.config.rewards.base_height_target)
        return torch.exp(-height_error/self.config.rewards.reward_tracking_sigma.height)
    
    def _reward_track_height(self):
        height_error = torch.square(self.simulator.robot_root_states[:, 2] - self.z_ref_buffer[self.phase])
        return torch.exp(-height_error/self.config.rewards.reward_tracking_sigma.height)
    
    def _reward_track_x_ref(self):
        x_ref_error = torch.square(self._get_obs_base_pos()[:,0] - self.x_ref_buffer[self.phase])
        return torch.exp(-x_ref_error/self.config.rewards.reward_tracking_sigma.height)

    def _reward_penalty_contact_consistency(self):
        contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices,:],dim=2)>0
        contact_target = (torch.floor(self.episode_length_buf/50)%2 ==0).unsqueeze(1)
        return torch.sum(torch.square(contact.float() - contact_target.float()),dim=1)
    
    def _reward_custom_feet_air_time(self):
            # Reward long steps
            # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
            contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
            contact_filt = torch.logical_or(contact, self.last_contacts) 
            self.last_contacts = contact
            first_contact = (self.feet_air_time > 0.) * contact_filt
            self.feet_air_time += self.dt
            rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
            self.feet_air_time *= ~contact_filt
            return rew_airTime

    def _reward_penalty_enforce_same_foot_z(self):
        foot_z_var = torch.var(self.simulator._rigid_body_pos[:,self.feet_indices, 2],dim=1)
        return foot_z_var

    def _reward_track_feet_height(self):
        height_error = torch.sum(torch.square(self.config.rewards.feet_height_target - self.simulator._rigid_body_pos[:,self.feet_indices, 2]),dim=1) 
        return torch.exp(-height_error/self.config.rewards.reward_tracking_sigma.height) * ((torch.floor(self.episode_length_buf/50)%2 ==1).float())
    
    def _reward_penalty_non_feet_contact(self):
        non_feet_contact = torch.sum(torch.sum(torch.square(self.simulator.contact_forces[:, self.non_feet_indices,:]), dim=2),dim=1)
        return non_feet_contact
    
    def _reward_penalty_contact_for_air(self):
        #penalise foot contact when clock is positive
        contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices,:],dim=2)>0
        return torch.exp(-torch.sum(contact.float(),dim=1)/self.config.rewards.reward_tracking_sigma.height)* (self.clock()[:,0]>0).float()
    
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

    ###observation for jump ###
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