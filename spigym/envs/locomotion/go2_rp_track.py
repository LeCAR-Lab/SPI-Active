from spigym.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from spigym.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from spigym.envs.locomotion.go2_locomotion import go2_locomotion
# from spigym.envs.env_utils.command_generator import CommandGenerator

class go2_rp_track(go2_locomotion):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.freq = 1.0
        self.amplitude = 0.4
        self.period = (1/self.freq)*50

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges

    def ori_ref(self, epi_len):
        t = epi_len%self.period
        slope = 4*self.amplitude/self.period

        if t < self.period/4:
            return slope*t
        elif t < 3*self.period/4:
            return 2*self.amplitude - slope*t
        else:
            return slope*t - 4*self.amplitude



    def step(self, actions):
        return super().step(actions)

    def _setup_simulator_control(self):
        self.simulator.commands = self.commands

    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        # super().super()._update_tasks_callback()

        # commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["roll"][0], self.command_ranges["roll"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["pitch"][0], self.command_ranges["pitch"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)

        # set small commands to zero



    

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.commands = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :2] = torch.tensor(command).to(self.device)  # only set the first 2 commands

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
            
            torques = A*torch.tanh(1*torques/A)

        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques

    ########################### TRACKING REWARDS ###########################

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.commands[:, :2]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.gravity_vec)

        return torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)/self.config.rewards.reward_tracking_sigma.height)

    def _reward_base_height(self):
        # Penalize base height away from target

        base_height = self.simulator.robot_root_states[:, 2]
        return torch.exp(-torch.square(base_height - self.config.rewards.desired_base_height)/self.config.rewards.reward_tracking_sigma.height)
    ########################### PENALTY REWARDS ###########################

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return (1.0 * (torch.norm(self.simulator.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1)).sum(dim=1)
    
   
    ########################### FEET REWARDS ###########################
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
    
    def _reward_penalty_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        # return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)*(current_time > 1.0)
        return torch.sum(torch.norm(foot_vel, dim=-1) , dim=1)
    
    ######################### Observations #########################
    def _get_obs_command_rp_ori(self):
        
        return self.commands[:,:2]
    
    def _get_obs_base_pos(self):
        init_pos = self.env_origins
        return self.simulator.robot_root_states[:, :3] - init_pos
