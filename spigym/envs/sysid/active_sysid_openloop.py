import torch
import numpy as np
from pathlib import Path
import os
from spigym.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from spigym.envs.locomotion.go2_omni import go2_omni_interface
from omegaconf import OmegaConf
from spigym.utils.torch_utils import torch_rand_float
# from isaacgym import gymtorch, gymapi, gymutil


class ActiveSysId_OpenLoop(go2_omni_interface):
    """Open-loop Active SysID environment.

    This environment expands each "main" environment into (1 main + N aux)
    paired envs, where each aux env perturbs one parameter by `delta_param`.
    The identification reward (e.g., FIM) is computed using differences
    between the main and aux states.

    Configuration knobs (see `config/env/active_sysid_openloop.yaml`):
    - `exploration_params`: which physical or motor model parameters to vary.
    - `delta_param`: finite-difference delta applied to aux envs.
    - `ksync_steps`: how often to force aux states to match the main (state sync).
    - `motor_model`: actuator mapping used to translate actions to torques.

    The `params_dict` is built from `default_param` and `exploration_params`
    and injected into the simulator to modify body properties for each env.
    """
    def __init__(self, config, device):
        '''
        params: {'rbd_params': {'body_name': {'param_name': param_value}}, 'motor_params': { motor_model_name: {'param_name': param_value}}, 'default_motor_params': {motor_model_name :{'param_name': param_value}}}
        '''
        
        # LeggedRobotBase init
        self.init_done = False
        
        self.config = config
        
        self.param_dim = len(self.config.exploration_params)
        self.main_commands = None
        self.expanded_main_commands = None
        
        self.delta_param = self.config.delta_param
        # self.resampling_horizon = self.config.horizon_length
        # k-step synchronization for aux->main state reset (1 = current behavior)
        self.ksync_steps = int(getattr(self.config, "ksync_steps", 1))
        self.motor_model = self.config.motor_model
        
        
        
        self.config.num_envs = self.config.num_envs*(self.param_dim+1)

        # Build per-env parameter dictionary for the simulator
        self.params_dict = {}
        for key, value in self.config.default_param.items():
            if 'motor_model' in key:
                self.params_dict[key] = {'value':[value[0]]*self.config.num_envs}
            else:
                self.params_dict[key] = {'body_name': value[0], 'value':[value[1]]*self.config.num_envs}
        
         
        # Assign finite-difference deltas to auxiliary envs (round-robin per param)
        indices = np.arange(self.config.num_envs)%(self.param_dim+1)
        for i, param in enumerate(self.config.exploration_params):
            # Convert to array, modify, convert back
            values = np.array(self.params_dict[param]['value'])
            values[indices == i + 1] += self.delta_param
            self.params_dict[param]['value'] = values.tolist()
        
        additional_config = OmegaConf.create({"params_dict": self.params_dict})
        self.config = OmegaConf.merge(self.config, additional_config)

        super().__init__(self.config, device)
             
        self._init_sysid_logger() # replay
        
        self.num_main_envs = int(self.num_envs) // (self.param_dim + 1)
        self.aux_main_env_mapping = torch.arange(0, self.num_envs, self.param_dim + 1, device=self.device).repeat(self.param_dim + 1, 1).T.flatten()
        self.aux_idx = torch.arange(0, self.num_envs, device=self.device).view(self.num_main_envs, self.param_dim + 1)
        self.main_idx, self.aux_idx = self.aux_idx[:, 0:1], self.aux_idx[:, 1:]
                
        
    
        

    
    def destroy(self):
        self.simulator.gym.destroy_sim(self.simulator.sim)
           
    

    def _init_sysid_logger(self):
        self.log_dict['root_rot'] = torch.zeros_like(self.simulator.robot_root_states[:, 3:7])
        self.log_dict['root_trans_offset'] = torch.zeros_like(self.simulator.robot_root_states[:, :3])
        self.log_dict['root_lin_vel'] = torch.zeros_like(self.simulator.robot_root_states[:, 7:10])
        self.log_dict['root_ang_vel'] = torch.zeros_like(self.simulator.robot_root_states[:, 10:13])
        self.log_dict['dof'] = torch.zeros_like(self.simulator.dof_pos)
        self.log_dict['dof_vel'] = torch.zeros_like(self.simulator.dof_vel)
        self.log_dict['action'] = torch.zeros_like(self.simulator.dof_pos)
        self.log_dict['torque'] = torch.zeros_like(self.simulator.dof_pos)
        self.log_dict['ctrl'] = torch.zeros_like(self.simulator.dof_pos)
        self.ctrls = torch.zeros_like(self.torques)
        
        self.log_dict['root_rot_ref'] = torch.zeros_like(self.simulator.robot_root_states[:, 3:7])
        self.log_dict['root_trans_offset_ref'] = torch.zeros_like(self.simulator.robot_root_states[:, :3])
        self.log_dict['root_lin_vel_ref'] = torch.zeros_like(self.simulator.robot_root_states[:, 7:10])
        self.log_dict['root_ang_vel_ref'] = torch.zeros_like(self.simulator.robot_root_states[:, 10:13])
        self.log_dict['dof_ref'] = torch.zeros_like(self.simulator.dof_pos)
        self.log_dict['dof_vel_ref'] = torch.zeros_like(self.simulator.dof_vel)
        self.log_dict['tau_est'] = torch.zeros_like(self.torques)
                
    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

    def reset_all(self, main_commands=None):
        """Reset envs and optionally feed a precomputed command trajectory.

        The command trajectory is expanded to match the (main + aux) layout by
        repeating the main trajectory across all paired envs.
        """
        self.step_idx = 0
        
        if main_commands is not None:
            self.expanded_main_commands = main_commands.repeat_interleave(self.param_dim+1, dim=0)
            self.commands = self.expanded_main_commands[:,0,:]
        else:
            self.commands = torch.zeros(self.num_envs,14, device=self.device, dtype=torch.float32)
            
        
        return super().reset_all()

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
            target_states (dict): Dictionary containing lists of target states for the robot
        """
        if len(env_ids) == 0:
            return
        ## only reset the envs at the first step, so that terminated envs stay terminated
        if self.step_idx > 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.config.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)        # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf

   
    
    def _reset_tasks_callback(self, env_ids):
        pass

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self.step_idx += 1


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        nominal_torques = super()._compute_torques(actions)
        torques = getattr(self, self.motor_model)(nominal_torques)
        self.ctrls = torques
        return torques   


    ##todo method to update the main commands, accesible from algo class 
    def update_main_commands(self, main_commands):
        """Update the main command trajectory during an active rollout."""
        self.expanded_main_commands = main_commands.repeat_interleave(self.param_dim+1, dim=0)
        self.commands = self.expanded_main_commands[:,0,:]

    def _update_tasks_callback(self):
        """Advance command pointer each sim step (open-loop playback)."""

       
        
        self.commands = self.expanded_main_commands[:,self.step_idx,:]

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        self.episode_length_buf += 1
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        # compute observations, rewards, resets, ...
        self._check_termination()
        self._compute_reward()
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # self.reset_envs_idx(env_ids)

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
            self.need_to_refresh_envs[refresh_env_ids] = False

        self._compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self._post_compute_observations_callback()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            if self.debug_viz:
                self._draw_debug_vis()
        self._log_sysid_states()
        # Reset aux envs to main envs every k steps (default 1 = every step)
        if self.ksync_steps <= 1:
            self._reset_envs_states_to_main_envs()
        else:
            # step_idx is incremented in _pre_physics_step and starts at 1 after first step
            if (self.step_idx % self.ksync_steps) == 1:
                self._reset_envs_states_to_main_envs()

   
    
    


    def _update_reset_buf(self):
        """Ensure resets occur in (main + aux) groups.

        If either the main or any of the aux envs terminates, reset the whole
        group so the finite-difference structure remains valid.
        """
        super()._update_reset_buf()
        main_idx = self.main_idx
        aux_idx = self.aux_idx

        reset_groups = self.reset_buf[main_idx] | torch.any(self.reset_buf[aux_idx], dim=1,keepdim=True)
        
        self.reset_buf[main_idx] = reset_groups
        self.reset_buf[aux_idx] = reset_groups

    def _reset_dofs(self, env_ids, target_states=None):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if target_states is not None:
            self.simulator.dof_pos[env_ids] = target_states.view(self.num_envs, -1, 2)[..., 0]
            self.simulator.dof_vel[env_ids] = target_states.view(self.num_envs, -1, 2)[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device))
            # self.simulator.dof_pos[env_ids] = self.default_dof_pos
            # import ipdb; ipdb.set_trace()
            
            self.simulator.dof_vel[env_ids] = 0.


    # def _reset_root_states(self, env_ids, target_root_states=None):
    #     """ Resets ROOT states position and velocities of selected environmments
    #         if target_root_states is not None, reset to target_root_states
    #     Args:
    #         env_ids (List[int]): Environemnt ids
    #         target_root_states (Tensor): Target root states
    #     """
    #     if target_root_states is not None:
    #         self.simulator.robot_root_states[env_ids] = target_root_states
    #         # self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]

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
            
    #         self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel
    def _reset_envs_states_to_main_envs(self):
        """Synchronize aux env states to their corresponding main env.

        This keeps the finite-difference comparisons meaningful across time by
        preventing divergence of aux envs from the main due to stochasticity.
        """
        # reset root states
        self.dof_aux_main_env_mapping = ((self.aux_main_env_mapping*self.num_dof).unsqueeze(1)+ torch.arange(self.num_dofs,device=self.device)).flatten()
        tar_dof_states = self.simulator.dof_state[self.dof_aux_main_env_mapping]
        
        tar_root_states = self.simulator.all_root_states[self.aux_main_env_mapping]
        
        env_ids = torch.arange(self.num_envs).to(self.device)
        
        self._reset_robot_states_callback(env_ids, {"root_states": tar_root_states, "dof_states": tar_dof_states})

        # reset dof states

    def _log_sysid_states(self):
        """Collect states and torques needed by the identification objective."""
        self.log_dict['root_rot'] = self.simulator.robot_root_states[:, 3:7]
        self.log_dict['root_trans_offset'] = self.simulator.robot_root_states[:, :3] - self.env_origins
        self.log_dict['root_lin_vel'] = self.simulator.robot_root_states[:, 7:10]
        self.log_dict['root_ang_vel'] = self.simulator.robot_root_states[:, 10:13]
        self.log_dict['dof'] = self.simulator.dof_pos
        self.log_dict['dof_vel'] = self.simulator.dof_vel
        self.log_dict['torque'] = self.torques
        self.log_dict['ctrl'] = self.ctrls
        
        
        
        # self.log_dict['rew_joint_pos_tracking'] = self._reward_joint_pos_tracking()
        # self.log_dict['rew_joint_vel_tracking'] = self._reward_joint_vel_tracking()
        # self.log_dict['rew_ctrl_tracking'] = self._reward_ctrl_tracking()
        # self.log_dict['rew_base_pos_tracking'] = self._reward_base_pos_tracking()
        # self.log_dict['rew_base_rot_tracking'] = self._reward_base_rot_tracking()
        # self.log_dict['rew_base_lin_vel_tracking'] = self._reward_base_lin_vel_tracking()
        # self.log_dict['rew_base_ang_vel_tracking'] = self._reward_base_ang_vel_tracking()
        self.log_dict['rew_fisher_information_matrix'] = self._reward_fisher_information_matrix()
        
    def act2tau_scalar(self, nominal_torques, **kwargs):
        # print(f"act2tau_scalar: {kwargs.keys()}")
        gain = kwargs["scalar_gain"]
        gain = torch.tensor(gain, dtype=torch.float32).to(self.device).reshape(-1, 1)
        
        # print(f"tau: {tau.dtype}, ctrl: {ctrl.dtype}, gain: {gain.dtype}")
        tau = nominal_torques * gain
        return tau
    
    def act2tau_vec3(self, nominal_torques, **kwargs):
        tau = nominal_torques
        hip_gain = kwargs["hip_gain"]
        thigh_gain = kwargs["thigh_gain"]
        calf_gain = kwargs["calf_gain"]
        hip_gain = torch.tensor(hip_gain, dtype=torch.float32).to(self.device).reshape(-1, 1)
        thigh_gain = torch.tensor(thigh_gain, dtype=torch.float32).to(self.device).reshape(-1, 1)
        calf_gain = torch.tensor(calf_gain, dtype=torch.float32).to(self.device).reshape(-1, 1)
        hip_idx = [0, 3, 6, 9]
        thigh_idx = [1, 4, 7, 10]
        calf_idx = [2, 5, 8, 11]
        tau[:, hip_idx] = tau[:, hip_idx] * hip_gain
        tau[:, thigh_idx] = tau[:, thigh_idx] * thigh_gain
        tau[:, calf_idx] = tau[:, calf_idx] * calf_gain
        return tau
    
    def act2tau_vec3_tanh(self, nominal_torques):
        # y = a tanh (bx) 
        # default b = 1/max_tau, a = max_tau
        tau = nominal_torques
        hip_a = np.array(self.params_dict["motor_model_hip_a"]["value"])
        hip_b = 1/hip_a
        thigh_a = np.array(self.params_dict["motor_model_thigh_a"]["value"])
        thigh_b = 1/thigh_a
        calf_a = np.array(self.params_dict["motor_model_calf_a"]["value"])
        calf_b = 1/calf_a
        hip_a, hip_b = torch.tensor(hip_a, dtype=torch.float32).to(self.device).reshape(-1, 1), torch.tensor(hip_b, dtype=torch.float32).to(self.device).reshape(-1, 1)
        thigh_a, thigh_b = torch.tensor(thigh_a, dtype=torch.float32).to(self.device).reshape(-1, 1), torch.tensor(thigh_b, dtype=torch.float32).to(self.device).reshape(-1, 1)
        calf_a, calf_b = torch.tensor(calf_a, dtype=torch.float32).to(self.device).reshape(-1, 1), torch.tensor(calf_b, dtype=torch.float32).to(self.device).reshape(-1, 1)
        hip_idx = [0, 3, 6, 9]
        thigh_idx = [1, 4, 7, 10]
        calf_idx = [2, 5, 8, 11]
        tau[:, hip_idx] = hip_a * torch.tanh(hip_b * tau[:, hip_idx])
        tau[:, thigh_idx] = thigh_a * torch.tanh(thigh_b * tau[:, thigh_idx])
        tau[:, calf_idx] = calf_a * torch.tanh(calf_b * tau[:, calf_idx])
        return tau

    def _reward_fisher_information_matrix(self):
        """Finite-difference Fisher Information proxy over root and DOF states."""
        # Use origin-compensated positions to avoid per-env offsets leaking into gradients
        root_state_main = self.simulator.robot_root_states[self.main_idx].clone()
        root_state_aux = self.simulator.robot_root_states[self.aux_idx].clone()
        # compensate only position terms by subtracting per-env origins
        root_state_main[..., :3] = root_state_main[..., :3] - self.env_origins[self.main_idx]
        root_state_aux[..., :3] = root_state_aux[..., :3] - self.env_origins[self.aux_idx]
        root_state_diff = (root_state_main - root_state_aux) / self.delta_param # constant d_theta
        # fisher_info_root = torch.norm(root_state_diff, dim=[-1, -2]) ** 2
        # print(f"fisher_info_root: {fisher_info_root.shape}")
        
        dof_pos_main = self.simulator.dof_pos[self.main_idx]
        dof_pos_aux = self.simulator.dof_pos[self.aux_idx]
        dof_pos_diff = (dof_pos_main - dof_pos_aux) / self.delta_param # constant d_theta
        # fisher_info_dof = torch.norm(dof_pos_diff, dim=[-1, -2]) ** 2
        # print(f"fisher_info_dof: {fisher_info_dof.shape}")
        gradient = torch.cat([root_state_diff, dof_pos_diff], dim=2)
        XtX = torch.bmm(gradient.transpose(1, 2), gradient)
        fisher_info = torch.diagonal(XtX, dim1=1, dim2=2).sum(dim=1)
        
        # print(f"rew: {rew.shape}")
        
        rew = fisher_info.repeat_interleave(self.param_dim+1, dim=0)
        return rew
    
   