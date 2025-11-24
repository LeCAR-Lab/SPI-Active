import torch
import numpy as np
from pathlib import Path
import os
from spigym.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw
)
from spigym.envs.env_utils.history_handler import HistoryHandler
from spigym.simulator.isaacgym.isaacgym_sysid import IsaacGymSysId
from hydra.utils import instantiate, get_class
# from isaacgym import gymtorch, gymapi, gymutil

from termcolor import colored
from loguru import logger

from scipy.spatial.transform import Rotation as sRot



class FIMCalc(LeggedRobotBase):
    def __init__(self, config, device, motion_data, **kwargs):
        
        # LeggedRobotBase init
        
        self.init_done = False
        
        
        # BaseTask init
        
        self.config = config
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # self.simulator = instantiate(config=self.config.simulator, device=device)
        SimulatorClass = get_class(self.config.simulator._target_)
        self.simulator: IsaacGymSysId = SimulatorClass(config=self.config, device=device, **kwargs)
        
        self.headless = config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.device = self.simulator.sim_device
        self.sim_dt = self.simulator.sim_dt
        self.up_axis_idx = 2 # Jiawei: HARD CODE FOR NOW

        self.dt = self.config.simulator.config.sim.control_decimation * self.sim_dt
        self.max_episode_length_s = self.config.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.num_envs = self.config.num_envs
        self.dim_obs = self.config.robot.policy_obs_dim
        self.dim_critic_obs = self.config.robot.critic_obs_dim
        self.dim_actions = self.config.robot.actions_dim

        terrain_mesh_type = self.config.terrain.mesh_type
        self.simulator.setup_terrain(terrain_mesh_type)

        # create envs, sim and viewer
        self._load_assets()
        self._get_env_origins()
        self._create_envs()
        self.dof_pos_limits, self.dof_vel_limits, self.torque_limits = self.simulator.get_dof_limits_properties()
        self._setup_robot_body_indices()
        # self._create_sim()
        self.simulator.prepare_sim()
        # if running with a viewer, set up keyboard shortcuts and camera
        self.viewer = None
        if self.headless == False:
            self.debug_viz = False
            self.simulator.setup_viewer()
            ###########################################################################
            # Jiawei: Should be removed
            ###########################################################################
            self.viewer = self.simulator.viewer
        self._init_buffers()

        ###########################################################################
        #### Jiawei: Should be removed
        ###########################################################################
        # self.gym = self.simulator.gym
        # self.sim = self.simulator.sim
        if self.headless == False:
            self.viewer = self.simulator.viewer
        
        
        
        self._domain_rand_config()
        self._prepare_reward_function()
        self.history_handler = HistoryHandler(self.num_envs, config.obs.obs_auxiliary, config.obs.obs_dims, device)
        self.is_evaluating = False
        self.init_done = True   
        
             
        # SysId_OpenLoop init
        
        self._init_motion_lib(motion_data)
        
        self._init_sysid_logger() # replay
        
        self.default_param = {}
        if self.simulator.defualt_base_mass is not None:
            self.default_param['base_mass'] = self.simulator.defualt_base_mass
        if self.simulator.defualt_base_xcom is not None:
            self.default_param['base_xcom'] = self.simulator.defualt_base_xcom
        if self.simulator.defualt_base_ycom is not None:
            self.default_param['base_ycom'] = self.simulator.defualt_base_ycom
        if self.simulator.defualt_base_zcom is not None:
            self.default_param['base_zcom'] = self.simulator.defualt_base_zcom
        if self.simulator.defualt_base_xinertia is not None:
            self.default_param['base_xinertia'] = self.simulator.defualt_base_xinertia
        if self.simulator.defualt_base_yinertia is not None:
            self.default_param['base_yinertia'] = self.simulator.defualt_base_yinertia
        if self.simulator.defualt_base_zinertia is not None:
            self.default_param['base_zinertia'] = self.simulator.defualt_base_zinertia



        self.sysid_kwargs = kwargs
        self.motor_model = self.sysid_kwargs['motor_model']
        print(f"motor model: {self.motor_model}")


        for key, value in self.sysid_kwargs.items():
            if '_default' in key:
                self.default_param[key.replace('_default', '')] = value
        
        self.init_done = True
        self.debug_viz = True
        
        
        # env indexing for FIM
        self.num_of_params = 2
        self.num_of_envs = 10
        self.total_envs = self.num_of_envs * (self.num_of_params + 1)
        
        
        self.aux_main_env_mapping = torch.arange(0, self.total_envs, self.num_of_params + 1, device=self.device).repeat(self.num_of_params + 1, 1).T.flatten()
        
        self.aux_idx = torch.arange(0, self.total_envs, device=self.device).view(self.num_of_envs, self.num_of_params + 1)
        self.main_idx, self.aux_idx = self.aux_idx[:, 0:1], self.aux_idx[:, 1:]
        
        
        
        
        
    
    def destroy(self):
        self.simulator.gym.destroy_sim(self.simulator.sim)
       
    
    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        # print('reset caused by timeout')
    

    def _init_motion_lib(self, motion_lib):
        self.total_steps = 0
        self.motion_data = motion_lib
        self.motion_data = list(self._gen_motion())
        self.traj_idx = 0
        self.ref_traj = self.motion_data[self.traj_idx]
        self.step_idx = 0
        
        for motion in self.motion_data:
            self.total_steps += len(motion['action'])
            
        
    
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
                

    def _gen_motion(self):
        for motion_name, motion in self.motion_data.items():
            for clip in motion:
                yield clip


    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()


    def _reset_root_states(self, env_ids, target_states=None):
        
        if target_states is not None:
            self.simulator.robot_root_states[env_ids] = target_states
        else:
            offset = self.env_origins

            root_rot = self.ref_traj['root_rot'][self.step_idx]
            root_pos = self.ref_traj['root_trans_offset'][self.step_idx]
            root_lin_vel = self.ref_traj['root_lin_vel'][self.step_idx]
            root_ang_vel = self.ref_traj['root_ang_vel'][self.step_idx]
            self.simulator.robot_root_states[env_ids, :3] = offset[env_ids] + root_pos - 0.
            root_rot = wxyz_to_xyzw(root_rot)
            self.simulator.robot_root_states[env_ids, 3:7] = root_rot
            self.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
            self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel


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
            offset = self.env_origins
            
            dof_pos = self.ref_traj['dof'][self.step_idx]
            dof_vel = self.ref_traj['dof_vel'][self.step_idx]
            # root_pos = self.ref_traj['root_pos'][self.step_idx]
            
            offset = self.env_origins
            
            self.simulator.dof_pos[env_ids] = dof_pos
            self.simulator.dof_vel[env_ids] = dof_vel

    
    
    def reset_all(self):
        self.step_idx = 0
        self.traj_idx = 0
        self.ref_traj = self.motion_data[self.traj_idx]
        
        env_ids = torch.arange(self.num_envs).to(self.device)
        self._reset_root_states(env_ids)
        self._reset_dofs(env_ids)
        self.reset_buf[env_ids] = 0
        self._refresh_sim_tensors()
        return super().reset_all()
        

    def act2tau(self, act, dof_pos, dof_vel,**kwargs):
        dof_pos_tar, dof_vel_tar, tau_tar = act[:, 0], act[:, 1], act[:, 2]
        kp, kd = act[:, 3], act[:, 4]
        tau = tau_tar + kp * (dof_pos_tar - dof_pos) + kd * (dof_vel_tar - dof_vel) 
        # tau = tau_tar + kp * (qpos_tar - self.data.sensordata[:12]) + kd * (qvel_tar - self.data.sensordata[12:24])
        tau = torch.clip(tau, -self.torque_limits, self.torque_limits)
        # print('dof_pos_err', dof_pos_tar - dof_pos)
        # print('dof_vel_err', dof_vel_tar - dof_vel)
        # print('tau_tar', tau_tar)
        ctrl = tau
        return tau, ctrl
    
    
    def act2tau_scalar(self, act, qpos, qvel, **kwargs):
        # print(f"act2tau_scalar: {kwargs.keys()}")
        gain = kwargs["scalar_gain"]
        gain = torch.tensor(gain, dtype=torch.float32).to(self.device).reshape(-1, 1)
        tau, ctrl = self.act2tau(act, qpos, qvel, **kwargs)
        # print(f"tau: {tau.dtype}, ctrl: {ctrl.dtype}, gain: {gain.dtype}")
        tau = tau * gain
        return tau, ctrl
    
    def act2tau_vec3(self, act, qpos, qvel, **kwargs):
        tau, ctrl = self.act2tau(act, qpos, qvel, **kwargs)
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
        return tau, ctrl
    
    def act2tau_vec3_tanh(self, act, qpos, qvel, **kwargs):
        # y = a tanh (bx) 
        # default b = 1/max_tau, a = max_tau
        tau, ctrl = self.act2tau(act, qpos, qvel, **kwargs)
        hip_a = kwargs["hip_a"]
        hip_b = kwargs["hip_b"]
        thigh_a = kwargs["thigh_a"]
        thigh_b = kwargs["thigh_b"]
        calf_a = kwargs["calf_a"]
        calf_b = kwargs["calf_b"]
        hip_a, hip_b = torch.tensor(hip_a, dtype=torch.float32).to(self.device).reshape(-1, 1), torch.tensor(hip_b, dtype=torch.float32).to(self.device).reshape(-1, 1)
        thigh_a, thigh_b = torch.tensor(thigh_a, dtype=torch.float32).to(self.device).reshape(-1, 1), torch.tensor(thigh_b, dtype=torch.float32).to(self.device).reshape(-1, 1)
        calf_a, calf_b = torch.tensor(calf_a, dtype=torch.float32).to(self.device).reshape(-1, 1), torch.tensor(calf_b, dtype=torch.float32).to(self.device).reshape(-1, 1)
        hip_idx = [0, 3, 6, 9]
        thigh_idx = [1, 4, 7, 10]
        calf_idx = [2, 5, 8, 11]
        tau[hip_idx] = hip_a * torch.tanh(hip_b * tau[hip_idx])
        tau[thigh_idx] = thigh_a * torch.tanh(thigh_b * tau[thigh_idx])
        tau[calf_idx] = calf_a * torch.tanh(calf_b * tau[calf_idx])
        return tau, ctrl
        

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
        act = self.ref_traj['action'][self.step_idx]
        dof_pos = self.simulator.dof_pos
        dof_vel = self.simulator.dof_vel
        # torques, ctrl = self.act2tau(act, dof_pos, dof_vel)
        torques, ctrl = getattr(self, self.motor_model)(act, dof_pos, dof_vel, **self.sysid_kwargs)
        self.ctrls = ctrl
        return torques   

    def _post_physics_step(self):
        super()._post_physics_step()
        self._log_sysid_states() # reward is computed here
        self._reset_envs_states_to_main_envs()
        
        # print(f"step {self.step_idx} of {self.ref_traj['root_rot'].shape[0]}")
        
        if self.step_idx >= self.ref_traj['action'].shape[0] - 1:
            
            if self.traj_idx >= len(self.motion_data) - 1:
                self.reset_buf.fill_(1)
                # logger.info("finished all trajectories")
            else:
                self.traj_idx += 1
                self.step_idx = 0
                self.ref_traj = self.motion_data[self.traj_idx]
                refresh_env_ids = torch.arange(self.num_envs).to(self.device)
                self._reset_robot_states_callback(refresh_env_ids)
                self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
                self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
                self.reset_buf.fill_(0)
                # logger.info(f"resetting to {self.traj_idx}, total traj: {len(self.motion_data)}")
                
                
    def _reset_envs_states_to_main_envs(self):
        # reset root states
        
        tar_dof_states = self.simulator.dof_state[self.aux_main_env_mapping]
        
        tar_root_states = self.simulator.all_root_states[self.aux_main_env_mapping]
        
        env_ids = torch.arange(self.num_envs).to(self.device)
        
        self._reset_robot_states_callback(env_ids, {"root_states": tar_root_states, "dof_states": tar_dof_states})

        
        
        # reset dof states
                
                
    def _log_sysid_states(self):
        self.log_dict['root_rot'] = self.simulator.robot_root_states[:, 3:7]
        self.log_dict['root_trans_offset'] = self.simulator.robot_root_states[:, :3] - self.env_origins
        self.log_dict['root_lin_vel'] = self.simulator.robot_root_states[:, 7:10]
        self.log_dict['root_ang_vel'] = self.simulator.robot_root_states[:, 10:13]
        self.log_dict['dof'] = self.simulator.dof_pos
        self.log_dict['dof_vel'] = self.simulator.dof_vel
        self.log_dict['torque'] = self.torques
        self.log_dict['ctrl'] = self.ctrls
        
        self.log_dict['root_rot_ref'] = wxyz_to_xyzw(self.ref_traj['root_rot'][self.step_idx])
        self.log_dict['root_trans_offset_ref'] = self.ref_traj['root_trans_offset'][self.step_idx]
        self.log_dict['root_lin_vel_ref'] = self.ref_traj['root_lin_vel'][self.step_idx]
        self.log_dict['root_ang_vel_ref'] = self.ref_traj['root_ang_vel'][self.step_idx]
        self.log_dict['dof_ref'] = self.ref_traj['dof'][self.step_idx]
        self.log_dict['dof_vel_ref'] = self.ref_traj['dof_vel'][self.step_idx]
        self.log_dict['action'] = self.ref_traj['action'][self.step_idx]
        self.log_dict['tau_est'] = self.ref_traj['tau_est'][self.step_idx]
        
        self.log_dict['rew_joint_pos_tracking'] = self._reward_joint_pos_tracking()
        self.log_dict['rew_joint_vel_tracking'] = self._reward_joint_vel_tracking()
        self.log_dict['rew_ctrl_tracking'] = self._reward_ctrl_tracking()
        self.log_dict['rew_base_pos_tracking'] = self._reward_base_pos_tracking()
        self.log_dict['rew_base_rot_tracking'] = self._reward_base_rot_tracking()
        self.log_dict['rew_base_lin_vel_tracking'] = self._reward_base_lin_vel_tracking()
        self.log_dict['rew_base_ang_vel_tracking'] = self._reward_base_ang_vel_tracking()
        self.log_dict['rew_fisher_infomation_matrix'] = self._reward_fisher_infomation_matrix()
    
    def _reward_fisher_infomation_matrix(self):
        root_state_main = self.simulator.robot_root_states[self.main_idx]
        root_state_aux = self.simulator.robot_root_states[self.aux_idx]
        root_state_diff = (root_state_main - root_state_aux) / 1.0 # constant d_theta
        fisher_info_root = torch.norm(root_state_diff, dim=[-1, -2]) ** 2
        # print(f"fisher_info_root: {fisher_info_root.shape}")
        
        dof_pos_main = self.simulator.dof_pos[self.main_idx]
        dof_pos_aux = self.simulator.dof_pos[self.aux_idx]
        dof_pos_diff = (dof_pos_main - dof_pos_aux) / 1.0 # constant d_theta
        fisher_info_dof = torch.norm(dof_pos_diff, dim=[-1, -2]) ** 2
        # print(f"fisher_info_dof: {fisher_info_dof.shape}")
        
        rew = fisher_info_root + fisher_info_dof
        
        rew = rew.repeat(self.num_of_params + 1, 1).T
        # print(f"rew: {rew.shape}")
        
        return rew.flatten()
    
    
    
    def _reward_joint_pos_tracking(self):
        dof_pos = self.simulator.dof_pos
        dof_pos_ref = self.ref_traj['dof'][self.step_idx]
        cost = torch.sum(torch.square(dof_pos - dof_pos_ref), dim=-1)
        return -cost
      
    def _reward_joint_vel_tracking(self):
        dof_vel = self.simulator.dof_vel
        dof_vel_ref = self.ref_traj['dof_vel'][self.step_idx]
        cost = torch.sum(torch.square(dof_vel - dof_vel_ref), dim=-1)
        return -cost
    
    def _reward_ctrl_tracking(self):
        ctrl = self.ctrls
        tau_est = self.ref_traj['tau_est'][self.step_idx]
        cost = torch.sum(torch.square(ctrl - tau_est), dim=-1)
        return -cost
    
    def _reward_base_pos_tracking(self):
        offset = self.env_origins
        root_pos = self.simulator.robot_root_states[:, :3] - offset
        root_pos_ref = self.ref_traj['root_trans_offset'][self.step_idx]
        cost = torch.sum(torch.square(root_pos - root_pos_ref), dim=-1)
        return -cost
    
    def _reward_base_rot_tracking(self):
        root_rot = self.simulator.robot_root_states[:, 3:7]
        root_rot_ref = wxyz_to_xyzw(self.ref_traj['root_rot'][self.step_idx])
        dot = torch.sum(root_rot * root_rot_ref, dim=-1)
        cost = 1.0 - torch.square(dot)
        return -cost
    
    def _reward_base_lin_vel_tracking(self):
        root_lin_vel = self.simulator.robot_root_states[:, 7:10]
        root_lin_vel_ref = self.ref_traj['root_lin_vel'][self.step_idx]
        cost = torch.sum(torch.square(root_lin_vel - root_lin_vel_ref), dim=-1)
        return -cost
    
    def _reward_base_ang_vel_tracking(self):
        root_ang_vel = self.simulator.robot_root_states[:, 10:13]
        root_ang_vel_ref = self.ref_traj['root_ang_vel'][self.step_idx]
        cost = torch.sum(torch.square(root_ang_vel - root_ang_vel_ref), dim=-1)
        return -cost