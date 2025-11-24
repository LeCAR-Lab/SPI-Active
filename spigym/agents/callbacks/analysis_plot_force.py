import torch
from torch import Tensor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import json
from flask import send_file

from spigym.agents.callbacks.base_callback import RL_EvalCallback
from spigym.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from spigym.agents.ppo.ppo import PPO


class AnalysisPlotForce(RL_EvalCallback):
    training_loop: PPO
    env: LeggedRobotBase

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        env: LeggedRobotBase = self.training_loop.env
        self.env = env
        self.policy = self.training_loop._get_inference_policy()
        self.num_envs = self.env.num_envs
        self.logger = WebLogger(self.config.sim_dt)
        self.reset_buffers()
        self.log_single_robot = self.config.log_single_robot

    def reset_buffers(self):
        self.obs_buf = [[] for _ in range(self.num_envs)]
        self.critic_obs_buf = [[] for _ in range(self.num_envs)]
        self.act_buf = [[] for _ in range(self.num_envs)]

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        self.robot_num_dofs = self.env.num_dofs
        self.log_dof_pos_limits = self.env.dof_pos_limits.cpu().numpy()
        self.log_dof_vel_limits = self.env.dof_vel_limits.cpu().numpy()
        self.log_dof_torque_limits = self.env.torque_limits.cpu().numpy()
        self.logger.set_robot_limits(self.log_dof_pos_limits, self.log_dof_vel_limits, self.log_dof_torque_limits)
        self.logger.set_robot_num_dofs(self.robot_num_dofs)

    def on_post_evaluate_policy(self):
        pass

    def on_pre_eval_env_step(self, actor_state):
        obs: Tensor = actor_state["obs"]["actor_obs"].cpu()
        critic_obs: Tensor = actor_state["obs"]["critic_obs"].cpu()
        actions: Tensor = actor_state["actions"].cpu()

        for i in range(self.num_envs):
            self.obs_buf[i].append(obs[i])
            self.critic_obs_buf[i].append(critic_obs[i])
            self.act_buf[i].append(actions[i])
    
        rigid_body_pos = self.env._rigid_body_pos_extend
        ref_body_pos = self.env.ref_body_pos_extend
        vr_id = self.env.motion_tracking_id



        if self.log_single_robot:
            self.logger.log_states(
                {
                'dof_pos_target': actions[0].cpu().numpy(),
                'dof_pos': self.env.simulator.dof_pos[0].cpu().numpy(),
                'dof_vel': self.env.simulator.dof_vel[0].cpu().numpy(),
                'dof_torque': self.env.torques[0].cpu().numpy(),
                'base_vel_x': self.env.base_lin_vel[0, 0].item(),
                'base_vel_y': self.env.base_lin_vel[0, 1].item(),
                'base_vel_z': self.env.base_lin_vel[0, 2].item(),
                'base_vel_yaw': self.env.base_ang_vel[0, 2].item(),
                'contact_forces_z': self.env.simulator.contact_forces[0, self.env.feet_indices, 2].cpu().numpy(),
                'left_hand_force': self.env.apply_force_tensor[0, self.env.left_hand_link_index, :].cpu().numpy(),
                'right_hand_force': self.env.apply_force_tensor[0, self.env.right_hand_link_index, :].cpu().numpy(),
                'vr_3point_pos': ref_body_pos[0, vr_id, :].cpu().numpy(),
                'vr_3point_pos_target': rigid_body_pos[0, vr_id, :].cpu().numpy()
                }
            )
        else:
            # log average of all robots
            self.logger.log_states(
                {
                    'dof_pos_target': actions.mean(dim=0).cpu().numpy(),
                    'dof_pos': self.env.simulator.dof_pos.mean(dim=0).cpu().numpy(),
                    'dof_vel': self.env.simulator.dof_vel.mean(dim=0).cpu().numpy(),
                    'dof_torque': self.env.torques.mean(dim=0).cpu().numpy(),
                    'base_vel_x': self.env.base_lin_vel[:, 0].mean().item(),
                    'base_vel_y': self.env.base_lin_vel[:, 1].mean().item(),
                    'base_vel_z': self.env.base_lin_vel[:, 2].mean().item(),
                    'base_vel_yaw': self.env.base_ang_vel[:, 2].mean().item(),
                    'contact_forces_z': self.env.simulator.contact_forces[:, self.env.feet_indices, 2].mean(dim=1).cpu().numpy(),
                    'left_hand_force': self.env.apply_force_tensor[:, self.env.left_hand_link_index, :].mean(dim=1).cpu().numpy(),
                    'right_hand_force': self.env.apply_force_tensor[:, self.env.right_hand_link_index, :].mean(dim=0).cpu().numpy(),
                    'vr_3point_pos': ref_body_pos[:, vr_id, :].mean(dim=0).cpu().numpy(),
                    'vr_3point_pos_target': rigid_body_pos[:, vr_id, :].mean(dim=0).cpu().numpy()
                }
            )
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        step = actor_state["step"]
        if (step + 1) % self.config.plot_update_interval == 0:
            self.logger.plot_states()
        return actor_state

class WebLogger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.thread = None

    def set_robot_limits(self, dof_pos_limits, dof_vel_limits, dof_torque_limits):
        self.log_dof_pos_limits = dof_pos_limits
        self.log_dof_vel_limits = dof_vel_limits
        self.log_dof_torque_limits = dof_torque_limits

    def set_robot_num_dofs(self, num_dofs):
        self.robot_num_dofs = num_dofs

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run_server)
            self.thread.start()
        self._update_plot()
    
    def _run_server(self):
        @self.app.route('/')
        def index():
            return send_file('analysis_plot_template.html')

        self.socketio.run(self.app, debug=False, use_reloader=False)
    
    def _update_plot(self):
        log = self.state_log.copy()
        total_time = len(next(iter(log.values()))) * self.dt
        time = np.linspace(0, total_time, len(next(iter(log.values()))))

        for key in log:
            if isinstance(log[key], list):
                log[key] = np.array(log[key])

        BLUE = '#005A9D'
        RED = '#DA2513'
        YELLOW = '#EEDE70'
        BLACK = '#000000'

        force_to_position_scale = 0.05 # force to position scale, for better visualization
        force_to_position_bias = 0.0 # bias to force to position, for better visualization

        num_dofs = self.robot_num_dofs

        def get_subplot_titles():
            titles = [
                'Left hand x',
                'Left hand y',
                'Left hand z',
                'Left hand force',
                'Right hand x',
                'Right hand y',
                'Right hand z',
                'Right hand force',
                'Head x',
                'Head y',
                'Head z',
                'Vertical Contact forces', 
                'Left hand force x',
                'Left hand force y',
                'Left hand force z',
                'Left hand force',
                'Right hand force x',
                'Right hand force y',
                'Right hand force z',
                'Right hand force',
                'Base velocity x', 
                'Base velocity y', 
                'Base velocity yaw', 
                'Base velocity z',
            ]
            for i in range(num_dofs):
                titles.extend([f'DOF {i} Position', f'DOF {i} Velocity', f'DOF {i} Torque', f'DOF {i} Torque/Velocity'])
            return titles

        # Calculate number of rows needed
        num_rows = 5 + num_dofs

        fig = make_subplots(rows=num_rows, cols=4, subplot_titles=get_subplot_titles())

        def add_trace(x, y, color, row, col, name=None, show_legend=False):
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color), name=name, showlegend=show_legend), row=row, col=col)

        add_trace(time, log["vr_3point_pos"][:, 0, 0], BLUE, 1, 1, "Left Hand Position x")
        add_trace(time, log["vr_3point_pos_target"][:, 0, 0], RED, 1, 1, "Left Hand Position Target x")
        # add_trace(time, log["left_hand_force"][:, 0] * force_to_position_scale + force_to_position_bias, RED, 1, 1, "Left Hand Force x")

        add_trace(time, log["vr_3point_pos"][:, 0, 1], BLUE, 1, 2, "Left Hand Position y")
        add_trace(time, log["vr_3point_pos_target"][:, 0, 1], RED, 1, 2, "Left Hand Position Target y")
        # add_trace(time, log["left_hand_force"][:, 1] * force_to_position_scale + force_to_position_bias, RED, 1, 2, "Left Hand Force y")

        add_trace(time, log["vr_3point_pos"][:, 0, 2], BLUE, 1, 3, "Left Hand Position z")
        add_trace(time, log["vr_3point_pos_target"][:, 0, 2], RED, 1, 3, "Left Hand Position Target z")
        # add_trace(time, log["left_hand_force"][:, 2] * force_to_position_scale + force_to_position_bias, RED, 1, 3, "Left Hand Force z")

        add_trace(time, np.linalg.norm(log["left_hand_force"], axis=1), RED, 1, 4, "Left Hand Force")

        add_trace(time, log["vr_3point_pos"][:, 1, 0], BLUE, 2, 1, "Right Hand Position x")
        add_trace(time, log["vr_3point_pos_target"][:, 1, 0], RED, 2, 1, "Right Hand Position Target x")
        # add_trace(time, log["right_hand_force"][:, 0] * force_to_position_scale + force_to_position_bias, RED, 2, 1, "Right Hand Force x")

        add_trace(time, log["vr_3point_pos"][:, 1, 1], BLUE, 2, 2, "Right Hand Position y")
        add_trace(time, log["vr_3point_pos_target"][:, 1, 1], RED, 2, 2, "Right Hand Position Target y")
        # add_trace(time, log["right_hand_force"][:, 1] * force_to_position_scale + force_to_position_bias, RED, 2, 2, "Right Hand Force y")

        add_trace(time, log["vr_3point_pos"][:, 1, 2], BLUE, 2, 3, "Right Hand Position z")
        add_trace(time, log["vr_3point_pos_target"][:, 1, 2], RED, 2, 3, "Right Hand Position Target z")
        # add_trace(time, log["right_hand_force"][:, 2] * force_to_position_scale + force_to_position_bias, RED, 2, 3, "Right Hand Force z")

        add_trace(time, np.linalg.norm(log["right_hand_force"], axis=1), RED, 2, 4, "Right Hand Force")

        add_trace(time, log["vr_3point_pos"][:, 2, 0], BLUE, 3, 1, "Head Position x")
        add_trace(time, log["vr_3point_pos_target"][:, 2, 0], BLACK, 3, 1, "Head Position Target x")

        add_trace(time, log["vr_3point_pos"][:, 2, 1], BLUE, 3, 2, "Head Position y")
        add_trace(time, log["vr_3point_pos_target"][:, 2, 1], BLACK, 3, 2, "Head Position Target y")

        add_trace(time, log["vr_3point_pos"][:, 2, 2], BLUE, 3, 3, "Head Position z")
        add_trace(time, log["vr_3point_pos_target"][:, 2, 2], BLACK, 3, 3, "Head Position Target z")

        # Vertical Contact forces
        forces = log["contact_forces_z"]
        for i in range(forces[0].shape[0]):
            add_trace(time, [force[i] for force in forces], BLUE, 3, 4, f"Force {i}")

        add_trace(time, log["left_hand_force"][:, 2], BLUE, 3, 4, "Left Hand Force z")
        add_trace(time, log["right_hand_force"][:, 2], RED, 3, 4, "Right Hand Force z")

        add_trace(time, log["left_hand_force"][:, 0], RED, 4, 1, "Left Hand Force x")
        add_trace(time, log["left_hand_force"][:, 1], RED, 4, 2, "Left Hand Force y")
        add_trace(time, log["left_hand_force"][:, 2], RED, 4, 3, "Left Hand Force z")
        add_trace(time, np.linalg.norm(log["left_hand_force"], axis=1), RED, 4, 4, "Left Hand Force")

        add_trace(time, log["right_hand_force"][:, 0], RED, 5, 1, "Right Hand Force x")
        add_trace(time, log["right_hand_force"][:, 1], RED, 5, 2, "Right Hand Force y")
        add_trace(time, log["right_hand_force"][:, 2], RED, 5, 3, "Right Hand Force z")
        add_trace(time, np.linalg.norm(log["right_hand_force"], axis=1), RED, 5, 4, "Right Hand Force")
        
        def add_limit_lines(row, col, lower, upper, color=YELLOW):
            fig.add_shape(type="rect", x0=time[0], x1=time[-1], y0=lower, y1=upper,
                        fillcolor=color, line=dict(width=0), layer='below', row=row, col=col)

        # DOF Positions, Velocities, and Torques
        for i in range(num_dofs):
            row = i + 6  # Start from the third row

            # Position
            add_trace(time, [pos[i] for pos in log["dof_pos"]], BLUE, row, 1, f"DOF {i} pos")
            add_trace(time, [pos[i] for pos in log["dof_pos_target"]], RED, row, 1, f"DOF {i} pos target")
            add_limit_lines(row, 1, self.log_dof_pos_limits[i, 0], self.log_dof_pos_limits[i, 1])
            # Velocity
            add_trace(time, [vel[i] for vel in log["dof_vel"]], BLUE, row, 2, f"DOF {i} vel")
            add_limit_lines(row, 2, -self.log_dof_vel_limits[i], self.log_dof_vel_limits[i])
            # Torque
            add_trace(time, [torque[i] for torque in log["dof_torque"]], BLUE, row, 3, f"DOF {i} torque")
            add_limit_lines(row, 3, -self.log_dof_torque_limits[i], self.log_dof_torque_limits[i])
            
            # Torque/Velocity curve
            fig.add_trace(go.Scatter(
                x=[vel[i] for vel in log["dof_vel"]], 
                y=[torque[i] for torque in log["dof_torque"]], 
                mode='markers', 
                marker=dict(color=BLUE, size=2), 
                showlegend=False,
                name=f"DOF {i} Torque/Velocity"
            ), row=row, col=4)

            # Add velocity limits
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=-self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

            # Add torque limits
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=-self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

        fig.update_layout(height=300*num_rows, width=1500, title_text="Robot State Plots", showlegend=True)
        
        # Update x and y axis labels
        for i in range(num_rows):
            for j in range(3):
                fig.update_xaxes(title_text="time [s]", row=i+1, col=j+1)
            fig.update_xaxes(title_text="", row=i+1, col=4)

        
        
        fig.update_yaxes(title_text="Left Hand Position x [m]", row=1, col=1)
        fig.update_yaxes(title_text="Left Hand Position y [m]", row=1, col=2)
        fig.update_yaxes(title_text="Left Hand Position z [m]", row=1, col=3)
        fig.update_yaxes(title_text="Left Hand Force [N]", row=1, col=4)
        fig.update_yaxes(title_text="Right Hand Position x [m]", row=2, col=1)
        fig.update_yaxes(title_text="Right Hand Position y [m]", row=2, col=2)
        fig.update_yaxes(title_text="Right Hand Position z [m]", row=2, col=3)
        fig.update_yaxes(title_text="Right Hand Force [N]", row=2, col=4)

        fig.update_yaxes(title_text="Head Position x [m]", row=3, col=1)
        fig.update_yaxes(title_text="Head Position y [m]", row=3, col=2)
        fig.update_yaxes(title_text="Head Position z [m]", row=3, col=3)

        fig.update_yaxes(title_text="Vertical Contact Forces [N]", row=3, col=4)

        for i in range(4, num_rows + 1):
            fig.update_yaxes(title_text="Position [rad]", row=i, col=1)
            fig.update_yaxes(title_text="Velocity [rad/s]", row=i, col=2)
            fig.update_yaxes(title_text="Torque [Nm]", row=i, col=3)
            fig.update_yaxes(title_text="Torque/Velocity", row=i, col=4)

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.socketio.emit('update_plots', plot_json)

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.thread:
            self.socketio.stop()
            self.thread.join()