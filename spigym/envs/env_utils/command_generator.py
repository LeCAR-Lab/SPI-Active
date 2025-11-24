import numpy as np
import torch
from spigym.envs.env_utils.general import class_to_dict
from isaac_utils.maths import torch_rand_float

class CommandGenerator:
    def __init__(self, config, device, num_envs):
        self.config = config
        self.device = device
        self.num_envs = num_envs

        self.command_ranges = self.config.locomotion_command_ranges
        self.commands = torch.zeros(
            (num_envs, 3), dtype=torch.float32, device=self.device
        )

    def get_commands(self, env_ids):
        return self.commands[env_ids]

    def resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)

    def reset(self, env_ids):
        self.commands[env_ids] = torch.zeros(
            (env_ids.shape[0], 3), dtype=torch.float32, device=self.device
        )
