import torch

from torch import nn, Tensor

import time
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict


from hydra.utils import instantiate

from spigym.envs.base_task.base_task import BaseTask

class BaseAlgo:
    def __init__(self, env: BaseTask, config, device):
        self.env = env
        self.config = config
        self.device = device

    def setup(self):
        return NotImplementedError

    def learn(self):
        return NotImplementedError

    def load(self, path):
        return NotImplementedError
    
    @property
    def inference_model(self):
        return NotImplementedError

    def env_step(self, actions, extra_info=None):
        obs_dict, rewards, dones, extras = self.env.step(actions, extra_info)
        return obs_dict, rewards, dones, extras

    @torch.no_grad()
    def evaluate_policy(self):
        return NotImplementedError

    def save(self, path=None, name="last.ckpt"):
        raise NotImplementedError
