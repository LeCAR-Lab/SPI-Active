from __future__ import annotations

import torch
import torch.nn as nn
from .modules import BaseModule


class Estimator(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(Estimator, self).__init__()
        self.module = BaseModule(obs_dim_dict, module_config_dict)

    # def estimate(self, obs_history):
    #     return self.module(obs_history)
    
    def forward(self, obs_history):
        return self.module(obs_history)