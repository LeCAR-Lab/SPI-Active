import torch
from torch import Tensor
from termcolor import colored
from loguru import logger

class HistoryHandler:
    
    def __init__(self, num_envs, history_config, obs_dims, device):
        self.obs_dims = obs_dims
        self.device = device
        self.num_envs = num_envs
        self.history = {}

        self.buffer_config = {}
        for aux_key, aux_config in history_config.items():
            for obs_key, obs_num in aux_config.items():
                if obs_key in self.buffer_config:
                    self.buffer_config[obs_key] = max(self.buffer_config[obs_key], obs_num)
                else:
                    self.buffer_config[obs_key] = obs_num
        
        for key in self.buffer_config.keys():
            self.history[key] = torch.zeros(num_envs, self.buffer_config[key], obs_dims[key], device=self.device)

        logger.info(colored("History Handler Initialized", "green"))
        for key, value in self.buffer_config.items():
            logger.info(f"Key: {key}, Value: {value}")

    def reset(self, reset_ids):
        if len(reset_ids)==0:
            return
        assert set(self.buffer_config.keys()) == set(self.history.keys()), f"History keys mismatch\n{self.buffer_config.keys()}\n{self.history.keys()}"
        for key in self.history.keys():
            self.history[key][reset_ids] *= 0.

    def add(self, key: str, value: Tensor):
        assert key in self.history.keys(), f"Key {key} not found in history"
        val = self.history[key].clone()
        self.history[key][:, 1:] = val[:, :-1]
        self.history[key][:, 0] = value.clone()
        
    def query(self, key: str):
        assert key in self.history.keys(), f"Key {key} not found in history"
        return self.history[key].clone()