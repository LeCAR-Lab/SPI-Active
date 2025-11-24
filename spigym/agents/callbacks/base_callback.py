import torch
from torch.nn import Module

class RL_EvalCallback(Module):
    def __init__(self, config, training_loop):
        super().__init__()
        self.config = config
        self.training_loop = training_loop
        self.device = self.training_loop.device

    def on_pre_evaluate_policy(self):
        pass

    def on_pre_eval_env_step(self, actor_state):
        return actor_state
    
    def on_post_eval_env_step(self, actor_state):
        return actor_state
    
    def on_post_evaluate_policy(self):
        pass
