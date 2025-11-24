import torch

num_of_params = 2
num_envs = 10
total_env = num_envs * (num_of_params + 1)
aux_main_env_mapping = torch.arange(0, num_envs * (num_of_params + 1), num_of_params + 1, device='cuda').repeat(num_of_params + 1, 1).T.flatten()
print(aux_main_env_mapping)

aux_idx = torch.arange(0, total_env, device='cuda').view(num_envs, num_of_params + 1)
main_idx, aux_idx = aux_idx[:, 0:1], aux_idx[:, 1:]
print(main_idx)
print(aux_idx)

state = torch.randn(num_envs * (num_of_params + 1), 17, device='cuda')
print(state.shape)
state_main = state[main_idx]
print(state_main.shape)
state_aux = state[aux_idx]
print(state_aux.shape)
state_diff = state_main - state_aux

reset_to_main_state = state[aux_main_env_mapping]
print(reset_to_main_state.shape)

