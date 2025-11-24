import torch
import torch.nn as nn
import inspect

class BaseModule(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(BaseModule, self).__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict

        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict.layer_config)

    def _calculate_input_dim(self):
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in self.module_config_dict['input_dim']:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim += each_input
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")
        
        self.input_dim = input_dim

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim = output_dim

    def _build_network_layer(self, layer_config):
        if layer_config['type'] == 'MLP':
            self._build_mlp_layer(layer_config)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_config['type']}")
        
    def _build_mlp_layer(self, layer_config):
        layers = []
        hidden_dims = layer_config['hidden_dims']
        output_dim = self.output_dim
        activation = getattr(nn, layer_config['activation'])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def forward(self, input):
        return self.module(input)