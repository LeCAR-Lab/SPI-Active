import torch
import numpy as np
from loguru import logger
from copy import deepcopy


class SysidRigidBodyInterface:

    def __init__(self):
        self.param_dict = {} # {body_name: {mass: mass_tensor, inertia: inertia_tensor, com: com_tensor}}
        self.default_param_dict = {}
        self.debug_print_dict = {}


    def set_body_CoM(self, body_name: str, com: np.ndarray):
        """
        set the CoM of the body, update the props and update default_param_dict
        """
        if body_name not in self.param_dict:
            self.param_dict[body_name] = {}
        self.param_dict[body_name]['com'] = com.copy()
        # print(f"Set body {body_name} CoM to {self.param_dict[body_name]['com']}")

    def set_body_mass(self, body_name: str, mass: np.ndarray):
        """
        set the mass of the body, update the props and update default_param_dict
        """
        if body_name not in self.param_dict:
            self.param_dict[body_name] = {}
        self.param_dict[body_name]['mass'] = mass.copy()
        # print(f"Set body {body_name} mass to {self.param_dict[body_name]['mass']}")
    
    def set_body_inertia(self, body_name: str, inertia: np.ndarray):
        """
        set the inertia of the body, update the props and update default_param_dict
        """
        if body_name not in self.param_dict:
            self.param_dict[body_name] = {}
        self.param_dict[body_name]['inertia'] = inertia.copy()
        # print(f"Set body {body_name} inertia to {self.param_dict[body_name]['inertia']}")
    
    def get_body_CoM(self, body_name: str):
        """
        get the CoM of the body
        """
        return self.param_dict[body_name]['com'].copy()
    
    def get_body_mass(self, body_name: str):
        """
        get the mass of the body
        """
        return self.param_dict[body_name]['mass'].copy()
    
    def get_body_inertia(self, body_name: str):
        """
        get the inertia of the body
        """
        return self.param_dict[body_name]['inertia'].copy()
    
    
    
    def get_body_default_CoM(self, body_name: str):
        """
        get the default CoM of the body
        """
        if body_name not in self.default_param_dict:
            raise ValueError(f"Body name {body_name} not found in body list.")
        return self.default_param_dict[body_name]['com'].mean(axis=0)
    
    
    def get_body_default_inertia(self, body_name: str):
        """
        get the default inertia of the body
        """
        if body_name not in self.default_param_dict:
            raise ValueError(f"Body name {body_name} not found in body list.")
        return self.default_param_dict[body_name]['inertia'].mean(axis=0)
    
    
    def get_body_default_mass(self, body_name: str):
        """
        get the default mass of the body
        """
        if body_name not in self.default_param_dict:
            raise ValueError(f"Body name {body_name} not found in body list.")
        return self.default_param_dict[body_name]['mass'].mean(axis=0)
      
    
    def _pre_create_env(self):
        if not self.param_dict:
            logger.warning("No parameter range set.")
            logger.warning("In replay mode, the simulator will use the default parameters.")            

        
        self.default_param_dict = deepcopy(self.param_dict)
        # print(f"Default param dict: {self.default_param_dict}")
        
        self.debug_print_dict = {
            'body_name': []
        }
    
    
    def set_param_for_each_env(self, props, env_id, _body_list):
        """
        prop: list of rigid body properties, after calling gym.get_actor_rigid_body_properties
        """

        
        for body_name, params in self.param_dict.items():
            if body_name not in _body_list:
                raise ValueError(f"Body name {body_name} not found in body list.")
            else:
                body_idx = _body_list.index(body_name)
                
                                
            if 'mass' in params:
                self._set_mass(props, env_id, body_name, body_idx, params['mass'][env_id])
            if 'inertia' in params:
                self._set_inertia(props, env_id, body_name, body_idx, params['inertia'][env_id]) 
            if 'com' in params:
                self._set_com(props, env_id, body_name, body_idx, params['com'][env_id])
                        
            if body_name not in self.debug_print_dict['body_name']:
                self.debug_print_dict['body_name'].append(body_name)
                logger.info(f"Setting parameters for body: {body_name}: mass: {'mass' in params}, inertia: {'inertia' in params}, com: {'com' in params}")
                logger.info(params.keys())
    
    def param_to_tensor(self):
        for body_name, params in self.param_dict.items():
            for key, value in params.items():
                self.param_dict[body_name][key] = torch.tensor(value)
        
        for body_name, params in self.default_param_dict.items():
            for key, value in params.items():
                self.default_param_dict[body_name][key] = torch.tensor(value)
            

    def _set_mass(self, props, env_id, body_name: str, body_idx: int, mass: np.ndarray):
        """
        set the mass of the body, update the props and update default_param_dict
        """
        raise NotImplementedError
        
    def _set_inertia(self, props, env_id, body_name: str, body_idx: int, inertia: np.ndarray):
        """
        set the inertia of the body, update the props and update default_param_dict
        """
        raise NotImplementedError
        
    def _set_com(self, props, env_id, body_name: str, body_idx: int, com: np.ndarray):
        """
        set the com of the body, update the props and update default_param_dict
        """
        raise NotImplementedError
    




class SysidRigidBodyInterfaceIsaacGym(SysidRigidBodyInterface):
    def _set_mass(self, props, env_id, body_name: str, body_idx: int, mass: np.ndarray):
        self.default_param_dict[body_name]['mass'][env_id] = props[body_idx].mass
        props[body_idx].mass = mass

    def _set_inertia(self, props, env_id, body_name: str, body_idx: int, inertia: np.ndarray):
        self.default_param_dict[body_name]['inertia'][env_id, 0] = props[body_idx].inertia.x.x
        self.default_param_dict[body_name]['inertia'][env_id, 1] = props[body_idx].inertia.y.y
        self.default_param_dict[body_name]['inertia'][env_id, 2] = props[body_idx].inertia.z.z
        props[body_idx].inertia.x.x = inertia[0]
        props[body_idx].inertia.y.y = inertia[1]
        props[body_idx].inertia.z.z = inertia[2]

    def _set_com(self, props, env_id, body_name: str, body_idx: int, com: np.ndarray):
        self.default_param_dict[body_name]['com'][env_id, 0] = props[body_idx].com.x
        self.default_param_dict[body_name]['com'][env_id, 1] = props[body_idx].com.y
        self.default_param_dict[body_name]['com'][env_id, 2] = props[body_idx].com.z
        props[body_idx].com.x = com[0]
        props[body_idx].com.y = com[1]
        props[body_idx].com.z = com[2]