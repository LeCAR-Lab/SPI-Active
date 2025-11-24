import numpy as np

from spigym.simulator.isaacgym.isaacgym import IsaacGym
from spigym.simulator.isaacgym.sim_interface import SysidRigidBodyInterfaceIsaacGym


class IsaacGymSysId(IsaacGym):
    def __init__(self, config, device):
        """
        rbd_params: {body_name: {mass: mass_tensor, inertia: inertia_tensor, com: com_tensor}}
        """
                        
        super().__init__(config, device)
        self.sysid_rbd_interface = SysidRigidBodyInterfaceIsaacGym()
    
    ##############################################################
    # Set Rigid Body Properties
    ##############################################################
    def set_body_CoM(self, body_name: str, com: np.ndarray):
        self.sysid_rbd_interface.set_body_CoM(body_name, com)
    
    def set_body_mass(self, body_name: str, mass: np.ndarray):
        self.sysid_rbd_interface.set_body_mass(body_name, mass)
    
    def set_body_inertia(self, body_name: str, inertia: np.ndarray):
        self.sysid_rbd_interface.set_body_inertia(body_name, inertia)
    
    ##############################################################
    # Get Rigid Body Properties
    ##############################################################
    def get_body_CoM(self, body_name: str):
        return self.sysid_rbd_interface.get_body_CoM(body_name)
    
    def get_body_mass(self, body_name: str):
        return self.sysid_rbd_interface.get_body_mass(body_name)
    
    def get_body_inertia(self, body_name: str):
        return self.sysid_rbd_interface.get_body_inertia(body_name)
    
    ##############################################################
    # Get Rigid Body Default Properties
    ##############################################################
    def get_body_default_CoM(self, body_name: str):
        return self.sysid_rbd_interface.get_body_default_CoM(body_name)
    
    def get_body_default_mass(self, body_name: str):
        return self.sysid_rbd_interface.get_body_default_mass(body_name)
    
    def get_body_default_inertia(self, body_name: str):
        return self.sysid_rbd_interface.get_body_default_inertia(body_name)
    
    ##############################################################
    # Process Rigid Body Properties
    ##############################################################
    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        
        self.sysid_rbd_interface.set_param_for_each_env(props, env_id, self._body_list)
        
        # print(f"props: {props[0].mass}, {props[0].inertia.x.x}, {props[0].com.x}")
        
        return props
        
    
    def create_envs(self, num_envs, env_origins, base_init_state):
        
        self.sysid_rbd_interface._pre_create_env()
        
        ret = super().create_envs(num_envs, env_origins, base_init_state)
        
        # self.sysid_rbd_interface.param_to_tensor()
        
        return ret