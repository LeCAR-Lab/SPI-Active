"""
IsaacGymActiveSysId simulator adapter.

Extends the base IsaacGym wrapper to allow per-environment overrides of
rigid-body properties (mass, CoM, inertia) and motor model parameters
based on a `params_dict` constructed by the Active SysID environment.

The `params_dict` is expected to have entries like:

  {
    "mass": {"body_name": "base", "value": [.. per-env values ..]},
    "comx": {"body_name": "base", "value": [...]},
    "motor_model_hip_a": {"value": [...]},
    ...
  }

For each env, `_process_rigid_body_props` will set the corresponding fields
on the PhysX-like property structure prior to finalizing the actor creation.
"""

from loguru import logger
import numpy as np
from spigym.simulator.isaacgym.isaacgym import IsaacGym





class IsaacGymActiveSysId(IsaacGym):
    def __init__(self, config, device, ):
        
        
        super().__init__(config, device)
        self.params_dict = config.params_dict
    
    
        
        
    def _process_rigid_body_props(self, props, env_id):
        """Hook to mutate rigid body properties for a specific env.

        Called during asset instantiation to set per-env masses/CoM/inertia.
        """
        # First, let the SysId base class apply any parameters provided via the interface
        props = super()._process_rigid_body_props(props, env_id)

         # Check params_dict and call corresponding setter methods
        if self.params_dict:
            for key, value in self.params_dict.items():
                # Convert key to setter method name (e.g., "mass" -> "set_mass")
                method_name = f"set_{key}"
                if hasattr(self, method_name):
                    setter_method = getattr(self, method_name)
                    setter_method(props, env_id, value)
                else:
                    logger.debug(f"No setter method found for key: {key}")
        
        return props

    
    def set_mass(self, props, env_id, value):
        """Set body mass for the specified env."""
        body_idx = self._body_list.index(value['body_name'])
        props[body_idx].mass = value['value'][env_id]
    
    def set_comx(self, props, env_id, value):
        """Set body center-of-mass X for the specified env."""
        body_idx = self._body_list.index(value['body_name'])
        props[body_idx].com.x = value['value'][env_id]
    
    def set_comy(self, props, env_id, value):
        """Set body center-of-mass Y for the specified env."""
        body_idx = self._body_list.index(value['body_name'])
        props[body_idx].com.y = value['value'][env_id]
    
    def set_comz(self, props, env_id, value):
        """Set body center-of-mass Z for the specified env."""
        body_idx = self._body_list.index(value['body_name'])
        props[body_idx].com.z = value['value'][env_id]
    
    def set_inertiax(self, props, env_id, value):
        """Set body inertia Ixx for the specified env."""
        body_idx = self._body_list.index(value['body_name'])
        props[body_idx].inertia.x.x = value['value'][env_id]
    
    def set_inertiaiy(self, props, env_id, value):
        """Set body inertia Iyy for the specified env."""
        body_idx = self._body_list.index(value['body_name'])
        props[body_idx].inertia.y.y = value['value'][env_id]
    
    def set_inertiaz(self, props, env_id, value):
        """Set body inertia Izz for the specified env."""
        body_idx = self._body_list.index(value['body_name'])
        props[body_idx].inertia.z.z = value['value'][env_id]
    

        
        
