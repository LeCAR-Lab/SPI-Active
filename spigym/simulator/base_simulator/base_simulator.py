import sys
import os
from loguru import logger
import torch

class BaseSimulator:
    """
    Base class for robotic simulation environments, providing a framework for simulation setup, 
    environment creation, and control over robotic assets and simulation properties.
    """
    def __init__(self, config, device):
        """
        Initializes the base simulator with configuration settings and simulation device.
        
        Args:
            config (dict): Configuration dictionary for the simulation.
            device (str): Device type for simulation ('cpu' or 'cuda').
        """
        self.config = config
        self.sim_device = device
        self.headless = False

        self._rigid_body_pos: torch.Tensor
        self._rigid_body_rot: torch.Tensor
        self._rigid_body_vel: torch.Tensor
        self._rigid_body_ang_vel: torch.Tensor

    # ----- Configuration Setup Methods -----

    def set_headless(self, headless):
        """
        Sets the headless mode for the simulator.

        Args:
            headless (bool): If True, runs the simulation without graphical display.
        """
        self.headless = headless

    def setup(self):
        """
        Initializes the simulator parameters and environment. This method should be implemented 
        by subclasses to set specific simulator configurations.
        """
        raise NotImplementedError("The 'setup' method must be implemented in subclasses.")

    # ----- Terrain Setup Methods -----

    def setup_terrain(self, mesh_type):
        """
        Configures the terrain based on specified mesh type. 

        Args:
            mesh_type (str): Type of terrain mesh ('plane', 'heightfield', 'trimesh').
        """
        raise NotImplementedError("The 'setup_terrain' method must be implemented in subclasses.")

    # ----- Robot Asset Setup Methods -----

    def load_assets(self, robot_config):
        """
        Loads the robot assets into the simulation environment.
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names
        Args:
            robot_config (dict): RoboVerse Configuration for the robot asset.
        """
        raise NotImplementedError("The 'load_assets' method must be implemented in subclasses.")

    # ----- Environment Creation Methods -----

    def create_envs(self, num_envs, env_origins, base_init_state, env_config):
        """
        Creates and initializes environments with specified configurations.

        Args:
            num_envs (int): Number of environments to create.
            env_origins (list): List of origin positions for each environment.
            base_init_state (array): Initial state of the base.
            env_config (dict): Configuration for each environment.
        """
        raise NotImplementedError("The 'create_envs' method must be implemented in subclasses.")

    # ----- Property Retrieval Methods -----

    def get_dof_limits_properties(self):
        """
        Retrieves the DOF (degrees of freedom) limits and properties.
        
        Returns:
            Tuple of tensors representing position limits, velocity limits, and torque limits for each DOF.
        """
        raise NotImplementedError("The 'get_dof_limits_properties' method must be implemented in subclasses.")
    
    def find_rigid_body_indice(self, body_name):
        """
        Finds the index of a specified rigid body.

        Args:
            body_name (str): Name of the rigid body to locate.

        Returns:
            int: Index of the rigid body.
        """
        raise NotImplementedError("The 'find_rigid_body_indice' method must be implemented in subclasses.")
    
    # ----- Simulation Preparation and Refresh Methods -----

    def prepare_sim(self):
        """
        Prepares the simulation environment and refreshes any relevant tensors.
        """
        raise NotImplementedError("The 'prepare_sim' method must be implemented in subclasses.")
    
    def refresh_sim_tensors(self):
        """
        Refreshes the state tensors in the simulation to ensure they are up-to-date.
        """
        raise NotImplementedError("The 'refresh_sim_tensors' method must be implemented in subclasses.")

    # ----- Control Application Methods -----

    def apply_torques_at_dof(self, torques):
        """
        Applies the specified torques to the robot's degrees of freedom (DOF).

        Args:
            torques (tensor): Tensor containing torques to apply.
        """
        raise NotImplementedError("The 'apply_torques_at_dof' method must be implemented in subclasses.")
    
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        """
        Sets the root state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states will be set.
            root_states (tensor): New root states to apply.
        """
        raise NotImplementedError("The 'set_actor_root_state_tensor' method must be implemented in subclasses.")
    
    def set_dof_state_tensor(self, set_env_ids, dof_states):
        """
        Sets the DOF state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states will be set.
            dof_states (tensor): New DOF states to apply.
        """
        raise NotImplementedError("The 'set_dof_state_tensor' method must be implemented in subclasses.")

    def simulate_at_each_physics_step(self):
        """
        Advances the simulation by a single physics step.
        """
        raise NotImplementedError("The 'simulate_at_each_physics_step' method must be implemented in subclasses.")

    # ----- Viewer Setup and Rendering Methods -----

    def setup_viewer(self):
        """
        Sets up a viewer for visualizing the simulation, allowing keyboard interactions.
        """
        raise NotImplementedError("The 'setup_viewer' method must be implemented in subclasses.")

    def render(self, sync_frame_time=True):
        """
        Renders the simulation frame-by-frame, syncing frame time if required.

        Args:
            sync_frame_time (bool): Whether to synchronize the frame time.
        """
        raise NotImplementedError("The 'render' method must be implemented in subclasses.")
