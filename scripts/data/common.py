"""Shared utilities for data collection scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_NAME = "env/go2_test"


def load_config(headless: bool = True) -> OmegaConf:
    """Load and configure the environment."""
    config_dir = PROJECT_ROOT / "spigym" / "config"
    
    robot_base_cfg = OmegaConf.load(config_dir / "robot" / "robot_base.yaml")
    ConfigStore.instance().store(
        name="robot_base", node=robot_base_cfg, group="robot/go2"
    )
    
    with hydra.initialize_config_dir(
        config_dir=str(config_dir), version_base="1.1"
    ):
        cfg = hydra.compose(config_name=CONFIG_NAME)

    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.env, False)
    OmegaConf.set_struct(cfg.env.config, False)

    if "algo" not in cfg:
        cfg.algo = OmegaConf.create({"config": {"module_dict": {}}})

    from spigym.utils.helpers import pre_process_config
    pre_process_config(cfg)

    obs_dims = cfg.robot.algo_obs_dim_dict
    cfg.robot.policy_obs_dim = obs_dims["actor_obs"]
    cfg.robot.critic_obs_dim = obs_dims["critic_obs"]
    cfg.env.config.robot.policy_obs_dim = obs_dims["actor_obs"]
    cfg.env.config.robot.critic_obs_dim = obs_dims["critic_obs"]

    cfg.num_envs = 1
    cfg.headless = headless
    cfg.base_dir = str(PROJECT_ROOT / "logs")
    cfg.save_rendering_dir = str(PROJECT_ROOT / "logs" / "renderings_test")

    cfg.env.config.num_envs = 1
    cfg.env.config.headless = headless
    cfg.env.config.save_rendering_dir = cfg.save_rendering_dir

    return cfg


def create_env(cfg: OmegaConf, base_height: Optional[float] = None):
    """Create and initialize the environment."""
    from spigym.envs.legged_base_task.legged_robot_base import LeggedRobotBase

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = LeggedRobotBase(cfg.env.config, device)
    
    if not hasattr(env, "episode_sums"):
        env.episode_sums = {}
    
    env.set_is_evaluating()
    env.reset_all()

    if base_height is not None:
        env_ids = torch.arange(env.num_envs, device=env.device)
        root_states = env.simulator.robot_root_states.clone()
        root_states[:, 2] = base_height
        env.simulator.set_actor_root_state_tensor(env_ids, root_states)
        env.simulator.refresh_sim_tensors()

    return env


def format_data(frames: List[Dict[str, np.ndarray]], env, dt: float) -> Dict[str, np.ndarray]:
    """Convert frame list to npz-compatible dict."""
    timestamps = np.array([frame.pop("timestamp") for frame in frames])
    stacked = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
    
    stacked["timestamps"] = timestamps
    stacked["sim_duration"] = len(timestamps) * dt
    stacked["data_frequency"] = int(round(1.0 / dt))
    stacked["robot_type"] = "go2"
    
    stacked["pd_gain_kp"] = np.asarray(
        env.p_gains[0].detach().cpu().numpy(), dtype=np.float32
    ).reshape(-1)
    stacked["pd_gain_kd"] = np.asarray(
        env.d_gains[0].detach().cpu().numpy(), dtype=np.float32
    ).reshape(-1)
    
    return stacked


def cleanup_env(env) -> None:
    """Clean up Isaac Gym resources."""
    if hasattr(env.simulator, "gym") and hasattr(env.simulator, "sim"):
        env.simulator.gym.destroy_sim(env.simulator.sim)
        if hasattr(env.simulator, "viewer") and env.simulator.viewer is not None:
            env.simulator.gym.destroy_viewer(env.simulator.viewer)


def video_enabled(env, record_video: bool) -> bool:
    """Check if video recording is available."""
    sim = env.simulator
    return bool(
        record_video
        and getattr(sim, "visualize_viewer", False)
        and getattr(sim, "viewer", None) is not None
    )

