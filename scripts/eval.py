"""Utilities for evaluating Isaac Gym replay datasets under mass variations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm import tqdm

from spigym.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from spigym.utils.helpers import pre_process_config



@dataclass
class DatasetBatch:
    init_base_pos: torch.Tensor
    target_base_pos: torch.Tensor
    init_base_ori: torch.Tensor
    target_base_ori: torch.Tensor
    init_base_lin_vel: torch.Tensor
    init_base_ang_vel: torch.Tensor
    init_joint_pos: torch.Tensor
    target_joint_pos: torch.Tensor
    init_joint_vel: torch.Tensor
    action_sequences: torch.Tensor
    motion_ends: torch.Tensor
    pd_gain_kp: torch.Tensor  # Shape: (num_samples, num_joints)
    pd_gain_kd: torch.Tensor  # Shape: (num_samples, num_joints)


@dataclass
class MassEvalResult:
    base_pos: float
    base_quat: float
    joint_pos: float


def load_env_config(base_dir: Path, num_envs: int, config_name: str, seed: int) -> OmegaConf:
    config_dir = Path(__file__).resolve().parent.parent / "spigym" / "config"
    cs = ConfigStore.instance()
    try:
        cs.store(
            name="robot_base",
            node=OmegaConf.load(config_dir / "robot" / "robot_base.yaml"),
            group="robot/go2",
            provider="mass_eval",
        )
    except ValueError:
        pass
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(config_name=config_name)

    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.env, False)
    OmegaConf.set_struct(cfg.env.config, False)

    if "algo" not in cfg:
        cfg.algo = OmegaConf.create({"config": {"module_dict": {}}})

    pre_process_config(cfg)

    obs_dims = cfg.robot.algo_obs_dim_dict
    cfg.robot.policy_obs_dim = obs_dims["actor_obs"]
    cfg.robot.critic_obs_dim = obs_dims["critic_obs"]
    cfg.env.config.robot.policy_obs_dim = obs_dims["actor_obs"]
    cfg.env.config.robot.critic_obs_dim = obs_dims["critic_obs"]

    cfg.num_envs = num_envs
    cfg.headless = True
    cfg.base_dir = str(base_dir)
    cfg.save_rendering_dir = str(base_dir / "renderings_test")
    cfg.env.config.num_envs = num_envs
    cfg.env.config.headless = True
    cfg.env.config.save_rendering_dir = cfg.save_rendering_dir

    cfg.seed = seed
    cfg.env.config.seed = seed

    if "domain_rand" in cfg.env.config:
        domain_rand = cfg.env.config.domain_rand
        for attr in (
            "push_robots",
            "randomize_gains",
            "randomize_base_mass",
            "randomize_link_mass",
            "randomize_friction",
        ):
            if attr in domain_rand:
                domain_rand[attr] = False

    return cfg


def load_dataset(paths: list[Path] | Path, horizon: int) -> Tuple[int, Dict[str, np.ndarray]]:
    """Load and concatenate multiple npz datasets.
    
    Returns: (num_samples, dataset_dict) with per-sample PD gains
    """
    paths = [paths] if isinstance(paths, Path) else paths
    
    all_datasets = []
    all_motion_ends = []
    all_pd_kp = []
    all_pd_kd = []
    
    for npz_path in paths:
        with np.load(npz_path) as data:
            base_positions = data["base_positions"]
            base_orientations = data["base_orientations"]
            base_lin_vel = data["base_linear_velocities"]
            base_ang_vel = data["base_angular_velocities"]
            joint_positions = data["joint_positions"]
            joint_velocities = data["joint_velocities"]
            actions = data["actions"].astype(np.float32)
            pd_gain_kp = data["pd_gain_kp"]
            pd_gain_kd = data["pd_gain_kd"]

        total_steps = base_positions.shape[0]
        num_samples = total_steps - horizon
        init_idx = np.arange(num_samples)
        target_idx = init_idx + horizon

        dataset = {
            "init_base_pos": base_positions[init_idx],
            "target_base_pos": base_positions[target_idx],
            "init_base_ori": base_orientations[init_idx],
            "target_base_ori": base_orientations[target_idx],
            "init_base_lin_vel": base_lin_vel[init_idx],
            "init_base_ang_vel": base_ang_vel[init_idx],
            "init_joint_pos": joint_positions[init_idx],
            "target_joint_pos": joint_positions[target_idx],
            "init_joint_vel": joint_velocities[init_idx],
        }

        action_seqs = np.stack([actions[init_idx + offset] for offset in range(horizon)], axis=1)
        dataset["action_sequences"] = action_seqs
        
        motion_ends = np.zeros(num_samples, dtype=bool)
        motion_ends[-1] = True
        
        kp_scalar = float(pd_gain_kp.flat[0])
        kd_scalar = float(pd_gain_kd.flat[0])
        num_joints = joint_positions.shape[1]
        pd_kp_samples = np.full((num_samples, num_joints), kp_scalar, dtype=np.float32)
        pd_kd_samples = np.full((num_samples, num_joints), kd_scalar, dtype=np.float32)
        
        all_datasets.append(dataset)
        all_motion_ends.append(motion_ends)
        all_pd_kp.append(pd_kp_samples)
        all_pd_kd.append(pd_kd_samples)

    # Concatenate all datasets
    concatenated = {}
    for key in all_datasets[0].keys():
        concatenated[key] = np.concatenate([d[key] for d in all_datasets], axis=0)
    
    concatenated["motion_ends"] = np.concatenate(all_motion_ends, axis=0)
    concatenated["pd_gain_kp"] = np.concatenate(all_pd_kp, axis=0)
    concatenated["pd_gain_kd"] = np.concatenate(all_pd_kd, axis=0)
    
    total_samples = concatenated["init_base_pos"].shape[0]
    print(f"Loaded {len(paths)} trajectory(s) with {total_samples} total samples")

    return total_samples, concatenated


def to_device(dataset: Dict[str, np.ndarray], device: torch.device) -> DatasetBatch:
    """Convert numpy dataset to torch tensors on device."""
    tensors = {}
    for key, value in dataset.items():
        if key == "motion_ends":
            tensors[key] = torch.from_numpy(value).to(device=device, dtype=torch.bool)
        else:
            tensors[key] = torch.from_numpy(value).to(device=device, dtype=torch.float32)
    return DatasetBatch(**tensors)


def capture_reference_masses(simulator) -> np.ndarray:
    env_ptr = simulator.envs[0]
    actor = simulator.robot_handles[0]
    props = simulator.gym.get_actor_rigid_body_properties(env_ptr, actor)
    return np.array([prop.mass for prop in props], dtype=np.float32)


def apply_pd_gains_batch(env: LeggedRobotBase, kp_batch: torch.Tensor, kd_batch: torch.Tensor) -> None:
    """Apply per-sample PD gains to environment batch.
    
    Args:
        env: Environment instance
        kp_batch: Kp gains, shape (num_envs, num_joints)
        kd_batch: Kd gains, shape (num_envs, num_joints)
    """
    env.p_gains[:] = kp_batch[0]
    env.d_gains[:] = kd_batch[0]


def apply_base_mass(simulator, reference_masses: np.ndarray, base_mass: float) -> None:
    masses = reference_masses.copy()
    masses[0] = base_mass
    for env_ptr, actor in zip(simulator.envs, simulator.robot_handles):
        props = simulator.gym.get_actor_rigid_body_properties(env_ptr, actor)
        for idx, prop in enumerate(props):
            prop.mass = float(masses[idx])
        simulator.gym.set_actor_rigid_body_properties(
            env_ptr, actor, props, recomputeInertia=True
        )
    simulator.gym.refresh_mass_matrix_tensors(simulator.sim)


@torch.no_grad()
def evaluate_batch(
    env: LeggedRobotBase,
    horizon: int,
    action_buffer: torch.Tensor,
    dataset: DatasetBatch,
) -> MassEvalResult:
    total_samples = dataset.init_base_pos.shape[0]
    batch_size = env.num_envs
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    base_pos_sum = 0.0
    base_quat_sum = 0.0
    joint_pos_sum = 0.0

    num_batches = (total_samples + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, total_samples, batch_size),
        desc="Evaluating batches",
        total=num_batches,
        unit="batch"
    )
    for start in pbar:
        end = min(start + batch_size, total_samples)
        chunk = end - start

        env.reset_all()
        
        kp_batch = dataset.pd_gain_kp[start:end]
        kd_batch = dataset.pd_gain_kd[start:end]
        if chunk < batch_size:
            kp_batch = torch.cat([kp_batch, kp_batch[0:1].expand(batch_size - chunk, -1)], dim=0)
            kd_batch = torch.cat([kd_batch, kd_batch[0:1].expand(batch_size - chunk, -1)], dim=0)
        apply_pd_gains_batch(env, kp_batch, kd_batch)

        root_states = env.simulator.robot_root_states.clone()
        root_states[:chunk, 0:3] = dataset.init_base_pos[start:end]
        root_states[:chunk, 3:7] = dataset.init_base_ori[start:end]
        root_states[:chunk, 7:10] = dataset.init_base_lin_vel[start:end]
        root_states[:chunk, 10:13] = dataset.init_base_ang_vel[start:end]
        if chunk < batch_size:
            root_states[chunk:] = root_states[0]
        env.simulator.set_actor_root_state_tensor(env_ids, root_states)

        dof_state_tensor = env.simulator.dof_state
        dof_state = dof_state_tensor.view(env.num_envs, env.dim_actions, 2)
        dof_state[:chunk, :, 0] = dataset.init_joint_pos[start:end]
        dof_state[:chunk, :, 1] = dataset.init_joint_vel[start:end]
        if chunk < batch_size:
            dof_state[chunk:, :, :] = dof_state[0]
        env.simulator.set_dof_state_tensor(env_ids, dof_state_tensor)

        env.simulator.refresh_sim_tensors()

        env.actions.zero_()
        env.last_actions.zero_()
        env.actions_after_delay.zero_()
        env.last_dof_vel.zero_()
        env.feet_air_time.zero_()
        env.episode_length_buf.zero_()
        env.reset_buf.zero_()

        motion_end_mask = dataset.motion_ends[start:end]
        eval_mask = ~(torch.cumsum(motion_end_mask.float(), dim=0) > 0)
        
        for step_idx in range(horizon):
            action_buffer.zero_()
            action_buffer[:chunk] = dataset.action_sequences[start:end, step_idx]
            env.step({"actions": action_buffer})

        final_root = env.simulator.robot_root_states[:chunk]
        final_dof = env.simulator.dof_pos[:chunk]

        base_pos_error = torch.norm(final_root[:, 0:3] - dataset.target_base_pos[start:end], dim=1)
        base_quat_error = torch.norm(final_root[:, 3:7] - dataset.target_base_ori[start:end], dim=1)
        joint_pos_error = torch.norm(final_dof - dataset.target_joint_pos[start:end], dim=1)
        
        base_pos_sum += (base_pos_error * eval_mask).sum().item()
        base_quat_sum += (base_quat_error * eval_mask).sum().item()
        joint_pos_sum += (joint_pos_error * eval_mask).sum().item()
        
        samples_processed = min(end, total_samples)
        pbar.set_postfix({
            "samples": f"{samples_processed}/{total_samples}",
            "avg_pos": f"{base_pos_sum / samples_processed:.4f}",
        })

    total_valid = (~dataset.motion_ends).sum().item()
    pbar.close()
    return MassEvalResult(
        base_pos=base_pos_sum / total_valid,
        base_quat=base_quat_sum / total_valid,
        joint_pos=joint_pos_sum / total_valid,
    )


def instantiate_env(base_dir: Path, num_envs: int, config_name: str = "env/go2_test", seed: int = 0) -> LeggedRobotBase:
    cfg = load_env_config(base_dir, num_envs, config_name, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = LeggedRobotBase(cfg.env.config, device)
    if not hasattr(env, "episode_sums"):
        env.episode_sums = {}
    env.set_is_evaluating()
    env.reset_all()
    return env
