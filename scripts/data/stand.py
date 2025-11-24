#!/usr/bin/env python3
"""Collect standing (zero action) reference data."""

from pathlib import Path
from typing import Dict, List

import isaacgym  # noqa: F401
import numpy as np
import torch

from common import cleanup_env, create_env, format_data, load_config

OUTPUT_PATH = Path("go2_stand_data.npz")
DURATION_S = 5.0


def collect(env) -> Dict[str, np.ndarray]:
    """Collect standing data with zero actions."""
    dt = env.dt
    steps = int(round(DURATION_S / dt))
    zero_actions = torch.zeros((env.num_envs, env.dim_actions), device=env.device)

    frames: List[Dict[str, np.ndarray]] = []
    for step in range(steps):
        env.step({"actions": zero_actions})
        
        sim = env.simulator
        torques = getattr(env, "torques", None)
        joint_torques = (
            torques[0].detach().cpu().numpy()
            if torques is not None
            else np.zeros_like(sim.dof_pos[0].cpu().numpy())
        )
        
        frames.append({
            "timestamp": step * dt,
            "joint_positions": sim.dof_pos[0].cpu().numpy(),
            "joint_velocities": sim.dof_vel[0].cpu().numpy(),
            "joint_torques": joint_torques,
            "actions": zero_actions[0].cpu().numpy(),
            "base_positions": sim.robot_root_states[0, 0:3].cpu().numpy(),
            "base_orientations": sim.robot_root_states[0, 3:7].cpu().numpy(),
            "base_linear_velocities": sim.robot_root_states[0, 7:10].cpu().numpy(),
            "base_angular_velocities": sim.robot_root_states[0, 10:13].cpu().numpy(),
        })

    return format_data(frames, env, dt)


def main() -> None:
    cfg = load_config(headless=True)
    env = create_env(cfg)

    print(f"Collecting {DURATION_S}s standing data at {1/env.dt:.0f} Hz")
    
    data = collect(env)
    np.savez(OUTPUT_PATH, **data)
    
    print(f"Saved {len(data['timestamps'])} samples to {OUTPUT_PATH}")
    print(f"PD gains - Kp: {data['pd_gain_kp'][0]:.1f}, Kd: {data['pd_gain_kd'][0]:.1f}")
    
    cleanup_env(env)


if __name__ == "__main__":
    main()
