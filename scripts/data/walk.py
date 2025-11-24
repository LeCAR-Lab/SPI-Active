#!/usr/bin/env python3
"""Collect Go2 walking data using unitree_rl_gym."""

import argparse
from pathlib import Path

import isaacgym  # noqa: F401
from isaacgym import gymutil
import numpy as np

from legged_gym.utils import task_registry


OUTPUT_PATH = Path("go2_walk_data.npz")
NUM_STEPS = 1000
COMMANDS = [
    (0.5, 0.0, 0.0),   # Forward 0.5 m/s
    (-0.5, 0.0, 0.0),  # Backward 0.5 m/s
    (0.0, 0.5, 0.0),   # Left 0.5 m/s
    (0.0, -0.5, 0.0),  # Right 0.5 m/s
]


def setup_env(args):
    """Setup unitree_rl_gym environment for data collection."""
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = 1000
    env_cfg.rewards.scales.termination = 0.0

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    train_cfg.runner.resume = True
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    return env, policy


def collect(env, policy, num_steps: int) -> dict:
    """Collect walking data with varying commands."""
    data = {
        "timestamps": [],
        "joint_positions": [],
        "joint_velocities": [],
        "joint_torques": [],
        "actions": [],
        "policy_observations": [],
        "base_positions": [],
        "base_orientations": [],
        "base_linear_velocities": [],
        "base_angular_velocities": [],
    }
    
    obs = env.get_observations()
    steps_per_command = num_steps // len(COMMANDS)
    
    for step in range(num_steps):
        cmd_idx = step // steps_per_command
        cmd_idx = min(cmd_idx, len(COMMANDS) - 1)
        env.commands[:, 0], env.commands[:, 1], env.commands[:, 2] = COMMANDS[cmd_idx]
        
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        
        data["timestamps"].append(step * env.dt)
        data["joint_positions"].append(env.dof_pos[0].cpu().numpy())
        data["joint_velocities"].append(env.dof_vel[0].cpu().numpy())
        data["joint_torques"].append(env.torques[0].cpu().numpy())
        data["actions"].append(actions[0].detach().cpu().numpy())
        data["policy_observations"].append(obs[0].cpu().numpy())
        data["base_positions"].append(env.root_states[0, 0:3].cpu().numpy())
        data["base_orientations"].append(env.root_states[0, 3:7].cpu().numpy())
        data["base_linear_velocities"].append(env.root_states[0, 7:10].cpu().numpy())
        data["base_angular_velocities"].append(env.root_states[0, 10:13].cpu().numpy())
        
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}")
    
    for key in data:
        data[key] = np.array(data[key])
    
    data["sim_duration"] = len(data["timestamps"]) * env.dt
    data["data_frequency"] = int(1.0 / env.dt)
    data["robot_type"] = "go2"
    data["pd_gain_kp"] = env.p_gains[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
    data["pd_gain_kd"] = env.d_gains[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
    
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="go2")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    
    args = gymutil.parse_arguments(
        description="Collect Go2 walking data",
        custom_parameters=[
            {"name": name, **params} 
            for name, params in vars(parser.parse_args([])).items()
        ]
    )
    
    print("="*60)
    print("Go2 Walking Data Collection")
    print(f"Steps: {args.num_steps}")
    print("="*60)
    
    env, policy = setup_env(args)
    data = collect(env, policy, args.num_steps)
    
    output = Path(args.output)
    np.savez(output, **data)
    
    print(f"\nSaved {len(data['timestamps'])} samples to {output}")
    print(f"PD gains - Kp: {data['pd_gain_kp'][0]:.1f}, Kd: {data['pd_gain_kd'][0]:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()
