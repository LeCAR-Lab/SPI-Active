#!/usr/bin/env python3
"""Collect jumping locomotion data."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import isaacgym  # noqa: F401
from isaacgym import gymapi
import numpy as np
import torch

from common import cleanup_env, create_env, format_data, load_config, video_enabled

OUTPUT_PATH = Path("go2_jump_data.npz")
DURATION_S = 5.0
OSC_AMPLITUDE = 1.2
OSC_FREQUENCY = 1.5
THIGH_IDX = [1, 4, 7, 10]
CALF_IDX = [2, 5, 8, 11]
BASE_HEIGHT = 0.32


def collect(env, record_video: bool) -> Tuple[Dict[str, np.ndarray], Optional[Path]]:
    """Collect jumping data."""
    dt = env.dt
    steps = int(round(DURATION_S / dt))
    sim = env.simulator
    
    is_recording = video_enabled(env, record_video)
    if record_video and not is_recording:
        print("Viewer not available; skipping video.")
    
    if is_recording:
        sim.user_is_recording = True
        sim.user_recording_state_change = True

    frames: List[Dict[str, np.ndarray]] = []
    for step in range(steps):
        t = step * dt
        phase = OSC_AMPLITUDE * np.sin(2 * np.pi * OSC_FREQUENCY * t)
        
        actions = torch.zeros((env.num_envs, env.dim_actions), device=env.device)
        actions[:, THIGH_IDX] = phase
        actions[:, CALF_IDX] = -phase

        env.step({"actions": actions})

        if is_recording:
            base = sim.robot_root_states[0, 0:3].cpu().numpy()
            cam_pos = gymapi.Vec3(*(base + np.array([1.5, 1.5, 1.0])))
            cam_target = gymapi.Vec3(*base)
            sim.gym.viewer_camera_look_at(sim.viewer, None, cam_pos, cam_target)

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
            "actions": actions[0].cpu().numpy(),
            "base_positions": sim.robot_root_states[0, 0:3].cpu().numpy(),
            "base_orientations": sim.robot_root_states[0, 3:7].cpu().numpy(),
            "base_linear_velocities": sim.robot_root_states[0, 7:10].cpu().numpy(),
            "base_angular_velocities": sim.robot_root_states[0, 10:13].cpu().numpy(),
        })

    video_path: Optional[Path] = None
    if is_recording:
        sim.user_is_recording = False
        sim.user_recording_state_change = True
        env.render(sync_frame_time=False)
        if hasattr(sim, "curr_user_recording_name"):
            video_path = Path(sim.curr_user_recording_name).with_suffix(".mp4")

    return format_data(frames, env, dt), video_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    record_video = not args.no_video

    cfg = load_config(headless=not record_video)
    env = create_env(cfg, base_height=BASE_HEIGHT)

    print(f"Collecting {DURATION_S}s jumping data at {OSC_FREQUENCY:.1f} Hz")
    print(f"Amplitude: {OSC_AMPLITUDE:.2f} rad, Control: {1/env.dt:.0f} Hz")
    
    data, video_path = collect(env, record_video)
    np.savez(OUTPUT_PATH, **data)
    
    print(f"Saved {len(data['timestamps'])} samples to {OUTPUT_PATH}")
    print(f"PD gains - Kp: {data['pd_gain_kp'][0]:.1f}, Kd: {data['pd_gain_kd'][0]:.1f}")
    if video_path:
        print(f"Video: {video_path}")
    
    cleanup_env(env)


if __name__ == "__main__":
    main()
