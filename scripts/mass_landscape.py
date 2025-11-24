#!/usr/bin/env python3
"""Command-line mass landscape sweep built atop scripts.eval."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import isaacgym  # noqa: F401  # ensure bindings load before torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from scripts import eval as mass_eval

# Environment configuration
CONFIG_NAME = "env/go2_test"
SEED = 0

# Mass search range
MASS_SCALE_MIN = 0.5
MASS_SCALE_MAX = 2.0

# Landscape settings
MASS_SAMPLES = 20
MAX_SAFE_ENV_BATCH = 8192

# Hyperparameters
COST_COEFF = {
    "base_pos": 10.0,      # Weight for base position error
    "base_quat": 5.0,     # Weight for base orientation error
    "joint_pos": 1.0,     # Weight for joint position error
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mass sensitivity replay scan")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Config name (e.g., 'walk', 'all') to load from scripts/config/"
    )
    parser.add_argument(
        "--horizon", type=int, default=5, help="Rollout steps between comparisons"
    )
    parser.add_argument(
        "--env-batch", type=int, default=4096, help="Parallel env batch size"
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("logs"),
        help="Top-level directory where runs are written",
    )
    return parser.parse_args()


def load_config(config_name: str) -> list[Path]:
    """Load data paths from yaml config file."""
    config_dir = Path(__file__).resolve().parent / "config"
    config_path = config_dir / f"{config_name}.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert to Path objects, resolve relative to workspace root
    workspace_root = Path(__file__).resolve().parent.parent.parent
    return [workspace_root / path_str for path_str in config["data_paths"]]


def plot_landscape(
    masses: Iterable[float], costs: np.ndarray, horizon: int, output_dir: Path
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    labels = ["Base position L2", "Base quaternion L2", "Joint position L2", "Total Cost (Weighted)"]

    # Plot individual costs
    for idx, ax in enumerate(axes[:3]):
        ax.plot(masses, costs[:, idx], marker="o", linewidth=1.5)
        ax.set_ylabel(labels[idx])
        ax.grid(True, linestyle="--", alpha=0.4)
    
    # Plot total weighted cost
    total_cost = (costs[:, 0] * COST_COEFF["base_pos"] + 
                  costs[:, 1] * COST_COEFF["base_quat"] + 
                  costs[:, 2] * COST_COEFF["joint_pos"])
    axes[3].plot(masses, total_cost, marker="o", linewidth=1.5, color='red')
    axes[3].set_ylabel(labels[3])
    axes[3].grid(True, linestyle="--", alpha=0.4)
    
    # Mark minimum on total cost plot
    min_idx = np.argmin(total_cost)
    axes[3].plot(masses[min_idx], total_cost[min_idx], 'r*', markersize=15, 
                 label=f'Min at {masses[min_idx]:.3f} kg')
    axes[3].legend()

    axes[-1].set_xlabel("Base link mass (kg)")
    fig.suptitle(f"Mass sensitivity over a {horizon}-step horizon\n"
                 f"Weights: pos={COST_COEFF['base_pos']}, quat={COST_COEFF['base_quat']}, joint={COST_COEFF['joint_pos']}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    path = output_dir / f"mass_sensitivity_h{horizon}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mass_sweep(
    env: mass_eval.LeggedRobotBase,
    dataset: mass_eval.DatasetBatch,
    horizon: int,
    reference_masses: np.ndarray,
    mass_scales: Iterable[float],
) -> np.ndarray:
    mass_scales = np.asarray(list(mass_scales), dtype=np.float32)
    action_buffer = torch.zeros((env.num_envs, env.dim_actions), device=env.device)
    results = np.zeros((mass_scales.shape[0], 3), dtype=np.float32)
    base_nominal = float(reference_masses[0])

    for idx, scale in enumerate(mass_scales):
        mass_eval.apply_base_mass(env.simulator, reference_masses, base_nominal * float(scale))
        outcome = mass_eval.evaluate_batch(env, horizon, action_buffer, dataset)
        results[idx] = (outcome.base_pos, outcome.base_quat, outcome.joint_pos)

    return results


def main() -> None:
    args = parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data_paths = load_config(args.config)
    num_samples, dataset_np = mass_eval.load_dataset(data_paths, args.horizon)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.project_dir / f"{timestamp}_{args.config}_h{args.horizon}"
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_size = min(args.env_batch, num_samples, MAX_SAFE_ENV_BATCH)
    env = mass_eval.instantiate_env(run_dir, batch_size, CONFIG_NAME, SEED)
    dataset = mass_eval.to_device(dataset_np, env.device)
    ref_masses = mass_eval.capture_reference_masses(env.simulator)
    base_nominal = float(ref_masses[0])
    total_nominal = float(ref_masses.sum())

    mass_scales = np.linspace(MASS_SCALE_MIN, MASS_SCALE_MAX, MASS_SAMPLES)
    base_masses = base_nominal * mass_scales

    print(f"Mass landscape: {args.config}, {MASS_SAMPLES} samples, horizon={args.horizon}")
    print(f"Nominal: {base_nominal:.3f} kg, Output: {run_dir}\n")

    costs = mass_sweep(env, dataset, args.horizon, ref_masses, mass_scales)
    
    # Calculate total weighted cost for each sample
    total_costs = (costs[:, 0] * COST_COEFF["base_pos"] + 
                   costs[:, 1] * COST_COEFF["base_quat"] + 
                   costs[:, 2] * COST_COEFF["joint_pos"])
    
    # Find and report best mass
    best_idx = np.argmin(total_costs)
    best_scale = mass_scales[best_idx]
    best_base_mass = base_masses[best_idx]
    best_total_mass = best_base_mass + (total_nominal - base_nominal)
    best_cost = total_costs[best_idx]
    
    # Analyze cost components at optimum
    best_costs = costs[best_idx]
    cost_contributions = np.array([
        best_costs[0] * COST_COEFF["base_pos"],
        best_costs[1] * COST_COEFF["base_quat"],
        best_costs[2] * COST_COEFF["joint_pos"]
    ])
    cost_percentages = 100 * cost_contributions / cost_contributions.sum()
    
    print(f"\nOptimal: scale={best_scale:.4f}, base={best_base_mass:.3f} kg, cost={best_cost:.6f}")
    print(f"Breakdown: pos {cost_percentages[0]:.0f}%, quat {cost_percentages[1]:.0f}%, joint {cost_percentages[2]:.0f}%")

    # Save numerical results
    results_path = run_dir / "landscape_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Mass Landscape Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Horizon: {args.horizon}\n")
        f.write(f"Cost coefficients: {COST_COEFF}\n")
        f.write(f"Number of samples: {MASS_SAMPLES}\n")
        f.write(f"\n")
        f.write(f"Nominal base mass: {base_nominal:.3f} kg\n")
        f.write(f"Optimal base mass: {best_base_mass:.3f} kg\n")
        f.write(f"Minimum total cost: {best_cost:.6f}\n")
        f.write(f"All samples:\n")
        f.write(f"scale,base_mass_kg,total_mass_kg,base_pos,base_quat,joint_pos,total_cost\n")
        for scale, base_mass, row, total_cost in zip(mass_scales, base_masses, costs, total_costs):
            total_mass = base_mass + (total_nominal - base_nominal)
            f.write(f"{scale:.4f},{base_mass:.3f},{total_mass:.3f},{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{total_cost:.6f}\n")
    
    plot_landscape(base_masses, costs, args.horizon, run_dir)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
