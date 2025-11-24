#!/usr/bin/env python3
"""Mass parameter optimization using Optuna."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import isaacgym  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

from scripts import eval as mass_eval
from scripts.mass_landscape import load_config

# Environment configuration
CONFIG_NAME = "env/go2_test"
SEED = 0

# Mass search range
MASS_SCALE_MIN = 0.5
MASS_SCALE_MAX = 2.0

# Hyperparameters
COST_COEFF = {
    "base_pos": 10.0,      # Weight for base position error
    "base_quat": 5.0,     # Weight for base orientation error
    "joint_pos": 1.0,     # Weight for joint position error
}

# Optimization settings
INITIAL_MASS_SCALE = 3.0
N_TRIALS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mass parameter optimization")
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


def compute_cost(costs: np.ndarray) -> float:
    """
    Compute weighted prediction cost.
    
    Args:
        costs: Array with [base_pos, base_quat, joint_pos] errors
        
    Returns:
        Weighted sum of costs
    """
    return (
        costs[0] * COST_COEFF["base_pos"]
        + costs[1] * COST_COEFF["base_quat"]
        + costs[2] * COST_COEFF["joint_pos"]
    )


def plot_optimization_results(study: optuna.Study, base_nominal: float, output_dir: Path) -> None:
    """
    Plot optimization trajectory and mass landscape.
    
    Args:
        study: Optuna study object
        base_nominal: Nominal base mass in kg
        output_dir: Directory to save plots
    """
    trials = study.trials
    trial_numbers = [t.number + 1 for t in trials]
    mass_scales = [t.params["mass_scale"] for t in trials]
    costs = [t.value for t in trials]
    base_masses = [m * base_nominal for m in mass_scales]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Optimization trajectory
    ax1 = axes[0]
    ax1.plot(trial_numbers, costs, marker='o', linestyle='-', linewidth=1.5, markersize=5, alpha=0.7)
    
    # Mark best trial
    best_idx = np.argmin(costs)
    ax1.plot(trial_numbers[best_idx], costs[best_idx], 'r*', markersize=15, label='Best trial')
    
    ax1.set_xlabel("Trial number")
    ax1.set_ylabel("Cost")
    ax1.set_title("Optimization Trajectory")
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend()
    
    # Plot 2: Sampled mass landscape
    ax2 = axes[1]
    scatter = ax2.scatter(base_masses, costs, c=trial_numbers, cmap='viridis', 
                         s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Mark best point
    ax2.plot(base_masses[best_idx], costs[best_idx], 'r*', markersize=15, label='Best mass')
    
    ax2.set_xlabel("Base mass (kg)")
    ax2.set_ylabel("Cost")
    ax2.set_title("Sampled Mass Landscape")
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Trial number")
    
    fig.tight_layout()
    
    path = output_dir / "optimization_results.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def evaluate_mass_scale(
    mass_scale: float,
    env: mass_eval.LeggedRobotBase,
    dataset: mass_eval.DatasetBatch,
    horizon: int,
    reference_masses: np.ndarray,
    action_buffer: torch.Tensor,
    return_details: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Evaluate a single mass scale and return weighted cost.
    
    Args:
        mass_scale: Scale factor for base mass
        env: Environment instance
        dataset: Dataset batch
        horizon: Rollout horizon
        reference_masses: Reference mass array
        action_buffer: Action buffer tensor
        return_details: If True, return (total_cost, individual_costs)
        
    Returns:
        If return_details=False: total weighted cost
        If return_details=True: (total_cost, individual_costs_array)
    """
    base_nominal = float(reference_masses[0])
    mass_eval.apply_base_mass(env.simulator, reference_masses, base_nominal * mass_scale)
    
    outcome = mass_eval.evaluate_batch(env, horizon, action_buffer, dataset)
    costs = np.array([outcome.base_pos, outcome.base_quat, outcome.joint_pos])
    total_cost = compute_cost(costs)
    
    if return_details:
        return total_cost, costs
    return total_cost


def main() -> None:
    args = parse_args()
    
    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    data_paths = load_config(args.config)
    num_samples, dataset_np = mass_eval.load_dataset(data_paths, args.horizon)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.project_dir / f"mass_opt_{args.config}" / f"{timestamp}_h{args.horizon}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = min(args.env_batch, num_samples, 8192)
    env = mass_eval.instantiate_env(run_dir, batch_size, CONFIG_NAME, SEED)
    dataset = mass_eval.to_device(dataset_np, env.device)
    ref_masses = mass_eval.capture_reference_masses(env.simulator)
    
    base_nominal = float(ref_masses[0])
    total_nominal = float(ref_masses.sum())
    
    print(f"Optimizing: {args.config}, {N_TRIALS} trials, horizon={args.horizon}")
    print(f"Nominal: {base_nominal:.3f} kg, Output: {run_dir}\n")
    
    # Prepare action buffer
    action_buffer = torch.zeros((env.num_envs, env.dim_actions), device=env.device)
    
    # Define objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        # Suggest mass_scale
        mass_scale = trial.suggest_float(
            "mass_scale",
            MASS_SCALE_MIN,
            MASS_SCALE_MAX,
        )
        
        cost = evaluate_mass_scale(mass_scale, env, dataset, args.horizon, ref_masses, action_buffer)
        return cost
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial({"mass_scale": INITIAL_MASS_SCALE})
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    # Get best results
    best_mass_scale = study.best_params["mass_scale"]
    best_cost = study.best_value
    
    # Re-evaluate best solution to get detailed cost breakdown
    _, best_costs = evaluate_mass_scale(
        best_mass_scale, env, dataset, args.horizon, ref_masses, action_buffer, return_details=True
    )
    
    # Analyze cost components
    cost_contributions = np.array([
        best_costs[0] * COST_COEFF["base_pos"],
        best_costs[1] * COST_COEFF["base_quat"],
        best_costs[2] * COST_COEFF["joint_pos"]
    ])
    cost_percentages = 100 * cost_contributions / cost_contributions.sum()
    
    # Report results
    best_mass = base_nominal * best_mass_scale
    total_mass = best_mass + (total_nominal - base_nominal)
    
    print(f"\nOptimal: scale={best_mass_scale:.4f}, base={best_mass:.3f} kg, cost={best_cost:.6f}")
    print(f"Breakdown: pos {cost_percentages[0]:.0f}%, quat {cost_percentages[1]:.0f}%, joint {cost_percentages[2]:.0f}%")
    
    # Save text results
    results_path = run_dir / "optimization_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Mass Optimization Results (Optuna)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Horizon: {args.horizon}\n")
        f.write(f"Cost coefficients: {COST_COEFF}\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"\n")
        f.write(f"Nominal base mass: {base_nominal:.3f} kg\n")
        f.write(f"Optimal base mass: {best_mass:.3f} kg\n")
        f.write(f"Best cost: {best_cost:.6f}\n")
    
    plot_optimization_results(study, base_nominal, run_dir)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()

