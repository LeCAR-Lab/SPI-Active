################################################################################################
# Code for plotting
################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
from plotly.io import write_image

import matplotlib

matplotlib.use('Agg')



def plot_traj_len_dist(motion_data, log_dir):
    len_list = []
    for key in motion_data.keys():
        for traj in motion_data[key]:
            len_list.append(len(traj['dof']))
    fig, ax = plt.subplots(1, 2, figsize=(25, 8))
    ax[0].hist(len_list, bins=50)
    ax[0].set_title("Trajectory Length Distribution")
    ax[0].set_xlabel("Length")
    ax[0].set_ylabel("Frequency")
    
    # portion of each motion
    motion_steps = []
    for key in motion_data.keys():
        total_steps = 0
        for traj in motion_data[key]:
            total_steps += len(traj['root_rot'])
        motion_steps.append(total_steps)
    ax[1].bar(motion_data.keys(), motion_steps)
    ax[1].set_title("Motion Distribution")
    ax[1].set_xlabel("Motion")
    ax[1].set_ylabel("Total Steps")
    
    plt.savefig(os.path.join(log_dir, "traj_len_dist.png"))
    plt.close(fig)

    

def calculate_plot_y_range(values, quantile=0.8, scale=1.2):
    """
    Calculate y-axis range based on a quantile of the study data.
    
    Args:
        study: optuna.Study object containing trials
        quantile: float between 0 and 1 indicating the quantile to use (default: 0.75)
    
    Returns:
        tuple: (min_y, max_y) range for the y-axis
    """    
    values = sorted(values)
    q_max_index = int(len(values) * quantile)
    q_max_value = values[q_max_index] if q_max_index < len(values) else values[-1]
    q_min_index = 0
    q_min_value = values[q_min_index] if q_min_index < len(values) else values[0]
    
    val_delta = q_max_value - q_min_value
    
    s = (scale - 1) / 2
    
    return (q_min_value - val_delta * s, q_max_value + val_delta * s)


def plot_optimization(opt_study, total_params, batch_results, log_dir):
    fig = optuna.visualization.plot_slice(opt_study, params=total_params)
    
    # Set ylimit to cover from 0 to the 3/4 point of the data
    values = [trial.value for trial in opt_study.trials if trial.value is not None]
    y_min, y_max = calculate_plot_y_range(values)
    fig.update_layout(yaxis=dict(range=[y_min, y_max]))
    
    write_image(fig, os.path.join(log_dir, "opt_slice.png"))
    
    fig = optuna.visualization.plot_optimization_history(opt_study)
    fig.update_layout(yaxis=dict(range=[y_min, y_max]))
    
    write_image(fig, os.path.join(log_dir, "opt_history.png"))
    
    params = []
    for r in batch_results:
        for t in r[0]:
            params.append([t.params[key] for key in total_params])
    
    params = np.array(params)        
    rewards = {key: np.concatenate([batch_results[i][1][key] for i in range(len(batch_results))], axis=0) for key in batch_results[0][1].keys()}
    
    
    fig, axs = plt.subplots(len(total_params), len(rewards), figsize=(len(rewards)*8, len(total_params)*5))
    
    # Handle the case when there's only one parameter or one reward
    if len(total_params) == 1 and len(rewards) == 1:
        axs = [[axs]]
    elif len(total_params) == 1:
        axs = [axs]
    elif len(rewards) == 1:
        axs = [[ax] for ax in axs]
    
    for i, key in enumerate(rewards.keys()):
        for j, param in enumerate(total_params):
            y_min, y_max = calculate_plot_y_range(-rewards[key])
            axs[j][i].scatter(params[:, j], -rewards[key])
            axs[j][i].set_title(key)
            axs[j][i].set_xlabel(param)
            axs[j][i].set_ylabel("Cost")
            axs[j][i].grid()
            axs[j][i].set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "opt_results.png"))
    plt.close(fig)
    # plot landscape of each rewards
    
