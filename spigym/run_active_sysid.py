"""
Entry-point to run Active SysID command optimization.

This script expects a trained locomotion policy checkpoint and the Active
SysID Hydra configs. It instantiates the ActiveSysId environment, loads the
policy, and launches the Optuna-driven optimization to find informative
command sequences.

Typical usage: see `spigym/scripts/active_sysid.sh`.
"""

import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from spigym.utils.logging import HydraLoggerBridge
import logging
from utils.config_utils import *  # noqa: E402, F403

# add argparse arguments

from spigym.utils.config_utils import *  # noqa: E402, F403
from loguru import logger

import threading
from pynput import keyboard

def on_press(key, env):
    try:
        if key.char == 'n':
            env.next_task()
            logger.info("Moved to the next task.")
    except AttributeError:
        pass

def listen_for_keypress(env):
    with keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
        listener.join()


# from spigym.envs.base_task.base_task import BaseTask
# from spigym.envs.base_task.omnih2o_cfg import OmniH2OCfg

@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    
    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())
    
    

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config
            
    simulator_type = config.simulator['_target_'].split('.')[-1]
    
    if simulator_type != 'IsaacGymActiveSysId':
        raise NotImplementedError("Only IsaacGym simulator customed for System Identification is supported for training.")
    import isaacgym  # noqa: F401
        
    from spigym.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from spigym.utils.helpers import pre_process_config
    import torch

    
    
    pre_process_config(config)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    eval_log_dir = Path(config.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    with open(eval_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)
    
    ckpt_num = config.checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")
    config.env.config.ckpt_dir = str(checkpoint.parent) # commented out for now, might need it back to save motion

    # Override rewards to focus on the SysID objective during optimization
    config.rewards.reward_scales = {"fisher_information_matrix": 1.0}
    env = instantiate(config.env, device=device)
    
    # Start a thread to listen for key press
    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()

    experiment_save_dir = Path(config.experiment_dir)
    
    experiment_save_dir.mkdir(exist_ok=True, parents=True)
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    logger.info(f"Saving config file to {experiment_save_dir}")
    with open(experiment_save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=experiment_save_dir)
    algo.setup()
    algo.load(config.checkpoint)


    

    

    algo.optimize()


if __name__ == "__main__":
    main()
