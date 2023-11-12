"""Script demonstrating the use of `gym_pybullet_drones` Gymnasium interface.

Class TrackingAviary is used as a learning env for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python tracking.py
"""
import argparse
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.multi_agent_rl.TrackingAviary import TrackingAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import str2bool, sync

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False


def run(
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    gui: bool = DEFAULT_GUI,
    plot: bool = True,
    colab: bool = DEFAULT_COLAB,
    record_video: bool = DEFAULT_RECORD_VIDEO,
):
    # Check the environment's spaces
    env = gym.make("tracking-aviary-v0")
    print("[INFO] Beginning tracking algorithm")
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    # Train the model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)  # Typically not enough

    # Show (and record a video of) the model's performance
    env = TrackingAviary(gui=gui, record=record_video)
    logger = Logger(
        logging_freq_hz=int(env.CTRL_FREQ),
        num_drones=2,
        output_folder=output_folder,
        colab=colab,
    )
    obs, info = env.reset(seed=42, options={})
    start = time.time()
    tracking_drone = 0
    tracked_drone = 1
    for i in range(3 * env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # For plotting purposes, we are primarily interested in the tracking drone
        obs_tracking_drone = obs[tracking_drone]
        tracking_actions_as_rpms = env._preprocessAction(action)[tracking_drone]
        logger.log(
            drone=tracking_drone,
            timestamp=i / env.CTRL_FREQ,
            state=np.hstack([obs_tracking_drone[0:3],
                             np.zeros(4),  # quaternions? ignored by logger
                             obs_tracking_drone[3:15],
                             np.resize(tracking_actions_as_rpms, 4)]),
            control=np.zeros(12),
        )

        obs_tracked_drone = obs[tracked_drone]
        tracked_actions_as_rpms = env._preprocessAction(action)[tracked_drone]
        logger.log(
            drone=tracked_drone,
            timestamp=i / env.CTRL_FREQ,
            state=np.hstack([obs_tracked_drone[0:3],
                             np.zeros(4),  # quaternions? ignored by logger
                             obs_tracked_drone[3:15],
                             np.resize(tracked_actions_as_rpms, 4)]),
            control=np.zeros(12),
        )
        env.render()
        print(f"{terminated=}")
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            obs = env.reset(seed=42, options={})
    env.close()

    if plot:
        logger.plot()


if __name__ == "__main__":
    # Define and parse (optional) arguments for the script
    parser = argparse.ArgumentParser(
        description="Multi agent reinforcement learning example script using TrackingAviary"
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VIDEO,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    args = parser.parse_args()

    run(**vars(args))
