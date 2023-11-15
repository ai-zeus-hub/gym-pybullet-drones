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

from gym_pybullet_drones.agents.DroneAgent import DroneAgent
from gym_pybullet_drones.envs.single_agent_rl.SingleTrackingAviary import SingleTrackingAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import str2bool, sync

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False


def calculateWaypoints(control_freq_hz):
    num_drones = 1
    H = .1        # height
    H_STEP = .05  # height difference between drones
    R = .3        # radius
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    # Initialize a circular trajectory
    PERIOD = 10  # Time for 1 full circle
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    return INIT_XYZS[0], INIT_RPYS[0], TARGET_POS


def agent_to_logger_state(drone: DroneAgent):
    state = np.zeros(20)
    state[0:3] = drone.kinematics.pos
    state[10:13] = drone.kinematics.vel
    state[7:10] = drone.kinematics.rpy
    state[13:16] = drone.kinematics.ang_v
    state[16:20] = drone.last_action


    # This is the internal logger mapping
    # state[0:3] = drone.kinematics.pos
    # state[3:6] = drone.kinematics.vel
    # state[6:9] = drone.kinematics.rpy
    # state[9:12] = drone.kinematics.ang_v
    # state[12:16] = drone.last_action # rpms

    return state

def run(
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    gui: bool = DEFAULT_GUI,
    plot: bool = True,
    colab: bool = DEFAULT_COLAB,
    record_video: bool = DEFAULT_RECORD_VIDEO,
):
    sim_freq = 240
    ctrl_freq = 48
    tracking_drone = 0
    initial_xyz, initial_rpy, waypoints = calculateWaypoints(ctrl_freq)

    # Check the environment's spaces
    tracking_init_pos = np.array([[1, 1, 0.5]])
    env = gym.make("single-tracking-aviary-v0",
                   initial_xyzs=tracking_init_pos,
                   target_initial_xyz=initial_xyz,
                   target_initial_rpy=initial_rpy,
                   target_waypoints=waypoints,
                   pyb_freq=sim_freq,
                   ctrl_freq=ctrl_freq)
    print("[INFO] Beginning tracking algorithm")
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    # Train the model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)  # Typically not enough

    # Show (and record a video of) the model's performance
    env = SingleTrackingAviary(gui=gui, record=record_video,
                               initial_xyzs=tracking_init_pos,
                               target_initial_xyz=initial_xyz,
                               target_initial_rpy=initial_rpy,
                               target_waypoints=waypoints,
                               pyb_freq=sim_freq,
                               ctrl_freq=ctrl_freq
                               )
    logger = Logger(
        duration_sec=5,  # time for now
        logging_freq_hz=int(env.CTRL_FREQ),
        num_drones=2,
        output_folder=output_folder,
        colab=colab,
    )
    obs, info = env.reset(seed=42, options={})
    start = time.time()
    for i in range(5 * env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # For plotting purposes, we are primarily interested in the tracking drone
        tracking_actions_as_rpms = env._preprocessAction(action)[tracking_drone]
        logger.log(drone=tracking_drone,
                   timestamp=i / env.CTRL_FREQ,
                   state=np.hstack([obs[0:3],
                                    np.zeros(4),  # quaternions? ignored by logger
                                    obs[3:15],
                                    np.resize(tracking_actions_as_rpms, 4)]),
                   control=np.zeros(12),
                   )

        logger.log(drone=1,
                   timestamp=i / env.CTRL_FREQ,
                   state=agent_to_logger_state(env.EXTERNAL_AGENTS[0]),
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
