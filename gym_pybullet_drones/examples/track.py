import os
import time
import argparse
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.TrackAviary import TrackAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = True

DEFAULT_OBS = ObservationType.MULTI
DEFAULT_ACT = ActionType('rpm')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False
DEFAULT_EPISODE_LEN = 8  # usually 8

MAX_LR = 0.0005


def piecewise_lr_schedule(remaining_percent: float) -> float:  # designed for 400k
    if remaining_percent >= 0.75:
        return MAX_LR
    elif remaining_percent >= 0.5:
        return MAX_LR - 0.0001
    elif remaining_percent >= 0.25:
        return MAX_LR - 0.0002
    else:
        return MAX_LR - 0.0003


def linear_lr_schedule(remaining_percent: float) -> float:
    lr_max = MAX_LR
    lr_diff = 0.0003
    lr_min = lr_max - lr_diff
    return lr_min + remaining_percent * lr_diff


def constant_lr_schedule(remaining_percent: float) -> float:
    return MAX_LR


def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO,
        episode_len=DEFAULT_EPISODE_LEN):
    filename = Path(output_folder) / 'save-latest'
    if not filename.exists():
        filename.mkdir(parents=True)

    train_env = make_vec_env(TrackAviary,
                             env_kwargs=dict(obs=DEFAULT_OBS,
                                             act=DEFAULT_ACT,
                                             episode_len=episode_len),
                             n_envs=1,
                             seed=0)
    eval_env = TrackAviary(obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           episode_len=episode_len)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    ### Train the model #######################################
    net_arch = [256, 256, 256]
    features_extractor_kwargs = dict()  # NatureCNN
    policy_kwargs = dict(net_arch=net_arch,
                         share_features_extractor=True,
                         features_extractor_kwargs=features_extractor_kwargs)

    observation_type = DEFAULT_OBS
    # if observation_type == ObservationType.KIN:
    #     policy_type = "MlpPolicy"
    # elif observation_type == ObservationType.RGB:
    #     policy_type = "CnnPolicy"
    # else:
    policy_type = "MultiInputPolicy"
    run_description = "_".join([
        f"PPO-{str(observation_type).split('.')[1]}-depth",
        f"Action-{str(DEFAULT_ACT).split('.')[1]}"
    ])

    model = PPO(policy_type,
                train_env,
                tensorboard_log=str(filename / 'tb'),
                verbose=1,
                seed=10281991,
                vf_coef=1.25,
                learning_rate=constant_lr_schedule,
                n_epochs=4,
                ent_coef=0.001,
                max_grad_norm=10.0,
                policy_kwargs=policy_kwargs)

    #### Target cumulative rewards (problem-dependent) ##########
    target_reward = episode_len * 24 * 0.9  # 24 is ctrl frequency
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=str(filename),
                                 log_path=str(filename),
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=350_000,
                callback=eval_callback,
                log_interval=100,
                tb_log_name=run_description)

    #### Save the model ########################################
    model.save(filename / 'final_model.zip')
    print(str(filename))

    #### Print training progression ############################
    with np.load(filename / 'evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if (filename / 'best_model.zip').is_file():
        path = filename / 'best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
        exit()
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    test_env = TrackAviary(gui=gui,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           episode_len=episode_len,
                           record=record_video)
    test_env_nogui = TrackAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, episode_len=episode_len)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        # print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        logger.log(drone=0,
            timestamp=i/test_env.CTRL_FREQ,
            state=test_env._getDroneStateVector(0),
            control=np.zeros(12),
            reward=reward,
            distance=info["total_distance"],
            )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs, info = test_env.reset(seed=42, options={})
    test_env.close()

    logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
