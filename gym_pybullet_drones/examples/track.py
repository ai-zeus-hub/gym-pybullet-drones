import os
import time
import argparse
import numpy as np
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
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False
DEFAULT_EPISODE_LEN = 8  # usually 8

MAX_LR = 0.0006


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
        local=True, episode_len=DEFAULT_EPISODE_LEN):
    filename = os.path.join(output_folder, 'save-latest')
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

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
    arch = [256, 256, 256]
    policy_kwargs = dict(net_arch=arch,
                         share_features_extractor=False)

    # ActorCriticPolicy
    run_description = " ".join([
        f"PPO",
        f"Action={str(DEFAULT_ACT).split('.')[1]}",
        f"Lemniscate",
        f"ActionBuffer=1",
        f"FutureSteps=1",
        f"{arch=}",
        f"lr=const@{MAX_LR=}"
    ])

    model = PPO('MlpPolicy',
                train_env,
                tensorboard_log=filename+'/tb/',
                verbose=1,
                seed=10281991,
                # clip_range=0.20,  # 0.1 will be slower but more steady. 0.2 default
                vf_coef=1.25,
                # omni settings
                # n_steps=64,
                # batch_size=16,
                learning_rate=constant_lr_schedule,
                n_epochs=4,
                ent_coef=0.001,
                max_grad_norm=10.0,
                policy_kwargs=policy_kwargs)

    #### Target cumulative rewards (problem-dependent) ##########
    target_reward = 2_000  # 467. * 4 if not multiagent else 920.  # 467.
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=750_000,  # 750_000, # int(1e6), # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=100,
                tb_log_name=run_description)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
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
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        # print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(drone=0,
                timestamp=i/test_env.CTRL_FREQ,
                state=np.hstack([obs2[0:3],
                                    np.zeros(4),
                                    obs2[3:15],
                                    act2 # todo: ajr - np.resize(action, (4))? reward=reward
                                    ]),
                control=np.zeros(12),
                reward=reward,
                distance=test_env.distance_from_next_target(),
                )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs, info = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
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
