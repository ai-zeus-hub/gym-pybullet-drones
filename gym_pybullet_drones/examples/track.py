import time
import argparse
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.policies import NatureCNN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.TrackAviary import TrackAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DepthType

from gym_pybullet_drones.bullet_track.bullet_track_extractor import BulletTrackCombinedExtractor, BulletTrackEfficientNet, TransformerExtractor, BulletTrackYOLO
from gym_pybullet_drones.bullet_track.bullet_track_policy import BulletTrackPolicy

DEFAULT_DEPTH_TYPE = DepthType.IMAGE
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = Path('final_results')
DEFAULT_SAVE_EVAL_IMAGE = True
DEFAULT_RL_ALGO = "PPO"
DEFAULT_PRETRAINED_PATH = Path()
DEFAULT_EPISODE_LEN = 8  # usually 8
MAX_LR = 0.0005
DEFAULT_IMAGE_EXTRACTOR = BulletTrackEfficientNet

DEFAULT_OBS = ObservationType.MULTI
DEFAULT_SUPER_MODE = True  # If True, rpos to target will be in the observation space
DEFAULT_ACT = ActionType.RPM


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


def run(output_folder=DEFAULT_OUTPUT_FOLDER, rl_algo=DEFAULT_RL_ALGO, gui=DEFAULT_GUI,
        save_eval_image=DEFAULT_SAVE_EVAL_IMAGE, record_video=DEFAULT_RECORD_VIDEO,
        pretrained=DEFAULT_PRETRAINED_PATH, super_mode=DEFAULT_SUPER_MODE, episode_len=DEFAULT_EPISODE_LEN):
    action_str = str(DEFAULT_ACT).split('.')[1]
    if DEFAULT_OBS == ObservationType.MULTI:
        obs_str = "RGBD-Kinematics"
    elif DEFAULT_OBS == ObservationType.RGB:
        obs_str = "RGBD"
    else:
        raise ValueError
    if super_mode:
        obs_str += "-RPOS"
    image_feature_extractor = DEFAULT_IMAGE_EXTRACTOR
    description = f"{image_feature_extractor.__name__}"
    filename = Path(output_folder) / obs_str / action_str / description

    if not filename.exists():
        filename.mkdir(parents=True)

    train_env = make_vec_env(TrackAviary,
                             env_kwargs=dict(obs=DEFAULT_OBS,
                                             act=DEFAULT_ACT,
                                             episode_len=episode_len,
                                             include_rpos_in_obs=super_mode,
                                             output_folder=filename),
                             n_envs=1,
                             seed=0)
    eval_env = TrackAviary(obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           episode_len=episode_len,
                           include_rpos_in_obs=super_mode,
                           output_folder=filename)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    ### Train the model #######################################
    net_arch = [256, 256, 256]
    # 0 for flatten after CNN, 1+ for linear. It's purpose is to
    # decouple the cnn and MLP network, for easier loading
    feature_dims = 0
    # This is directly after the CNN and needs to stay the same between
    # save/load (if starting with initial weights)
    cnn_features = 128
    features_extractor_kwargs = dict(image_feature_extractor=image_feature_extractor,
                                     cnn_output_dim=cnn_features,
                                     feature_dims=feature_dims)
    # features_extractor_kwargs = dict()
    policy_kwargs = dict(net_arch=net_arch,
                         share_features_extractor=True,
                         features_extractor_class=BulletTrackCombinedExtractor,  # TransformerExtractor
                         features_extractor_kwargs=features_extractor_kwargs)

    run_description = "_".join([obs_str, action_str, description])

    if rl_algo == "PPO":
        model = PPO(BulletTrackPolicy,
                    train_env,
                    tensorboard_log=str(output_folder / 'tensorboard'),
                    verbose=1,
                    seed=10281991,
                    vf_coef=1.25,
                    learning_rate=constant_lr_schedule,
                    n_epochs=4,
                    ent_coef=0.001,
                    max_grad_norm=10.0,
                    policy_kwargs=policy_kwargs)
        model_cls = PPO
    else:
        raise ValueError(f"Unsupported rl algo: {rl_algo}")

    if pretrained.is_file():
        print(f"Loading from {str(pretrained)}")
        old_model = model_cls.load(pretrained)
        model.policy.load_from_policy(old_model.policy)

    #### Target cumulative rewards (problem-dependent) ##########
    max_reward_per_step = 1.5
    target_reward = max_reward_per_step * episode_len * eval_env.CTRL_FREQ * 0.95
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=str(filename),
                                 log_path=str(filename),
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=450_000,
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

    if (filename / 'best_model.zip').is_file():
        path = filename / 'best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
        exit()
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    test_env = TrackAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, episode_len=episode_len,
                           record=record_video, include_rpos_in_obs=super_mode, output_folder=filename,
                           static_idx=6)
    test_env_nogui = TrackAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT,
                                 episode_len=episode_len, include_rpos_in_obs=super_mode, output_folder=filename,
                                 static_idx=6)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ), num_drones=1,
                    output_folder=str(output_folder), colab=save_eval_image)

    # mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    # print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
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
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs, info = test_env.reset(seed=42, options={})
    test_env.close()

    logger.plot(output_folder=filename)

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',             default=DEFAULT_GUI,             type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',    default=DEFAULT_RECORD_VIDEO,    type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',   default=DEFAULT_OUTPUT_FOLDER,   type=Path,     help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--rl-algo',         default=DEFAULT_RL_ALGO,         type=str,      help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--pretrained',      default=DEFAULT_PRETRAINED_PATH, type=Path,     help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--super-mode',      default=DEFAULT_SUPER_MODE,      type=str2bool, help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--save-eval-image', default=DEFAULT_SAVE_EVAL_IMAGE, type=bool,     help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
