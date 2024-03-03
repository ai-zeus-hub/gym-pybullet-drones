import numpy as np
import torch
import torch.distributions as D

from gymnasium import spaces

from gym_pybullet_drones.agents.WaypointDroneAgent import WaypointDroneAgent
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def scale_time(t, a: float = 1.0):
    return t / (1 + 1/(a*torch.abs(t)))


def lemniscate(t: np.array, c) -> np.array:
    """
    t: array of time points [0, ...] -- loops at 2*pi

    returns: ndarray of shape (len(t), 3)
    """
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    sin2p1 = torch.square(sin_t) + 1
    x = torch.stack([cos_t, sin_t * cos_t, c * sin_t], dim=-1) / sin2p1.unsqueeze(-1)
    return x.numpy()

# def circle(t, height_scale: float = 1.):
#     cos_t = torch.cos(t)
#     sin_t = torch.sin(t)
#     z = 1 * height_scale
#     points = torch.stack([cos_t, sin_t, z], dim=-1)
#     return points.numpy()
#
# def compute_traj(self, steps: int, step_size: float = 1., func = circle):
#     t = self.step_counter + step_size * torch.arange(steps)
#     t = self.traj_t0 + scale_time(t * self.PYB_TIMESTEP)
#     target_pos = func(t, self.traj_c)
#     return self.INIT_XYZS[0] + target_pos

def circle(control_freq_hz, period = 6, height = 1.0, radius = 0.3):
    num_drones = 1
    H_STEP = .05  # height difference between drones

    init_xyzs = np.array([[radius*np.cos((i/6)*2*np.pi+np.pi/2),
                           radius*np.sin((i/6)*2*np.pi+np.pi/2)-radius,
                           height+i*H_STEP] for i in range(num_drones)])
    init_rpys = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    # Initialize a circular trajectory
    num_wp = control_freq_hz * period
    waypoints = np.zeros((num_wp,3))
    for i in range(num_wp):
        waypoints[i, :] = (radius*np.cos((i/num_wp)*(2*np.pi)+np.pi/2)+init_xyzs[0, 0],
                           radius*np.sin((i/num_wp)*(2*np.pi)+np.pi/2)-radius+init_xyzs[0, 1],
                           height)
    return init_xyzs[0], init_rpys[0], waypoints

class TrackAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 120,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 episode_len: int = 8
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thrust and torques, or waypoint with PID control)

        """
        self.future_traj_steps = 1

        self.EPISODE_LEN_SEC = episode_len
        self.traj_c_dist = D.Uniform(torch.tensor(-0.6), torch.tensor(0.6))
        self.traj_scale_dist = D.Uniform(torch.tensor([1.8, 1.8, 1.]), torch.tensor([3.2, 3.2, 1.5]))
        self.traj_w_dist = D.Uniform(torch.tensor(0.8), torch.tensor(1.1))
        self.traj_t0 = np.pi / 2  # t0 on a unit circle -- radians

        num_tracking_drones = 1
        self.traj_c = np.zeros((num_tracking_drones,))
        self.traj_scale = torch.zeros((num_tracking_drones, 3))
        self.traj_w = torch.ones((num_tracking_drones,))

        self.target_pos = np.zeros((num_tracking_drones, self.future_traj_steps, 3))

        initial_xyzs = np.array([[.25, .25, 1.]])  # Start from a hover

        xyzs, rpys, waypoints = circle(ctrl_freq)
        tracked_drone = WaypointDroneAgent(initial_xyz=xyzs,
                                           initial_rpy=rpys,
                                           pyb_freq=pyb_freq,
                                           ctrl_freq=ctrl_freq,
                                           waypoints=waypoints,
                                           episode_length_sec=episode_len)

        super().__init__(drone_model=drone_model,
                         num_drones=num_tracking_drones,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         external_agents=[tracked_drone]
                         )

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value for the drone doing the tracking

        Returns
        -------
        float
            The reward.
        """
        reward_distance_scale = 1.2
        distance = self.distance_from_next_target()
        reward_pose = np.exp(-distance * reward_distance_scale)
        reward = reward_pose
        return reward

    ################################################################################

    def target_waypoint(self):
        target_state = self.EXTERNAL_AGENTS[0].stateVector()
        target_waypoint = target_state[0:3]
        return target_waypoint

    def distance_from_next_target(self) -> float:
        """Distance from tracking to tracked drone
        """
        state = self._getDroneStateVector(0)
        # target_waypoint = self.target_pos[0]
        # distance_to_waypoint = np.linalg.norm(target_waypoint - state[0:3])
        # return distance_to_waypoint

        distance_to_waypoint = np.linalg.norm(self.target_waypoint() - state[0:3])
        return distance_to_waypoint


    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if ((self.distance_from_next_target() > 4.) or       # Truncate when the drone is too far away
            (abs(state[7]) > .4) or
            (abs(state[8]) > .4)     # Truncate when the drone is too tilted
        ):
            return True
        return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years

    def reset(self,
              seed : int = None,
              options : dict = None):
        self.traj_c = self.traj_c_dist.sample()
        # self.traj_rot = euler_to_quaternion(self.traj_rpy_dist.sample())
        self.traj_scale = self.traj_scale_dist.sample()
        traj_w = self.traj_w_dist.sample()
        self.traj_w = torch.randn_like(traj_w).sign() * traj_w
        return super().reset(seed, options)

    def _computeObs(self):
        base_action_size = 12 + (self.future_traj_steps * 3)
        obs = np.zeros((self.NUM_DRONES, base_action_size))
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            # For each drone, obs12 will be:
            #   0-2: x, y, z
            #   3-5: roll, pitch, yaw
            #   6-8: vx, vy, vz
            #  9-11: wx, wy, wz
            pos = state[0:3]
            # self.target_pos[:] = self._compute_traj(self.future_traj_steps)  # step_size=5
            # tpos = self.target_pos
            tpos = self.target_waypoint()
            rpos = tpos - pos
            obs[i, :] = np.hstack([pos, state[7:10], state[10:13], state[13:16], rpos.flatten()]).reshape(base_action_size,)
        ret = np.array([obs[i, :] for i in range(self.NUM_DRONES)]).astype('float32')

        #### Add action buffer to observation #######################
        for i in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
        return ret

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        ############################################################
        #### OBS SPACE OF SIZE 12
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo,lo,lo, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])

        for _ in range(self.future_traj_steps):
            obs_lower_bound = np.hstack([obs_lower_bound, np.array([[lo, lo, lo]])])
            obs_upper_bound = np.hstack([obs_upper_bound, np.array([[hi, hi, hi]])])

        #### Add action buffer to observation space ################
        act_lo = -1
        act_hi = +1
        for i in range(self.ACTION_BUFFER_SIZE):
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL, ActionType.DISC_RPM]:
                obs_lower_bound = np.hstack(
                    [obs_lower_bound, np.array([[act_lo, act_lo, act_lo, act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack(
                    [obs_upper_bound, np.array([[act_hi, act_hi, act_hi, act_hi] for i in range(self.NUM_DRONES)])])
            elif self.ACT_TYPE == ActionType.PID:
                obs_lower_bound = np.hstack(
                    [obs_lower_bound, np.array([[act_lo, act_lo, act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack(
                    [obs_upper_bound, np.array([[act_hi, act_hi, act_hi] for i in range(self.NUM_DRONES)])])
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _compute_traj(self, steps: int):
        t = self.step_counter + torch.arange(steps)  # * step_size (=1. default)
        t = self.traj_t0 + scale_time(self.traj_w * t * self.PYB_TIMESTEP)
        target_pos = lemniscate(t, self.traj_c)
        return self.INIT_XYZS[0] + target_pos
