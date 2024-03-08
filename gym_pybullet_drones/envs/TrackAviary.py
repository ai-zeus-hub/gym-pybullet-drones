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


def circle(control_freq_hz, period=6, height=1.0, radius=0.3):
    init_xyzs = np.array([radius*np.cos(np.pi/2),
                          radius*np.sin(np.pi/2) - radius,
                          height])
    init_rpys = np.array([0, 0, 0])

    # Initialize a circular trajectory
    num_wp = int(control_freq_hz * period)
    waypoints = np.zeros((num_wp, 3))
    for i in range(num_wp):
        waypoints[i, :] = (radius*np.cos((i/num_wp)*(2*np.pi)+np.pi/2)+init_xyzs[0],
                           radius*np.sin((i/num_wp)*(2*np.pi)+np.pi/2)-radius+init_xyzs[1],
                           init_xyzs[2])
    return init_xyzs, init_rpys, waypoints


def polygon_trajectory(control_freq_hz, period=6, height=1.0, radius=1.0, n_sides=3):
    """
    Generates waypoints along a polygon path with n sides, starting at [0, 0, height].

    Parameters:
    - control_freq_hz: Control frequency in Hz, determining the number of waypoints per second.
    - period: Total time to complete one cycle of the polygon path in seconds.
    - height: The height (Z coordinate) at which the polygon path should be generated.
    - radius: The circumradius of the polygon, which determines its size.
    - n_sides: The number of sides of the polygon.

    Returns:
    - Tuple of initial position, initial orientation (roll, pitch, yaw), and an array of waypoints.
    """
    # Initial position and orientation
    init_xyzs = np.array([0, 0, height])
    init_rpys = np.array([0, 0, 0])

    # Calculate the number of waypoints per side
    num_wp_per_side = control_freq_hz * period // n_sides
    waypoints = np.zeros((num_wp_per_side * n_sides, 3))

    # Generate vertices of the polygon relative to the center [0, 0]
    vertices = np.array([
        [radius * np.cos(2 * np.pi * i / n_sides), radius * np.sin(2 * np.pi * i / n_sides)]
        for i in range(n_sides)
    ])

    # Generate waypoints for each side of the polygon
    for i in range(n_sides):
        start_vertex = vertices[i]
        end_vertex = vertices[(i + 1) % n_sides]
        for j in range(num_wp_per_side):
            t = j / num_wp_per_side
            waypoint = (1 - t) * start_vertex + t * end_vertex
            waypoints[i * num_wp_per_side + j, :2] = waypoint
            waypoints[i * num_wp_per_side + j, 2] = height

    return init_xyzs, init_rpys, waypoints


class TrackAviary(BaseRLAviary):
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 120,
                 ctrl_freq: int = 24,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 episode_len: int = 8,
                 use_depth: bool = True,
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

        self.use_depth = use_depth

        # xyzs, rpys, waypoints = polygon_trajectory(ctrl_freq, n_sides=4, radius=0.5)
        xyzs, rpys, waypoints = circle(ctrl_freq, radius=0.3)

        tracked_drone = WaypointDroneAgent(initial_xyz=xyzs,
                                           initial_rpy=rpys,
                                           pyb_freq=pyb_freq,
                                           ctrl_freq=ctrl_freq,
                                           waypoints=waypoints,
                                           episode_length_sec=episode_len)

        super().__init__(drone_model=drone_model,
                         num_drones=num_tracking_drones,
                         initial_xyzs=np.array([[.25, .25, 1.]]),
                         initial_rpys=np.array([[0., 0., 0.]]),
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
        total_dist, x_dist, y_dist, z_dist = self._distance_from_next_target()
        reward_pose = np.exp(-total_dist * reward_distance_scale)

        z_penalty = 0  # 1./2 * np.exp(-z_dist * 0.8)

        reward = reward_pose - z_penalty
        return reward

    ################################################################################

    def _target_waypoint(self, distance_behind: float = 0.25) -> np.ndarray:
        """
        Calculates a waypoint position directly behind the drone at a specified distance.

        Parameters:
        - distance_behind: The distance behind the drone to calculate the waypoint position.

        Returns:
        - A numpy ndarray representing the x, y, z coordinates of the waypoint position.
        """
        # Extract current state vector of the target drone
        target_state = self.EXTERNAL_AGENTS[0].stateVector()

        # Extract current position and orientation
        target_position = target_state[0:3]
        target_rpy = target_state[7:10]  # roll, pitch, yaw in radians

        # Calculate the change in position due to the yaw angle
        # Yaw rotation matrix about the Z-axis
        yaw = target_rpy[2]
        delta_x = -distance_behind * np.sin(yaw)  # Change in x position
        delta_y = distance_behind * np.cos(yaw)  # Change in y position

        # Calculate the new position behind the drone
        waypoint_position = target_position + np.array([delta_x, delta_y, 0])
        return waypoint_position

    def _distance_from_next_target(self) -> tuple[float, float, float, float]:
        """Distance from tracking to tracked drone
        """
        # state = self._getDroneStateVector(0)
        # distance_to_waypoint = np.linalg.norm(self.target_waypoint() - state[0:3])
        # return distance_to_waypoint

        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        target_pos = self._target_waypoint()

        x_distance = np.abs(current_pos[0] - target_pos[0])
        y_distance = np.abs(current_pos[1] - target_pos[1])
        z_distance = np.abs(current_pos[2] - target_pos[2])
        distance_to_waypoint = np.linalg.norm(target_pos - current_pos)
        return distance_to_waypoint, x_distance, y_distance, z_distance


    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        total_dist, x_dist, y_dist, z_dist = self._distance_from_next_target()
        if ((total_dist > 4.) or       # Truncate when the drone is too far away
            # (z_dist > .2) or
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
        dict[str, Number]
            Dummy value.

        """
        total_distance, x_dist, y_dist, z_dist = self._distance_from_next_target()
        return {"answer": 42,
                "total_distance": total_distance,
                "x_dist": x_dist,
                "y_dist": y_dist,
                "z_dist": z_dist}

    def reset(self,
              seed: int = None,
              options: dict = None):
        self.traj_c = self.traj_c_dist.sample()
        # self.traj_rot = euler_to_quaternion(self.traj_rpy_dist.sample())
        self.traj_scale = self.traj_scale_dist.sample()
        traj_w = self.traj_w_dist.sample()
        self.traj_w = torch.randn_like(traj_w).sign() * traj_w
        return super().reset(seed, options)

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0, segmentation=False)
                # #### Printing observation to PNG frames example ############
                # if self.RECORD:
                #     self._exportImage(img_type=ImageType.RGB,
                #                       img_input=self.rgb[i],
                #                       path=self.ONBOARD_IMG_PATH + "drone_" + str(i),
                #                       frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                #                       )
            ret = self.rgb[0, :, :, 0:3].astype(np.uint8)
            if self.use_depth:
                expanded = np.expand_dims(self.dep[0], axis=-1)
                expanded = (expanded * 255).astype(np.uint8)
                ret = np.concatenate((ret, expanded), axis=2)
            return ret
        elif self.OBS_TYPE == ObservationType.KIN:
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
                tpos = self._target_waypoint()
                rpos = tpos - pos
                obs[i, :] = np.hstack([pos, state[7:10], state[10:13], state[13:16], rpos.flatten()]).reshape(base_action_size,)
            ret = np.array([obs[i, :] for i in range(self.NUM_DRONES)]).astype('float32')

            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            return ret
        else:
            print("[ERROR] in TrackAviary._observationSpace()")

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.RGB:
            channels = 4 if self.use_depth else 3
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.IMG_RES[1], self.IMG_RES[0], channels), dtype=np.uint8)

        #### OBS SPACE OF SIZE 12
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
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

    def _addObstacles(self):
        pass