import gymnasium
import numpy as np
import torch
from pathlib import Path
from gymnasium import spaces
import pybullet as p
import imageio

from gym_pybullet_drones.agents.WaypointDroneAgent import WaypointDroneAgent
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, DepthType, ImageType


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
    if n_sides == 0:
        return circle(control_freq_hz, radius=radius, height=height, period=period)
    # Initial position and orientation
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

    neg_wp_0 = -waypoints[0, :2]
    for i in range(len(waypoints)):
        waypoints[i, :2] = waypoints[i, :2] + neg_wp_0
    return waypoints[0], init_rpys, waypoints


class TrackAviary(BaseRLAviary):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 output_folder: str = 'results',
                 pyb_freq: int = 120,
                 ctrl_freq: int = 24,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 episode_len: int = 8,
                 distance_reward_scale: float = 1.2,
                 depth_type: DepthType = DepthType.IMAGE,
                 max_distance: float = 2.,
                 include_rpos_in_obs: bool = False,
                 static_idx: int | None = None,
                 ):
        self.include_rpos_in_obs = include_rpos_in_obs
        self.EPISODE_LEN_SEC = episode_len

        num_tracking_drones = 1

        self.env_choices = [0, 3, 4, 5, 6]
        self.depth_type = depth_type
        self.add_shapes = True
        self.distance_reward_scale = distance_reward_scale
        self.env_idx = static_idx if static_idx is not None else 0
        self.static_idx = static_idx
        self.max_distance = max_distance
        self.desired_distance = 0.25
        self._collisions: list[tuple[tuple[float, float, float], float]] = []  # center to radius

        xyz, rpy, waypoints = polygon_trajectory(ctrl_freq, n_sides=self.env_idx, radius=0.5)


        tracked_drone = WaypointDroneAgent(initial_xyz=xyz,
                                           initial_rpy=rpy,
                                           pyb_freq=pyb_freq,
                                           ctrl_freq=ctrl_freq,
                                           waypoints=waypoints,
                                           episode_length_sec=episode_len)

        init_yaw = rpy[2]
        xyz_delta = np.array([np.cos(init_yaw), np.sin(init_yaw), 0]) * self.desired_distance
        init_xyz = xyz - xyz_delta
        super().__init__(drone_model=drone_model,
                         num_drones=num_tracking_drones,
                         initial_xyzs=np.expand_dims(init_xyz, axis=0),
                         initial_rpys=np.expand_dims(rpy, axis=0),
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         external_agents=[tracked_drone],
                         action_buffer_size=1,
                         output_folder=output_folder
                         )
        self.img_counter = 0
        self.IMG_RES = np.array([64, 64])  # todo: ajr - remove
        self.IMG_FRAME_PER_SEC = 24
        self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ / self.IMG_FRAME_PER_SEC)
        self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
        self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
        self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))

    ################################################################################

    def _init_env(self) -> None:
        xyzs, rpys, waypoints = polygon_trajectory(self.CTRL_FREQ, n_sides=self.env_idx, radius=0.5)
        for agent in self.EXTERNAL_AGENTS:
            agent.reset(xyzs, rpys, waypoints)


    def is_target_in_fov(self, target_pos: np.ndarray) -> bool:
        # Get the view matrix from the drone's current position and orientation
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[0, :])).reshape(3, 3)
        target_dir = np.dot(rot_mat, np.array([1000, 0, 0]))
        view_matrix = p.computeViewMatrix(cameraEyePosition=self.pos[0, :] + np.array([0, 0, self.L]),
                                          cameraTargetPosition=self.pos[0, :] + target_dir,
                                          cameraUpVector=[0, 0, 1])

        # Convert the view matrix to a NumPy array and reshape it
        view_matrix_np = np.array(view_matrix).reshape(4, 4).T  # Transpose to match PyBullet's row-major order

        # Get the projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(fov=60.0, aspect=1.0, nearVal=self.L, farVal=1000.0)
        projection_matrix_np = np.array(projection_matrix).reshape(4, 4).T  # Transpose for consistency

        # Transform target position to homogeneous coordinates
        target_pos_homogeneous = np.append(target_pos, 1.0)  # Append 1 for homogeneous coordinates

        # Transform target position to camera coordinates
        camera_coords = np.dot(view_matrix_np, target_pos_homogeneous)

        # Project camera coordinates to 2D using the projection matrix
        projected = np.dot(projection_matrix_np, camera_coords)

        # Perform perspective division to get normalized device coordinates
        ndc = projected[:-1] / projected[-1]

        # Check if x and y in NDC are within [-1, 1], indicating the target is within the FOV
        in_fov = np.all(ndc[:-1] >= -1) and np.all(ndc[:-1] <= 1)

        return in_fov

    def _collided(self, collision_threshold: float = .08) -> bool:
        state = self._getDroneStateVector(0)
        my_position = state[0:3]
        for center, radius in self._collisions:
            center_point = np.array(center)

            my_position_xy = my_position[:2]
            center_point_xy = center_point[:2]

            vector_to_point = my_position_xy - center_point_xy

            distance_to_cylinder_axis = np.linalg.norm(vector_to_point)

            if distance_to_cylinder_axis < (radius + collision_threshold):
                return True
        return False
    def _computeReward(self):
        """Computes the current reward value for the drone doing the tracking

        Returns
        -------
        float
            The reward.
        """
        if self._collided():
            return -10

        target_pos, target_rpy = self._target_waypoint()
        total_dist, x_dist, y_dist, z_dist = self._distance_from_next_target(target_pos)
        reward_pose = np.exp(-total_dist * self.distance_reward_scale)

        state = self._getDroneStateVector(0)
        current_rpy = state[7:10]

        target_fov_penalty = 0
        # if not self.is_target_in_fov(target_pos):
        #     target_fov_penalty = 1

        # spin = np.square(velocity[..., -1])
        vel_w = state[13:16]
        angular_velocity_penalty = 0  # 0.005 * np.sum(np.square(vel_w))
        z_penalty = 0  # 1./2 * np.exp(-z_dist * 0.8)
        penalties = z_penalty + angular_velocity_penalty + target_fov_penalty

        max_steps = self.EPISODE_LEN_SEC * self.CTRL_FREQ
        step_reward = 1. / max_steps
        keep_alive = step_reward * self.step_counter

        # reward = min(1, 0.9 * reward_pose + 0.1 * keep_alive) - penalties
        reward = reward_pose - penalties
        return reward

    ################################################################################

    def _target_waypoint(self) -> tuple[np.ndarray, np.ndarray]:
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

        delta_x = -self.desired_distance * np.cos(yaw)  # Change in x position
        delta_y = -self.desired_distance * np.sin(yaw)  # Change in y position

        # Calculate the new position behind the drone
        waypoint_position = target_position + np.array([delta_x, delta_y, 0])
        return waypoint_position, target_rpy

    def _distance_from_next_target(self, target_pos: np.ndarray) -> tuple[float, float, float, float]:
        """Distance from tracking to tracked drone
        """
        # state = self._getDroneStateVector(0)
        # distance_to_waypoint = np.linalg.norm(self.target_waypoint() - state[0:3])
        # return distance_to_waypoint

        state = self._getDroneStateVector(0)
        current_pos = state[0:3]

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
        target_pos, target_rpy = self._target_waypoint()
        total_dist, x_dist, y_dist, z_dist = self._distance_from_next_target(target_pos)
        if ((total_dist > self.max_distance) or # Terminate when the drone is too far away
            # (z_dist > .2) or
            (abs(state[7]) > .4) or
            (abs(state[8]) > .4)     # Terminate when the drone is too tilted
        ):
            return True
        elif self._collided():
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
        target_pos, target_rpy = self._target_waypoint()
        total_distance, x_dist, y_dist, z_dist = self._distance_from_next_target(target_pos)
        return {"answer": 42,
                "total_distance": total_distance,
                "x_dist": x_dist,
                "y_dist": y_dist,
                "z_dist": z_dist}

    def reset(self,
              seed: int = None,
              options: dict = None):
        self.env_idx = self.static_idx if self.static_idx is not None else np.random.choice(self.env_choices)
        self.add_shapes = self.GUI or bool(np.random.choice([True, False]))
        self._init_env()
        self.img_counter = 0
        return super().reset(seed, options)

    def _exportRGBD(self,
                     path: str,
                     frame_num: int=0):
        img_name = Path(path) / f"reset_{self.reset_counter}_frame_{frame_num}_step_{self.step_counter}.png"

        img = self.rgb[0, :, :, 0:3].astype(np.uint8)  # strip off alpha channel
        if self.depth_type == DepthType.IMAGE:
            expanded = np.expand_dims(self.dep[0], axis=-1)
            expanded = (expanded * 255).astype(np.uint8)
            img = np.concatenate((img, expanded), axis=2)
        imageio.imwrite(img_name, img)

    def _computeObs(self):
        save_dataset = False
        img_cap_freq = 5
        if save_dataset and (self.img_counter % img_cap_freq == 0):
            self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0, segmentation=False)
            output_dir: Path = Path(self.OUTPUT_FOLDER) / "drone_0_rgbd"
            output_dir.mkdir(exist_ok=True)
            self._exportRGBD(path=str(output_dir),
                             frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ))
        self.img_counter += 1
        observation = {}
        if self.OBS_TYPE == ObservationType.RGB or self.OBS_TYPE == ObservationType.MULTI:
            # self.IMG_RES is (w, h), but getDroneImage returns (h, w)
            self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0, segmentation=self.GUI)
            #### Printing observation to PNG frames example ############
            save_dataset = False
            if save_dataset:
                output_dir: Path = Path(self.OUTPUT_FOLDER) / "drone_0"
                output_dir.mkdir(exist_ok=True)
                self._exportImage(img_type=ImageType.RGB,
                                    img_input=self.rgb[0],
                                    path=str(output_dir),
                                    frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                                    )
            img = self.rgb[0, :, :, 0:3].astype(np.uint8)  # strip off alpha channel
            if self.depth_type == DepthType.IMAGE:
                expanded = np.expand_dims(self.dep[0], axis=-1)
                expanded = (expanded * 255).astype(np.uint8)
                img = np.concatenate((img, expanded), axis=2)
            elif self.depth_type == DepthType.DOWN_SAMPLED:
                reshaped_array = self.dep[0].reshape((6, 8, 8, 8))
                down_sampled_array = reshaped_array.min(axis=(1, 3))
                observation["depth"] = down_sampled_array * 2 - 1
            else:
                print("Depth ignored")
            observation["img"] = img
        if self.OBS_TYPE == ObservationType.KIN or self.OBS_TYPE == ObservationType.MULTI:
            base_kin_obs_size = 12 if self.include_rpos_in_obs else 9
            obs = np.zeros((self.NUM_DRONES, base_kin_obs_size))
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)

                # rpy
                rpy = state[7:10]  # needed if we instead just use quats?
                norm_rpy = rpy / np.pi

                # linear velocity
                vel = state[10:13]
                max_vel = np.array([3.5, 3.5, 3.5])  # empirically observed in simulation
                clipped_vel = np.clip(vel, -max_vel, max_vel)
                norm_vel = np.clip(clipped_vel, -max_vel, max_vel) / max_vel
                # for j in range(len(max_vel)):
                #     if vel[j] > max_vel[j]:
                #         print(f"***WARNING: max_vel too low! Saw {vel=} and {max_vel=}")

                # angular velocity
                w_vel = state[13:16]
                max_w_vel = np.array([16, 16, 16])  # empirically observed in simulation 13.1, 8.465
                clipped_w_vel = np.clip(w_vel, -max_w_vel, max_w_vel)
                norm_w_vel = np.clip(clipped_w_vel, -max_w_vel, max_w_vel) / max_w_vel
                # for j in range(len(w_vel)):
                #     if w_vel[j] > max_w_vel[j]:
                #         print(f"***WARNING: max_w_vel too low! Saw {w_vel=} and {max_w_vel=}")

                # relative position
                if self.include_rpos_in_obs:
                    pos = state[0:3]
                    rpos = self._target_waypoint()[0] - pos
                    norm_rpos = np.clip(rpos / self.max_distance, np.array([-1, -1, -1]), np.array([1, 1, 1]))
                    obs[i, :] = np.hstack([norm_rpy, norm_vel, norm_w_vel, norm_rpos.flatten()]).reshape(base_kin_obs_size,)
                else:
                    obs[i, :] = np.hstack([norm_rpy, norm_vel, norm_w_vel]).reshape(base_kin_obs_size,)
            obs = np.array([obs[i, :] for i in range(self.NUM_DRONES)]).astype('float32')

            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                obs = np.hstack([obs, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            observation["kin"] = obs
        return observation

    def _observationSpace(self):
        dict_space = gymnasium.spaces.Dict()
        if self.OBS_TYPE == ObservationType.RGB or self.OBS_TYPE == ObservationType.MULTI:
            channels = 4 if self.depth_type == DepthType.IMAGE else 3
            dict_space["img"] = spaces.Box(low=0,
                                           high=255,
                                           shape=(self.IMG_RES[1], self.IMG_RES[0], channels), dtype=np.uint8)
        if self.depth_type == DepthType.DOWN_SAMPLED:
            dict_space["depth"] = spaces.Box(low=-1., high=+1.,
                                             shape=(self.IMG_RES[1]//8, self.IMG_RES[0]//8))
        if self.OBS_TYPE == ObservationType.KIN or self.OBS_TYPE == ObservationType.MULTI:
            # without rpos
            norm_lo = -1
            norm_hi = +1
            if self.include_rpos_in_obs:
                obs_lower_bound = np.array([[norm_lo, norm_lo, norm_lo, norm_lo, norm_lo, norm_lo, norm_lo, norm_lo,
                                             norm_lo, norm_lo, norm_lo, norm_lo] for _ in range(self.NUM_DRONES)])
                obs_upper_bound = np.array([[norm_hi, norm_hi, norm_hi, norm_hi, norm_hi, norm_hi, norm_hi, norm_hi,
                                             norm_hi, norm_hi, norm_hi, norm_hi] for _ in range(self.NUM_DRONES)])
            else:
                obs_lower_bound = np.array([[norm_lo, norm_lo, norm_lo, norm_lo, norm_lo, norm_lo, norm_lo, norm_lo,
                                            norm_lo] for _ in range(self.NUM_DRONES)])
                obs_upper_bound = np.array([[norm_hi, norm_hi, norm_hi, norm_hi, norm_hi, norm_hi, norm_hi, norm_hi,
                                            norm_hi] for _ in range(self.NUM_DRONES)])

            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack(
                        [obs_lower_bound, np.array([[act_lo, act_lo, act_lo, act_lo] for _ in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack(
                        [obs_upper_bound, np.array([[act_hi, act_hi, act_hi, act_hi] for _ in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE == ActionType.PID:
                    obs_lower_bound = np.hstack(
                        [obs_lower_bound, np.array([[act_lo, act_lo, act_lo] for _ in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack(
                        [obs_upper_bound, np.array([[act_hi, act_hi, act_hi] for _ in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for _ in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for _ in range(self.NUM_DRONES)])])
            dict_space["kin"] = spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        if len(dict_space) == 0:
            raise ValueError("Dict space should contain at least item")
        return dict_space

    def add_cylinder(self,
                     position: tuple[float, float, float],
                     height: float = 2.5,
                     radius: float = .02):
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self.CLIENT,
        )
        cylinder_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=position,
            physicsClientId=self.CLIENT,
        )
        self._collisions.append((position, radius))

    def _addObstacles(self):
        self._collisions = []
        if not self.add_shapes:
            return
        if self.env_idx == 0:
            self.add_cylinder((-0.45, -0.25, 0))
            self.add_cylinder((-0.45, +0.25, 0))
            self.add_cylinder((-1.3, +0.15, 0))
            self.add_cylinder((+0.5, 0, 0))
        elif self.env_idx == 3:
            self.add_cylinder((-0.45, +0, 0))
            self.add_cylinder((+0.45, +0, 0))
            self.add_cylinder((-1.1, 0, 0))
        elif self.env_idx == 4:
            self.add_cylinder((-0.45, -0.75, 0))
            self.add_cylinder((-0.45, -0.2, 0))
            self.add_cylinder((-0.45, +0.2, 0))
            self.add_cylinder((-0.45, +0.75, 0))
            self.add_cylinder((-1.3, +0.15, 0))
            self.add_cylinder((+0.5, 0, 0))
        elif self.env_idx == 5:
            self.add_cylinder((-0.45, -0.75, 0))
            self.add_cylinder((-0.45, -0.25, 0))
            self.add_cylinder((-0.45, +0.25, 0))
            self.add_cylinder((-0.45, +0.75, 0))
            self.add_cylinder((-1.3, +0.15, 0))
            self.add_cylinder((+0.5, 0, 0))
        elif self.env_idx == 6:
            self.add_cylinder((-0.45, -0.8, 0))
            self.add_cylinder((-0.45, -0.2, 0))
            self.add_cylinder((-0.45, +0.2, 0))
            self.add_cylinder((-0.45, +0.8, 0))
            self.add_cylinder((-1.3, +0.15, 0))
            self.add_cylinder((+0.5, 0, 0))
        else:
            raise NotImplementedError
