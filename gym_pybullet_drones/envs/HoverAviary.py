import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class HoverAviary(BaseRLAviary):
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
                 episode_len: int = 8,
                 waypoints: np.array = None,
                 eval_mode: bool = False
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
        self.waypoints = waypoints
        self.waypoint_index = 0
        self.EPISODE_LEN_SEC = episode_len
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)

        # max_reward = 5
        # target_pos = self.waypoints[self.waypoint_index]
        # distance_from_target = np.linalg.norm(target_pos - state[0:3])
        # distance_scale = -1.0

        current_pos = state[0:3]
        lin_vel = state[10:12]
        ang_vel = state[13:15]

        reward_distance_scale = 1.2

        target_pos = self.waypoints[self.waypoint_index]
        distance = np.linalg.norm(target_pos - current_pos)
        reward_pose = np.exp(-distance * reward_distance_scale)

        reward = reward_pose
        return reward

    ################################################################################

    def waypointDistance(self) -> float:
        state = self._getDroneStateVector(0)
        target_waypoint = self.waypoints[self.waypoint_index]
        distance_to_waypoint = np.linalg.norm(target_waypoint - state[0:3])
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
        if ((self.waypointDistance() > 4.) or       # Truncate when the drone is too far away
            (abs(state[7]) > .4) or
            (abs(state[8]) > .4)     # Truncate when the drone is too tilted
        ):
            return True

        terminated = False
        if self._targetingFinalWaypoint():
            if self._atWaypoint(self.waypoints.shape[0] - 1, threshold=.0001):
                terminated = True
        return terminated

    ################################################################################

    def _targetingFinalWaypoint(self) -> bool:
        last_waypoint_index = self.waypoints.shape[0] - 1
        return self.waypoint_index == last_waypoint_index

    ################################################################################

    def _atWaypoint(self, waypoint_index, threshold=0.1) -> bool:
        state = self._getDroneStateVector(0)
        target_waypoint = self.waypoints[waypoint_index]
        distance_to_waypoint = np.linalg.norm(target_waypoint - state[0:3])
        if distance_to_waypoint <= threshold:
            return True
        return False

    ################################################################################

    def _advanceWaypoint(self):
        if not self._targetingFinalWaypoint():
            if self._atWaypoint(self.waypoint_index):
                self.waypoint_index += 1

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

    def wp_index(self):
        return self.waypoint_index

    def step(self,
             action: np.array
             ):
        ret = super().step(action)
        self._advanceWaypoint()
        return ret

    def reset(self,
              seed : int = None,
              options : dict = None):
        self.waypoint_index = 0
        return super().reset(seed, options)

    def _computeObs(self):
        obs = super()._computeObs()
        obs[0, 0:3] = obs[0, 0:3] - self.waypoints[self.waypoint_index]
        return obs
