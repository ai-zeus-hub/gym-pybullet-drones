import numpy as np

from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class MultiTrackingAviary(BaseMultiagentAviary):
    """TODO: doc string"""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 240,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,  # todo: change to pid
        tracking_drones: int = 1,
        target_drones: int = 1,
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
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        todo: ajr - other parameters
        """
        # One controller can be used for all drones as they are
        # the same model
        self.num_tracking_drones = tracking_drones  # For convenience
        self.num_target_drones = target_drones

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            num_drones=tracking_drones + target_drones,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )
        R = .3  # Copied from pid.py for now

        # Initialize a circular trajectory
        NUM_WP = self.CTRL_FREQ * self.PYB_STEPS_PER_CTRL
        self.TARGET_POS = np.zeros((NUM_WP, self.num_target_drones))
        for i in range(NUM_WP):
            self.TARGET_POS[i, :] = R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2) + self.INIT_XYZS[0, 0]
                                    # (R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2) + self.INIT_XYZS[0, 0],
                                    #  R * np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2) - R + self.INIT_XYZS[0, 1],
                                    #  0)
        self.wp_counters = np.array([int((i * NUM_WP / 6) % NUM_WP) for i in range(self.num_target_drones)])
        self.ctrl = [DSLPIDControl(drone_model=drone_model) for _ in range(self.NUM_DRONES)]
        self.target_drones = np.arange(tracking_drones, self.NUM_DRONES)
        self.tracking_drones = np.arange(tracking_drones)

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment. The action space is only
        for the first "tracking" drone, as the tracked drone uses a static set
        of waypoints

        Returns
        -------
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in TrackingAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for _ in range(self.num_tracking_drones)])
        act_upper_bound = np.array([+1*np.ones(size) for _ in range(self.num_tracking_drones)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        # TODO: ajr
        return super()._observationSpace()
    def _calcuateTargetWaypoint(self):
        return np.zeros((1, 4))

    def _appendActions(self, action):
        """Rotate the target drone
        """
        target_action = self._calculateTargetWaypoint()
        return np.concatenate((action, target_action))

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """

        # TODO: ajr - Reward based upon distance
        # For now, just 1 to 1, later I can add
        # clusters or something else
        state_a = self._getDroneStateVector(0)
        state_b = self._getDroneStateVector(1)
        return -1 * np.linalg.norm(np.array([0, 0, 1]) - state_a[0:3]) ** 2

    # def abv(self, drone_num):
    #     shifted_drone = drone_num - self.tracking_drones
    #     target_pos = np.hstack([self.TARGET_POS[self.wp_counters[shifted_drone], 0:2],
    #                             self.INIT_XYZS[shifted_drone, 2]])
    #     x = self.ctrl[shifted_drone].computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
    #                                     state=obs[shifted_drone], # TODO: AJR ?
    #                                     target_pos=target_pos,
    #                                     target_rpy=self.INIT_RPYS[shifted_drone, :]
    #                                     )
    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused in this implementation.

        Returns
        -------
        bool
            Always false.

        """
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
        return {
            "answer": 42
        }  #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """Normalizes a single drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

              0-2: x, y, z
              3-6: quat's
              7-9: roll, pitch, yaw
            10-12: vx, vy, vz
            13-15: wx, wy, wz
            16-19: last action

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

              0-2: x, y, z -- normalized to the max possible distance (for that axis) per episode
              3-6: quat's -- unmodified
              7-9: roll, pitch, yaw -- normalized to 2*pi
            10-12: vx, vy, vz -- normalized to 3, 3, 1
            13-15: wx, wy, wz -- normalized angular velocity
            16-19: last action -- unmodified
        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(
                state,
                clipped_pos_xy,
                clipped_pos_z,
                clipped_rp,
                clipped_vel_xy,
                clipped_vel_z,
            )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = (
            state[13:16] / np.linalg.norm(state[13:16])
            if np.linalg.norm(state[13:16]) != 0
            else state[13:16]
        )

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_y,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20],
            ]
        ).reshape(
            20,
        )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                    state[0], state[1]
                ),
            )
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(
                    state[2]
                ),
            )
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                    state[7], state[8]
                ),
            )
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                    state[10], state[11]
                ),
            )
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(
                    state[12]
                ),
            )