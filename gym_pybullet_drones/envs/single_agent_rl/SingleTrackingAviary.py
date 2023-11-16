import numpy as np

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, \
    BaseSingleAgentAviary
from gym_pybullet_drones.agents.DroneAgent import Kinematics
from gym_pybullet_drones.agents.WaypointDroneAgent import WaypointDroneAgent


class SingleTrackingAviary(BaseSingleAgentAviary):
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 initial_xyzs=None,
                 initial_rpys=None,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 episode_length_sec = 5,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 target_initial_xyz=None,
                 target_initial_rpy=None,
                 target_waypoints=None,
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
        external_agent = WaypointDroneAgent(
            initial_xyz=target_initial_xyz,
            initial_rpy=target_initial_rpy,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            waypoints=target_waypoints,
            episode_length_sec=5
        )
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         episode_length_sec=episode_length_sec,  # todo: ajr - remove
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         external_agents=[external_agent]
                         )
        self.num_target_drones = 1


    ################################################################################
    def _observationSpace(self):
        # TODO: ajr
        return super()._observationSpace()

    @property
    def _target_pos(self):
        # target_pos = self.INIT_XYZS[0]
        # target_pos[2] = 1
        # return target_pos
        kin: Kinematics = self.EXTERNAL_AGENTS[0].kinematics
        return kin.pos

    def upside_down(self, rpy):
        return np.abs(rpy[0]) > 3.0 or np.abs(rpy[1]) > 3.0

    def target_distance(self, cur_pos):
        return np.abs(np.linalg.norm(self._target_pos - cur_pos))

    def too_far(self, cur_pos, threshold=1):
        return self.target_distance(cur_pos) > threshold

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """

        state = self._getDroneStateVector(0)
        distance = self.target_distance(cur_pos=state[0:3])
        distance_reward = -1 * (distance ** 2)

        orientation_penalty = 0
        if self.upside_down(rpy=state[7:10]):
            orientation_penalty = -1000

        time_reward = self.step_counter * 0.01

        reward = distance_reward + orientation_penalty + time_reward
        return reward

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            print("past episode down!")
            return True
        elif self.upside_down(state[7:10]):
            print("upside down!")
            return True
        elif self.too_far(cur_pos=state[0:3], threshold=4):
            print("too far!")
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
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

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
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
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
            print("[WARNING] it", self.step_counter,
                  "in SingleTrackingAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                      state[0], state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in SingleTrackingAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in SingleTrackingAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                      state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in SingleTrackingAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                      state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in SingleTrackingAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
