import numpy as np
from typing import Optional

from gym_pybullet_drones.agents.DroneAgent import DroneAgent
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class WaypointDroneAgent(DroneAgent):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyz=None,
                 initial_rpy=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 waypoints: Optional[np.array] = None,
                 episode_length_sec: int = 5):
        super().__init__(drone_model=drone_model,
                         initial_xyz=initial_xyz,
                         initial_rpy=initial_rpy,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         physics=physics)
        self.EPISODE_LEN_SEC = episode_length_sec
        self.ctrl = DSLPIDControl(drone_model=drone_model, g=self.G)
        self.waypoints = waypoints if waypoints is not None else np.array([[0, 0, 0]])
        self.wp_counter = 0
        self.cur_rpms = np.zeros(4)

    def current_waypoint(self):
        return self.waypoints[self.wp_counter]

    def calculateRpms(self):
        # target_pos = np.hstack(
        #     [self.waypoints[self.wp_counter, 0:2],
        #      self.INIT_XYZ[2]]),

        rpms, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                              cur_pos=self.kinematics.pos,
                                              cur_quat=self.kinematics.quat,
                                              cur_vel=self.kinematics.vel,
                                              cur_ang_vel=self.kinematics.ang_v,
                                              target_pos=self.current_waypoint(),
                                              target_rpy=self.INIT_RPY
                                              )
        rpms = np.clip(rpms, 0, self.MAX_RPM)
        return rpms

    def _bulletStep(self):
        if (self.sim_step_counter % self.PYB_STEPS_PER_CTRL) == 0:
            self.cur_rpms = self.calculateRpms()
            self.wp_counter = (self.wp_counter + 1) % len(self.waypoints)
        if self.PHYSICS == Physics.PYB:
            self._physics(self.cur_rpms)
        else:
            raise NotImplementedError
        self.last_action = self.cur_rpms

    def reset(self, init_xyz: np.array=None, init_rpy: np.array=None, waypoints: Optional[np.array] = None):
        super().reset(init_xyz, init_rpy)
        if waypoints is not None:
            self.waypoints = waypoints
        self.wp_counter = 0
        self.cur_rpms = np.zeros(4)
        self.ctrl.reset()

    # def clipAndNormalizeState(self, state):
    #     """Normalizes a drone's state to the [-1,1] range.
    #
    #     Parameters
    #     ----------
    #     state : ndarray
    #         (16,)-shaped array of floats containing the non-normalized state of a single drone.
    #
    #     Returns
    #     -------
    #     ndarray
    #         (16,)-shaped array of floats containing the normalized state of a single drone.
    #     """
    #     MAX_LIN_VEL_XY = 3
    #     MAX_LIN_VEL_Z = 1
    #
    #     MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
    #     MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC
    #
    #     MAX_PITCH_ROLL = np.pi  # Full range
    #
    #     # clip
    #     clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
    #     clipped_pos_z = np.clip(state[2], 0, MAX_Z)
    #     clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
    #     clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
    #     clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
    #
    #     # normalize
    #     normalized_pos_xy = clipped_pos_xy / MAX_XY
    #     normalized_pos_z = clipped_pos_z / MAX_Z
    #     normalized_rp = clipped_rp / MAX_PITCH_ROLL
    #     normalized_y = state[9] / np.pi  # No reason to clip
    #     normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
    #     normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
    #     normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
    #         state[13:16]) != 0 else state[13:16]
    #
    #     norm_and_clipped = np.hstack([normalized_pos_xy,
    #                                   normalized_pos_z,
    #                                   state[3:7],
    #                                   normalized_rp,
    #                                   normalized_y,
    #                                   normalized_vel_xy,
    #                                   normalized_vel_z,
    #                                   normalized_ang_vel,
    #                                   ]).reshape(16, )
    #     return norm_and_clipped
