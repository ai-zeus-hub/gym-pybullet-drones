import numpy as np

import pkg_resources
import pybullet as p

from dataclasses import dataclass

from gym_pybullet_drones.agents.BaseAgent import BaseAgent
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.urdf import parseURDFParameters

@dataclass
class Kinematics:
    pos: np.array = np.zeros(3)
    quat: np.array = np.zeros(4)
    rpy: np.array = np.zeros(3)
    vel: np.array = np.zeros(3)
    ang_v: np.array = np.zeros(3)


def calc_max_xy_torque(drone_model, params, max_rpm):
    max_xy_torque = -1.0
    if drone_model == DroneModel.CF2X:
        max_xy_torque = (2 * params.L * params.KF * max_rpm ** 2) / np.sqrt(2)
    elif drone_model == DroneModel.CF2P:
        max_xy_torque = (params.L * params.KF * max_rpm ** 2)
    elif drone_model == DroneModel.RACE:
        max_xy_torque = (2 * params.L * params.KF * max_rpm ** 2) / np.sqrt(2)
    else:
        print("[ERROR]: Unsupported drone type")
    return max_xy_torque


class DroneAgent(BaseAgent):
    def __init__(self,
                drone_model: DroneModel = DroneModel.CF2X,
                initial_xyz: np.array = None,
                initial_rpy: np.array = None,
                physics: Physics = Physics.PYB,
                pyb_freq: int = 240,
                ctrl_freq: int = 240):
        super().__init__(physics=physics, pyb_freq=pyb_freq, ctrl_freq=ctrl_freq)

        self.DRONE_MODEL = drone_model

        # Constants
        self.G = 9.8

        # URDF
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.PARAMS = parseURDFParameters(self.URDF)

        # Compute Constants
        self.GRAVITY = self.G * self.PARAMS.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.PARAMS.KF))
        self.MAX_RPM = np.sqrt((self.PARAMS.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.PARAMS.KF))
        self.MAX_THRUST = (4 * self.PARAMS.KF * self.MAX_RPM ** 2)
        self.MAX_XY_TORQUE = calc_max_xy_torque(drone_model, self.PARAMS, self.MAX_RPM)
        self.MAX_Z_TORQUE = (2 * self.PARAMS.KM * self.MAX_RPM ** 2)

        self.INIT_XYZ = initial_xyz if initial_xyz is not None else np.zeros(3)
        assert self.INIT_XYZ.shape == (3,), "Invalid initial XYZ shape"
        self.INIT_RPY = initial_rpy if initial_rpy is not None else np.zeros(3)
        assert self.INIT_RPY.shape == (3,), "Invalid initial RPY shape"

        self.kinematics = Kinematics()
        self.last_action = np.zeros(4)

    def _physics(self, rpm):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        """
        forces = np.array(rpm**2) * self.PARAMS.KF
        torques = np.array(rpm**2) * self.PARAMS.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.AGENT_ID,
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.AGENT_ID,
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )

    def updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).
        """
        self.kinematics.pos[:], self.kinematics.quat[:] = p.getBasePositionAndOrientation(self.AGENT_ID,
                                                                                          physicsClientId=self.CLIENT)
        self.kinematics.rpy[:] = p.getEulerFromQuaternion(self.kinematics.quat[:])
        self.kinematics.vel[:], self.kinematics.ang_v[:] = p.getBaseVelocity(self.AGENT_ID,
                                                                             physicsClientId=self.CLIENT)

    def stateVector(self):
        """Returns the state vector of the drone.

        Returns
        -------
        ndarray
            (16,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

              0-2: x, y, z
              3-6: quat's
              7-9: roll, pitch, yaw
            10-12: vx, vy, vz
            13-15: wx, wy, wz
            16-19: last action
        """
        state = np.hstack([self.kinematics.pos[:],
                           self.kinematics.quat[:],
                           self.kinematics.rpy[:],
                           self.kinematics.vel[:],
                           self.kinematics.ang_v[:],
                           self.last_action[:]])
        return state.reshape(20,)
    def reset(self):
        self.AGENT_ID = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + self.URDF),
                                   self.INIT_XYZ,
                                   p.getQuaternionFromEuler(self.INIT_RPY),
                                   flags=p.URDF_USE_INERTIA_FROM_FILE,
                                   physicsClientId=self.CLIENT
                                   )
