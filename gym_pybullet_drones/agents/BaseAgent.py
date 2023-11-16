from gym_pybullet_drones.utils.enums import Physics


class BaseAgent:
    def __init__(self, physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240
                 ):
        self.AGENT_ID = -1
        self.PHYSICS = physics
        self.CLIENT = -1
        self.PYB_FREQ = pyb_freq
        self.CTRL_FREQ = ctrl_freq
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        self.sim_step_counter = 0

    def reset(self):
        raise NotImplementedError

    def _bulletStep(self):
        raise NotImplementedError

    def bulletStep(self):
        self._bulletStep()
        self.sim_step_counter += 1

    def stateVector(self):
        raise NotImplementedError

    def updateAndStoreKinematicInformation(self):
        raise NotImplementedError

    def kinematics(self):
        raise NotImplementedError
