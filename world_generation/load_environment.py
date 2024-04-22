import pybullet as p
import time

def load_environment(environment_path: str):
    """
    Loads the specified environment URDF into PyBullet.

    :param environment_path: The path to the environment URDF file.
    """
    p.connect(p.GUI)
    p.loadURDF(environment_path, [0, 0, 0])

    while True:
        p.stepSimulation()
        time.sleep(1./20000.)

# Example usage:
load_environment("environment.urdf")
