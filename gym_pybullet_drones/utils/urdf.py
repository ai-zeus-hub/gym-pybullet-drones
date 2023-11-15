import numpy as np
import pkg_resources
from dataclasses import dataclass

import xml.etree.ElementTree as etxml


@dataclass
class Parameters:
    M: float
    L: float
    THRUST2WEIGHT_RATIO: float
    J: np.array
    J_INV: np.array
    KF: float
    KM: float
    COLLISION_H: float
    COLLISION_R: float
    COLLISION_Z_OFFSET: float
    MAX_SPEED_KMH: float
    GND_EFF_COEFF: float
    PROP_RADIUS: float
    DRAG_COEFF: np.array
    DW_COEFF_1: float
    DW_COEFF_2: float
    DW_COEFF_3: float


def parseURDFParameters(urdf_filename: str):
    """Loads parameters from an URDF file.

    This method is nothing more than a custom XML parser for the .urdf
    files in folder `assets/`.

    """
    urdf_tree = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + urdf_filename)).getroot()
    M = float(urdf_tree[1][0][1].attrib['value'])
    L = float(urdf_tree[0].attrib['arm'])
    thrust2_weight_ratio = float(urdf_tree[0].attrib['thrust2weight'])
    IXX = float(urdf_tree[1][0][2].attrib['ixx'])
    IYY = float(urdf_tree[1][0][2].attrib['iyy'])
    IZZ = float(urdf_tree[1][0][2].attrib['izz'])
    J = np.diag([IXX, IYY, IZZ])
    J_INV = np.linalg.inv(J)
    KF = float(urdf_tree[0].attrib['kf'])
    KM = float(urdf_tree[0].attrib['km'])
    COLLISION_H = float(urdf_tree[1][2][1][0].attrib['length'])
    COLLISION_R = float(urdf_tree[1][2][1][0].attrib['radius'])
    COLLISION_SHAPE_OFFSETS = [float(s) for s in urdf_tree[1][2][0].attrib['xyz'].split(' ')]
    COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
    MAX_SPEED_KMH = float(urdf_tree[0].attrib['max_speed_kmh'])
    GND_EFF_COEFF = float(urdf_tree[0].attrib['gnd_eff_coeff'])
    PROP_RADIUS = float(urdf_tree[0].attrib['prop_radius'])
    DRAG_COEFF_XY = float(urdf_tree[0].attrib['drag_coeff_xy'])
    DRAG_COEFF_Z = float(urdf_tree[0].attrib['drag_coeff_z'])
    DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
    DW_COEFF_1 = float(urdf_tree[0].attrib['dw_coeff_1'])
    DW_COEFF_2 = float(urdf_tree[0].attrib['dw_coeff_2'])
    DW_COEFF_3 = float(urdf_tree[0].attrib['dw_coeff_3'])
    return Parameters(M=M, L=L, THRUST2WEIGHT_RATIO=thrust2_weight_ratio, J=J, J_INV=J_INV, KF=KF, KM=KM,
                      COLLISION_H=COLLISION_H, COLLISION_R=COLLISION_R, COLLISION_Z_OFFSET=COLLISION_Z_OFFSET,
                      MAX_SPEED_KMH=MAX_SPEED_KMH, GND_EFF_COEFF=GND_EFF_COEFF, PROP_RADIUS=PROP_RADIUS,
                      DRAG_COEFF=DRAG_COEFF, DW_COEFF_1=DW_COEFF_1, DW_COEFF_2=DW_COEFF_2, DW_COEFF_3=DW_COEFF_3)
