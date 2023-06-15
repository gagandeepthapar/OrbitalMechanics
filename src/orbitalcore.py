from datetime import datetime
from typing import List, Optional, Union
from dataclasses import dataclass

import numpy as np

from . import astroconsts as c

# pylint: disable=pointless-string-statement

"""
ORBITAL CLASSES
"""


@dataclass
class COES:
    """
    Class containing COES for single orbit
    No methods but can be useful for managing variables

    Args:
        h (float): angular momentum [km2/s]
        ecc (float): eccentricity [-]
        inc_rad (float): inclination [rad]
        raan_rad (float): right ascension of ascending node [rad]
        arg_peri_rad (float): argument of periapsis [rad]
        theta_rad (float): true anomaly [rad]
        semi_major (float): semi major axis [km]

    Returns:
        COES: COES object with user-supplied parameters
    """

    ecc: float
    inc_rad: float
    raan_rad: float
    arg_peri_rad: float
    theta_rad: float
    h: Optional[float] = None
    semi_major: Optional[float] = None

    def __post_init__(self):
        if (self.h is None) and (self.semi_major is None):
            raise AttributeError(
                "Must provide at least angular momentum (h)" "or semi-major axis (km)"
            )

        # TODO: Implement conversion from h to a, a to h
        if self.h is None:
            self.h = 0.1

        if self.semi_major is None:
            self.semi_major = 0.1

        return

    def __str__(self) -> str:
        return (
            f"h [km2/s]: {self.h}\n"
            f"Eccentricity[~]: {self.ecc}\n"
            f"Inclination [rad]: {self.inc_rad}\n"
            f"RAAN [rad]: {self.raan_rad}\n"
            f"Argument of Perigee [rad]: {self.arg_peri_rad}\n"
            f"True Anomaly [rad]: {self.theta_rad}\n"
            f"Semi-Major Axis [km]: {self.semi_major}\n"
        )

    def __eq__(self, other):
        if isinstance(other, COES):
            return (
                (self.h == other.h)
                and (self.ecc == other.ecc)
                and (self.inc_rad == other.inc_rad)
                and (self.raan_rad == other.raan_rad)
                and (self.arg_peri_rad == other.arg_peri_rad)
                and (self.semi_major == other.semi_major)
            )

        return False


@dataclass
class StateVector:
    """
    Class containing State Vector for single orbit
    Simple methods but can be useful for managing variables

    Returns:
       StateVector object with user-supplied arguments
    """

    Rx: float
    Ry: float
    Rz: float
    Vx: float
    Vy: float
    Vz: float

    def to_arr(self) -> np.ndarray:
        """
        Method to convert StateVector into array for preference
        """
        return np.array([self.Rx, self.Ry, self.Rz, self.Vx, self.Vy, self.Vz])

    @staticmethod
    def from_arr(arr: np.ndarray):
        """
        Method to convert array into StateVector type
        Array must be of length 6 and in the following order:
        [Rx, Ry, Rz, Vx, Vy, Vz]
        """
        if len(arr) != 6:
            raise ValueError("Improper length for state vector in 3-D")
        return StateVector(*arr)

    def __eq__(self, other):
        """
        Method to check if two state vectors are equivalent
        """
        if isinstance(other, StateVector):
            return (
                (self.Rx == other.Rx)
                and (self.Ry == other.Ry)
                and (self.Rz == other.Rz)
                and (self.Vx == other.Vx)
                and (self.Vy == other.Vy)
                and (self.Vz == other.Vz)
            )

        return False

    def __str__(self) -> str:
        return f"{self.to_arr()}"


"""
GENERIC FUNCTIONS
"""


# Rotation Matrices
def rot_z(theta: float) -> np.ndarray:
    """
    Rotation about Z Axis
    Adapted from Section 1.3.1 "Spacecraft Dynamics and Control", de Ruiter

    Args:
        theta (float): angle to rotate through [rad]

    Returns:
        np.ndarray: Rotation matrix about Z axis
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def rot_y(theta: float) -> np.ndarray:
    """
    Rotation about Y Axis
    Adapted from Section 1.3.1 "Spacecraft Dynamics and Control", de Ruiter

    Args:
        theta (float): angle to rotate through [rad]

    Returns:
        np.ndarray: Rotation matrix about Y axis
    """
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def rot_x(theta: float) -> np.ndarray:
    """
    Rotation about X Axis
    Adapted from Section 1.3.1 "Spacecraft Dynamics and Control", de Ruiter

    Args:
        theta (float): angle to rotate through [rad]

    Returns:
        np.ndarray: Rotation matrix about X axis
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


# plotters
def plot_earth():
    # TODO: Implement
    return


def plot_orbit():
    # TODO: Implement
    return


"""
CONVERSIONS
"""


def juliandate(calendar_date: datetime):
    """
    Convert calendar date (month, day, year) into julian date
    Adapted from Algorithm 14 from...
        "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        calendar_date (datetime): date to convert into julian date

    Returns:
        float: julian date
    """

    yr = calendar_date.year
    mo = calendar_date.month
    d = calendar_date.day
    h = calendar_date.hour
    m = calendar_date.minute
    s = calendar_date.second

    yr_num = yr + int((mo + 9) / 12)
    yr_B = -1 * int(7 * yr_num / 4)

    yr_calc = 367 * yr + yr_B
    mo_calc = int(275 * mo / 9)
    day_calc = d + 1_721_013.5
    time_calc = (s / 3600 + m / 60 + h) / 24

    return yr_calc + mo_calc + day_calc + time_calc


def coes_to_statevector(coes: COES, mu: float = c.EARTH_MU) -> np.ndarray:
    """
    Convert COES to State Vector: R, V

    Args:
        coes (COES): orbital COES
        mu (float, optional): Gravitational parameter of central body.
        Defaults to c.EARTH_MU.

    Returns:
        np.ndarray: 6x1 state vector: [Rx, Ry, Rz, Vx, Vy, Vz]
    """
    h = coes.h
    ecc = coes.ecc
    inc = coes.inc_rad
    raan = coes.raan_rad
    arg = coes.arg_peri_rad
    theta = coes.theta_rad

    p_r = (
        h**2
        / mu
        * (1 / (1 + ecc * np.cos(theta)))
        * np.array([[np.cos(theta)], [np.sin(theta)], [0]])
    )
    p_v = mu / h * np.array([[-np.sin(theta)], [ecc + np.cos(theta)], [0]])

    q_bar = rot_z(arg) @ rot_x(inc) @ rot_z(raan)

    r_km = np.transpose(q_bar) @ p_r
    v_kms = np.transpose(q_bar) @ p_v

    return np.append(r_km, v_kms)


def statevector_to_coes(statevector: List, mu: int = c.EARTH_MU) -> COES:
    """
    Calculate Classical Orbital Elements from State Vector
    Adapted from Algorithm 9: "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        statevector (np.ndarray): state vector (rx, ry, rz, vx, vy, vz)
        mu (int, optional): central body grav parameter. Defaults to EARTH_MU.

    Returns:
        COES: COES object containing orbital information
    """

    r = np.array(statevector[:3])
    v = np.array(statevector[3:])
    R = np.linalg.norm(r)
    V = np.linalg.norm(v)
    v_r = np.dot(r, v) / R

    h_bar = np.cross(r, v)
    h = np.linalg.norm(h_bar)

    n_bar = np.cross(np.array([0, 0, 1]), h_bar)
    n = np.linalg.norm(n_bar)

    ecc_bar = 1 / mu * ((V**2 - mu / R) * r - r.dot(v) * v)
    ecc = np.linalg.norm(ecc_bar)

    inc = np.arccos(h_bar[2] / h)

    if n != 0:
        raan = np.arccos(n_bar[0] / n)
        if n_bar[1] < 0:
            raan = 2 * np.pi - raan

    else:
        raan = 0

    if n != 0:
        w = np.arccos(np.dot(n_bar, ecc_bar) / (n * ecc))
        if ecc_bar[2] < 0:
            w = 2 * np.pi - w

    else:
        w = 0

    theta = np.arccos(np.dot(ecc_bar, r) / (ecc * R))

    if v_r < 0:
        theta = 2 * np.pi - theta

    rp = h**2 / c.EARTH_MU * 1 / (1 + ecc)
    ra = h**2 / c.EARTH_MU * 1 / (1 - ecc)
    semi = 0.5 * (ra + rp)

    return COES(
        h,
        ecc,
        np.rad2deg(inc),
        np.rad2deg(raan),
        np.rad2deg(w),
        np.rad2deg(theta),
        semi,
    )


def two_line_element_to_coes() -> COES:
    # TODO: Implement
    return


"""
CORE FUNCTIONS
"""


# basic two_body_prop
def two_body() -> np.ndarray:
    # TODO: Implement
    return


def stumpff_s() -> float:
    # TODO: Implement
    return


def stumpff_c() -> float:
    # TODO: Implement
    pass
    return


def la_grange():
    # TODO: Implement
    pass
    return


def lambert_velocity_solver():
    # TODO: Implement
    pass
    return


def universal_variable_propagation():
    # TODO Implement
    pass
    return
