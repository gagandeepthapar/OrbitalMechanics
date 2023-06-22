"""
orbitalcore 
Core module for majority of orbital mechanics functions
"""

from datetime import datetime
from typing import List, Optional, Union
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

import numpy as np
import pandas as pd

from . import astroconsts as ast


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
    mu: Optional[int] = ast.EARTH_MU

    def __post_init__(self):
        if (self.h is None) and (self.semi_major is None):
            raise AttributeError(
                "Must provide at least angular momentum (h)"
                "or semi-major axis (semi_major / a)"
            )

        if self.h is None:
            self.h: float = np.sqrt(self.semi_major * self.mu * (1 - self.ecc**2))

        if self.semi_major is None:
            self.semi_major: float = self.h**2 / self.mu * 1 / (1 - self.ecc**2)

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


# Math
def cosd(ang: float) -> float:
    """
    MATLAB-esque method for cosine of degree value
    """
    return np.cos(np.deg2rad(ang))


def sind(ang: float) -> float:
    """
    MATLAB-esque method for sine of degree value
    """
    return np.sin(np.deg2rad(ang))


def tand(ang: float) -> float:
    """
    MATLAB-esque method for tangent of degree value
    """
    return np.tan(np.deg2rad(ang))


def acosd(val: float) -> float:
    """
    MATLAB-esque method for returning degrees of arccosine of value
    """
    return np.rad2deg(np.arccos(val))


def asind(val: float) -> float:
    """
    MATLAB-esque method for returning degrees of arcsine of value
    """
    return np.rad2deg(np.arcsin(val))


def atand(val: float) -> float:
    """
    MATLAB-esque method for returning degrees of arctangent of value
    """
    return np.rad2deg(np.arctan(val))


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


def sphere_data(rad: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Method to return array of arrays that describe a sphere with supplied radius.
    Data is useful when using `plot_surface` method from Matplotlib

    Args:
        rad (float): radius of sphere to return. Defaults to 1

    Returns:
        x (np.ndarray): x surface
        y (np.ndarray): y surface
        z (np.ndarray): z surface
    """

    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
    x = rad * np.cos(u) * np.sin(v)
    y = rad * np.sin(u) * np.sin(v)
    z = rad * np.cos(v)

    return (x, y, z)


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


def local_sidereal_time(date: datetime, east_long: float) -> float:
    """
    Compute local sidereal time from date
    Adapted from Algorithm 5.3 in "Orbital Mechanics for Engineering Students", Curtis

    Args:
        date (datetime): calendar date to compute lst from
        east_long (float): east longitude of site from greenwich

    Returns:
        lst (float): local sidereal time at longitude of site

    """

    # calc julian date
    J_0 = juliandate(date)

    # calc greenwich sidereal time
    T_0 = (J_0 - 2_451_545) / (36_525)

    # eqn. 5.50 in curtis
    theta_g0 = (
        100.4606184 + 36_000.77004 * T_0 + 0.000387933 * T_0**2 - 2.583e-8 * T_0**3
    )
    theta_g0 = np.mod(theta_g0, 360)

    # eqn 5.51 in curtis
    theta_g = theta_g0 + 360.98564724 * date.hour / 24

    # eqn 5.52 in curtis
    lst = theta_g + east_long

    return lst


def coes_to_statevector(coes: COES, mu: float = ast.EARTH_MU) -> np.ndarray:
    """
    Convert COES to State Vector: R, V

    Args:
        coes (COES): orbital COES
        mu (float, optional): Gravitational parameter of central body.
        Defaults to ast.EARTH_MU.

    Returns:
        np.ndarray: 6x1 state vector: [Rx, Ry, Rz, Vx, Vy, Vz]
    """
    h: float = coes.h
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


def statevector_to_coes(
    state: Union[List, np.ndarray, StateVector], mu: int = ast.EARTH_MU
) -> COES:
    """
    Calculate Classical Orbital Elements from State Vector
    Adapted from Algorithm 9: "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        state (List, np.ndarray, StateVector): state vector (rx, ry, rz, vx, vy, vz)
        mu (int, optional): central body grav parameter. Defaults to EARTH_MU.

    Returns:
        COES: COES object containing orbital information
    """

    # convert to array if in state class
    if isinstance(state, StateVector):
        state = state.to_arr()

    # convert to np array if List
    state = np.array(state)
    r = np.array(state[:3])
    v = np.array(state[3:])
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

    rp = h**2 / ast.EARTH_MU * 1 / (1 + ecc)
    ra = h**2 / ast.EARTH_MU * 1 / (1 - ecc)
    semi = 0.5 * (ra + rp)

    return COES(ecc, inc, raan, w, theta, h, semi)


def tle_to_coes(tle: np.ndarray, mu: int = ast.EARTH_MU) -> COES:
    """
    Convert two-line element to classical orbital elements.
    Requires some user-supplied work to format TLE properly; only second line required

    Args:
        tle (np.ndarray): second line of TLE in following format:
            inclination (deg),
            raan (deg),
            eccentricity (-),
            argument of perigee (deg),
            mean anomaly (deg),
            mean motion (rev/day),

            the second line is already formatted properly with the correct units
        mu (int): gravitational parameter of central body. Defaults to Earth

    Returns:
        COES: COES object described by TLE

    """

    [inc_deg, raan_deg, ecc, arg_deg, anom_deg, mean_mot] = tle

    # convert from deg to rad
    inc_rad = np.deg2rad(inc_deg)
    raan_rad = np.deg2rad(raan_deg)
    arg_rad = np.deg2rad(arg_deg)

    Me = np.deg2rad(anom_deg)

    if Me < np.pi:
        E_0 = Me - ecc

    else:
        E_0 = Me + ecc

    # functions to compute theta using Newton's Iteration
    def f(E: float) -> float:
        """
        Eqn 1 for Newton's Iteration
        """
        return Me - E + ecc * np.sin(E)

    def fp(E: float) -> float:
        """
        Eqn 2 for Newton's Iteration
        """
        return -1 + ecc * np.sin(E)

    # calc dE_0, error
    err = 1
    count = 1
    while err > 1e-8 and count < 1_000:
        E_1 = E_0 - (f(E_0) / fp(E_0))
        err = np.abs(E_1 - E_0)
        E_0 = E_1
        count += 1

    # compute theta
    theta = 2 * np.arctan((np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E_0 / 2)))
    theta = np.mod(theta, 2 * np.pi)

    # calc period of orbit
    T = (24 * 3_600 * 3_600) * 1 / mean_mot

    # compute semi-major axis
    a = (T * np.sqrt(mu) / (2 * np.pi)) ** (2 / 3)

    # compute ang mom
    h = np.sqrt(a * mu * (1 - ecc**2))

    return COES(ecc, inc_rad, raan_rad, arg_rad, theta, h, a)


"""
CORE FUNCTIONS
"""


def two_body(time: float, state: np.ndarray, mu: int = ast.EARTH_MU) -> np.ndarray:
    """
    Basic Two-Body Propagation; to be used in ODE function
    Args:
        time[float]: time argument for ODE
        state[np.ndarray]: state vector of two body propagation
            in form of [Rx, Ry, Rz, Vx, Vy, Vz]
        mu[float, Optional]: gravitational parameter of central body.
            Defaults to EarthMU

    Returns:
        np.ndarray: derivative of state vector in two-body orbit
    """
    _r_vec = state[:3]
    _v_vec = state[3:]
    _r_norm = np.linalg.norm(_r_vec)
    _a_vec = -mu * _r_vec / (_r_norm**3)

    _dstate = np.array([*_v_vec, *_a_vec])
    return _dstate


def solve_two_body(
    state0: Union[np.ndarray, StateVector],
    tspan: np.ndarray,
    mu: int = ast.EARTH_MU,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> pd.DataFrame:
    """
    ODE45 to solve two-body problem with tolerance and gravitational parameter arguments
    Formats output into dataframe for simple extraction; can be extracted with
        `frame.TIME, frame.RX, frame.RY, etc.` to extract arrays from dataframe

    Args:
        state0 (Union[np.ndarray, StateVector]): initial state either as 6-element
            vector or StateVector class
        tspan (np.ndarray): 2-element vector for initial and final time
        mu (int, Optional): Gravitational parameter of central body. Defaults to Earth.
        atol (float): Absolute tolerance for ODE45. Defaults to 1e-8
        rtol (float): Relative tolerance for ODE45. Defaults to 1e-8

    Returns:
        pd.DataFrame: dataframe of time and state history of orbit given conditions

    """

    # convert to array if in state vector form
    if isinstance(state0, StateVector):
        state0 = state0.to_arr()

    # solve IVP
    ode_sol: OdeResult = solve_ivp(
        two_body, tspan, state0, args=(mu,), atol=atol, rtol=rtol
    )

    time = ode_sol["t"]
    state = ode_sol["y"]

    [rx, ry, rz, vx, vy, vz] = state

    data = {"TIME": time, "RX": rx, "RY": ry, "RZ": rz, "VX": vx, "VY": vy, "VZ": vz}

    return pd.DataFrame(data)


def stumpff_S(z: float) -> float:
    """
    Stumpff S Function

    Args:
        z (float): Z-Value

    Returns:
        float: Result of Function
    """

    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z) ** 3)
    if z < 0:
        return ((np.sinh(np.sqrt(-z)) - np.sqrt(-z))) / (np.sqrt(z) ** 3)
    return 1 / 6


def stumpff_C(z: float) -> float:
    """
    Stumpff C Function

    Args:
        z (float): Z-Value

    Returns:
        float: Result of Function
    """

    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    if z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)

    return 1 / 2


def la_grange_coefficients(
    alpha: float,
    R0: np.ndarray,
    V0: np.ndarray,
    delta_t: float,
    X: float,
    mu: int = ast.EARTH_MU,
) -> np.ndarray:
    """
    Method to compute LaGrange Coefficients: f, g, fdot, gdot
    Args:
        alpha (float): alpha value
        R0 (np.ndarray): position vector
        V0 (np.ndarray): velocity vector
        delta_t (float): time delta
        X (float): Universal Variable
        mu (int): Gravitational parameter of central body

    Returns:
        np.ndarray: LaGrange Coefficients in order [f, g, fdot, gdot]
    """

    r0 = np.linalg.norm(R0)

    f = 1 - X**2 / r0 * stumpff_C(alpha * X**2)
    g = delta_t - 1 / np.sqrt(mu) * X**3 * stumpff_S(alpha * X**2)

    rf_vec = f * R0 + g * V0
    rf = np.linalg.norm(rf_vec)

    fdot = np.sqrt(mu) / (rf * r0) * (alpha * X**3 * stumpff_S(alpha * X**2) - X)
    gdot = 1 - X**2 / rf * stumpff_C(alpha * X**2)

    return np.array([f, g, fdot, gdot])


def lambert_problem_solver(
    R0: np.ndarray, R1: np.ndarray, delta_t: float, *, traj_type: int = 1
) -> List[np.ndarray]:
    """
    Lambert's Problem Solver:
        Solve for Velocity at Time 1, V1, and at Time 2, V2
        Given Position at Time 1, R1, and at Time 2, R2,
        and delta-T

    Adapted from Alg D.25 from "Orbital Mechanics for Engineering Students", Curtis

    Args:
        R0 (np.ndarray):
        R1 (np.ndarray):
        delta_t (float):
        traj_type (int): Prograde (1) or Retrograde (-1). Default Prograde

    Returns:
        List[np.ndarray]: 2-element list; first element is Velocity at Time 1
            second element is Velocity at Time 2
    """

    R0 = np.array(R0)
    R1 = np.array(R1)

    r1 = np.sqrt(R0.dot(R0))
    r2 = np.sqrt(R1.dot(R1))

    c12 = np.cross(R0, R1)
    theta = np.arccos(R0.dot(R1) / (r1 * r2))

    # assert theta based on progression of orbit
    if traj_type == 1:
        if c12[2] <= 0:
            theta = 2 * np.pi - theta

    if traj_type == -1:
        if c12[2] >= 0:
            theta = 2 * np.pi - theta

    A = np.sin(theta) * np.sqrt(r1 * r2 / (1 - np.cos(theta)))

    z = -100

    raise NotImplementedError("TODO: Implement Lambert Problem Solver")
    return


def universal_variable_propagation(
    R0: Union[np.ndarray, List],
    V0: Union[np.ndarray, List],
    delta_t: float,
    mu: int = ast.EARTH_MU,
) -> np.ndarray:
    """
    Universal Variable Method for Orbit Propagation
    Adapted from Alg. 3.4 from "Orbital Mechanics for Engineering Students", Curtis

    Args:
        R0 (np.ndarray): Position at Time 1
        V0 (np.ndarray): Position at Time 2
        delta_t (float): Time of propagation
        mu (int): Gravitational parameter of central body

    Returns:
        np.ndarray: State Vector at final time with format:
            [Rx, Ry, Rz, Vx, Vy, Vz]
    """

    def compute_universal_variable(
        r0: float, vr0: float, alpha: float, delT: float, mu: int
    ) -> float:
        """
        Inner function to compute universal variable, X
        """
        abs_tol = 1e-8
        nMax = 1000
        n = 0
        ratio = 1

        # initial guess
        x = np.sqrt(mu) * np.abs(alpha) * delT

        # Newton's Iteration to find true X
        while np.abs(ratio) > abs_tol and n < nMax:
            n += 1
            C = stumpff_C(alpha * x**2)
            S = stumpff_S(alpha * x**2)

            F = (
                r0 * vr0 / np.sqrt(mu) * x**2 * C
                + (1 - alpha * r0) * x**3 * S
                + r0 * x
                - np.sqrt(mu) * delT
            )

            Fp = (
                r0 * vr0 / np.sqrt(mu) * x * (1 - alpha * x**2 * S)
                + (1 - alpha * r0) * x**2 * C
                + r0
            )

            ratio = F / Fp
            x = x - ratio

        return x

    R0 = np.array(R0)
    V0 = np.array(V0)

    r0 = np.sqrt(R0.dot(R0))
    v0 = np.sqrt(V0.dot(V0))
    vr0 = V0.dot(R0) / r0

    alpha = 2 / r0 - v0**2 / mu

    X = compute_universal_variable(r0, vr0, alpha, delta_t, mu)
    la_grange_coeff = la_grange_coefficients(alpha, R0, V0, delta_t, X, mu)

    [f, g, fdot, gdot] = la_grange_coeff

    RF = f * R0 + g * V0
    VF = fdot * R0 + gdot * V0

    return np.array([*RF, *VF])
