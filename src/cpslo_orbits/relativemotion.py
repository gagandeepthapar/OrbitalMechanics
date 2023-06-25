"""
relativemotion
Module concerned with relative motion and rendezvous mechanics
"""

from typing import List, Union

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from . import astroconsts as ast
from .orbitalcore import ivp_result_to_dataframe

"""
CONVERSIONS
"""


def eci_to_lvlh(
    target_R: Union[List, np.ndarray], target_V: Union[List, np.ndarray]
) -> np.ndarray:
    """
    Computes rotation matrix from ECI to LVLH ref. frames
    Adapted from Eqn. 7.1-7.3 from "Orbital Mechanics for Engineers", Curtis

    Args:
        target_R (np.ndarray): target position (ECI)
        target_V (np.ndarray): target velocity (ECI)

    Returns:
        Q (np.ndarray): 3x3 rotation matrix to rotate from ECI to LVLH
    """

    target_R = np.array(target_R)
    target_V = np.array(target_V)

    h = np.cross(target_R, target_V)
    i_hat = target_R / np.sqrt(target_R.dot(target_R))
    k_hat = h / np.sqrt(h.dot(h))
    j_hat = np.cross(k_hat, i_hat)

    Q = np.array([[*i_hat], [*j_hat], [*k_hat]])

    return Q


def five_term_acceleration(
    target_R: Union[List, np.ndarray],
    target_V: Union[List, np.ndarray],
    chaser_R: Union[List, np.ndarray],
    chaser_V: Union[List, np.ndarray],
    mu: int = ast.EARTH_MU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Five-Term Acceleration for Relative Motion between spacecraft
    Adapted from Alg. 7.1 from "Orbital Mechanics for Engineers", Curtis

    Args:
        target_R (Union[List, np.ndarray]): position of target (ECI)
        target_V (Union[List, np.ndarray]): velocity of target (ECI)
        chaser_R (Union[List, np.ndarray]): position of chaser (ECI)
        chaser_V (Union[List, np.ndarray]): velocity of chaser (ECI)

    Returns:
        relR (np.ndarray): relative position of chaser w.r.t. target (LVLH)
        relV (np.ndarray): relative velocity of chaser w.r.t. target (LVLH)
        relA (np.ndarray): relative acceleration of chaser w.r.t. target (LVLH)
    """

    # renaming + creating np arrays
    RT = np.array(target_R)
    VT = np.array(target_V)
    RC = np.array(chaser_R)
    VC = np.array(chaser_V)

    Q = eci_to_lvlh(RT, VT)

    Omega = np.cross(RT, VT) / (RT.dot(RT))
    OmegaDot = -2 * VT.dot(RT) * Omega / (RT.dot(RT))

    # abs accels
    AT = -1 * mu * RT / (RT.dot(RT)) ** (3 / 2)
    AC = -1 * mu * RC / (RC.dot(RC)) ** (3 / 2)

    # rel position
    delR = RC - RT

    # rel velocity
    delV = VC - VT - (np.cross(Omega, delR))

    # rel accel
    delA = (
        AC
        - AT
        - np.cross(OmegaDot, delR)
        - np.cross(Omega, np.cross(Omega, delR))
        - np.cross(2 * Omega, delV)
    )

    # convert to lvlh
    relR = Q @ delR
    relV = Q @ delV
    relA = Q @ delA

    return (relR, relV, relA)


def clohessy_wiltshire_relmot(
    delR0: Union[List, np.ndarray],
    delV0: Union[List, np.ndarray],
    mean_mot: float,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clohessy-Wiltshire Equations for relative motion in a circular orbit
    Adapted from Eqn 7.53 in "Orbital Mechanics for Engineering Students", Curtis

    Args:
        delR0 (np.ndarray): relative position of chaser
        delV0 (np.ndarray): relative velocity of chaser
        mean_mot (float): mean motion of bodies [rad/sec]
        delta_t (float): time to propagate relative motion

    Returns:
        delRF (np.ndarray): relative position of chaser at time end
        delVF (np.ndarray): relative velocity of chaser at time end
    """

    # convert to np arrays
    delR0 = np.array(delR0)
    delV0 = np.array(delV0)

    n = mean_mot
    nt = mean_mot * delta_t

    # eqns. 7.53a - 7.53d in Curtis
    Psi_rr = np.array(
        [[4 - 3 * np.cos(nt), 0, 0], [6 * (np.sin(nt) - nt), 1, 0], [0, 0, np.cos(nt)]]
    )

    Psi_rv = np.array(
        [
            [1 / n * np.sin(nt), 2 / n * (1 - np.cos(nt)), 0],
            [2 / n * (np.cos(nt) - 1), 1 / n * (4 * np.sin(nt) - 3 * nt), 0],
            [0, 0, 1 / n * np.sin(nt)],
        ]
    )

    Psi_vr = np.array(
        [
            [3 * n * np.sin(nt), 0, 0],
            [6 * n * (np.cos(nt) - 1), 0, 0],
            [0, 0, -n * np.sin(nt)],
        ]
    )

    Psi_vv = np.array(
        [
            [np.cos(nt), 2 * np.sin(nt), 0.0],
            [-2 * np.sin(nt), 4 * np.cos(nt) - 3, 0.0],
            [0.0, 0.0, np.cos(nt)],
        ]
    )

    delRF = Psi_rr @ delR0 + Psi_rv @ delV0
    delVF = Psi_vr @ delR0 + Psi_vv @ delV0

    return (delRF, delVF)


"""
RELATIVE MOTION PROPAGATION
"""


def linearized_relative_motion(
    time: float, state: np.ndarray, mu: int = ast.EARTH_MU
) -> np.ndarray:
    """
    Function for ODE to compute relative motion of chaser and propagate target orbit
    Adapted from Eqn. 7.36 from "Orbital Mechanics for Engineers", Curtis

    Args:
        time (float): time parameter for ODE function
        state (np.ndarray): statevector of target/chaser system setup as follows:
            [ target RX,
              target RY,
              target RZ,
              target VX,
              target VY,
              target VZ,
              chaser delRX,
              chaser delRY,
              chaser delRZ,
              chaser delVX,
              chaser delVY,
              chaser delVZ
            ]
        mu (int): gravitational parameter of central body. Defaults to Earth

    Returns:
        dstate (np.ndarray): time derivative of state vector
    """

    # unpack
    t_r_vec = state[:3]
    t_v_vec = state[3:6]
    c_dr_vec = state[6:9]
    c_dv_vec = state[9:]

    # propate target with standard two-body
    dt_r_vec = t_v_vec
    dt_v_vec = -mu * t_r_vec / (np.linalg.norm(t_r_vec) ** 3)

    # update chaser relative motion
    dc_dr_vec = c_dv_vec

    R = np.linalg.norm(t_r_vec)
    h = np.linalg.norm(np.cross(t_r_vec, t_v_vec))

    ddrx = (
        (2 * mu / (R**3) + h**2 / (R**4)) * c_dr_vec[0]
        - (2 * h * t_v_vec.dot(t_r_vec) * c_dr_vec[1]) / (R**4)
        + 2 * h * c_dv_vec[1] / (R**2)
    )
    ddry = (
        (h**2 / (R**4) - mu / (R**3)) * c_dr_vec[1]
        + (2 * h * t_v_vec.dot(t_r_vec) * c_dr_vec[0]) / (R**4)
        - 2 * h * c_dv_vec[0] / (R**2)
    )
    ddrz = -mu * c_dr_vec[2] / (np.linalg.norm(R) ** 3)
    dc_dv_vec = np.array([ddrx, ddry, ddrz])

    return np.array([*dt_r_vec, *dt_v_vec, *dc_dr_vec, *dc_dv_vec])


def solve_linearized_relative_motion(
    target_R: Union[List, np.ndarray],
    target_V: Union[List, np.ndarray],
    chaser_delR: Union[List, np.ndarray],
    chaser_delV: Union[List, np.ndarray],
    tspan: Union[List, np.ndarray],
    mu: int = ast.EARTH_MU,
) -> pd.DataFrame:
    """
    Solve linearized relative motion problem
    Adapted from Eqn. 7.36 from "Orbital Mechanics for Engineers", Curtis

    Args:
        target_R (Union[List, np.ndarray): target position at t0
        target_V (Union[List, np.ndarray): target velocity at t0
        chaser_delR (Union[List, np.ndarray): chaser rel position at t0
        chaser_delV (Union[List, np.ndarray): chaser rel velocity at t0
        tspan (Union[List, np.ndarray): time span to propagate

    Returns:
        dataframe: Dataframe of state history for target and chaser
    """

    state0 = np.array([*target_R, *target_V, *chaser_delR, *chaser_delV])
    ivp_sol = solve_ivp(linearized_relative_motion, tspan, state0, atol=1e-8, rtol=1e-8)

    labels = [
        "TargetRX",
        "TargetRY",
        "TargetRZ",
        "TargetVX",
        "TargetVY",
        "TargetVZ",
        "ChaserDelRX",
        "ChaserDelRY",
        "ChaserDelRZ",
        "ChaserDelVX",
        "ChaserDelVY",
        "ChaserDelVZ",
    ]

    fdata = ivp_result_to_dataframe(ivp_sol, labels)
    return fdata


def continuous_thrust_relmot(
    time: float, state: np.ndarray, n: float, thrust: float, mu: int = ast.EARTH_MU
) -> np.ndarray:
    """
    Function for ODE to compute relative motion of chaser and propagate target orbit
    Adapted from Eqn. 7.36 from "Orbital Mechanics for Engineers", Curtis
        and AERO 452 Notes

    Args:
        time (float): time parameter for ODE function
        state (np.ndarray): statevector of target/chaser system setup as follows:
            [ target RX,
              target RY,
              target RZ,
              target VX,
              target VY,
              target VZ,
              chaser delRX,
              chaser delRY,
              chaser delRZ,
              chaser delVX,
              chaser delVY,
              chaser delVZ
            ]
        n (float): mean motion of target [rev/day]
        thrust (float): thrust magnitude of chaser [km/s]
        mu (int): gravitational parameter of central body. Defaults to Earth

    Returns:
        dstate (np.ndarray): time derivative of state vector
    """

    # unpack
    t_r_vec = state[:3]
    t_v_vec = state[3:6]
    c_dr_vec = state[6:9]
    c_dv_vec = state[9:]

    # propate target with standard two-body
    dt_r_vec = t_v_vec
    dt_v_vec = -mu * t_r_vec / (np.linalg.norm(t_r_vec) ** 3)

    # update chaser relative motion
    dc_dr_vec = c_dv_vec

    R = np.linalg.norm(t_r_vec)
    h = np.linalg.norm(np.cross(t_r_vec, t_v_vec))

    ddrx = (
        (2 * mu / (R**3) + h**2 / (R**4)) * c_dr_vec[0]
        - (2 * h * t_v_vec.dot(t_r_vec) * c_dr_vec[1]) / (R**4)
        + 2 * h * c_dv_vec[1] / (R**2)
        + 2 * n * thrust
    )
    ddry = (
        (h**2 / (R**4) - mu / (R**3)) * c_dr_vec[1]
        + (2 * h * t_v_vec.dot(t_r_vec) * c_dr_vec[0]) / (R**4)
        - 2 * h * c_dv_vec[0] / (R**2)
    )
    ddrz = -mu * c_dr_vec[2] / (np.linalg.norm(R) ** 3)
    dc_dv_vec = np.array([ddrx, ddry, ddrz])

    return np.array([*dt_r_vec, *dt_v_vec, *dc_dr_vec, *dc_dv_vec])


def canonical_propagation(
    time: float, state: Union[List, np.ndarray], muStar: float = ast.EARTH_MUSTAR
) -> np.ndarray:
    """
    Function for ODE Solver to propagate orbit in Canonical units
    Adapted from AERO 452 Notes

    Args:
        time (float): time parameter for ODE
        state (Union[List, np.ndarray]): state vector for canonical system formatted as:
            [
            Rx,
            Ry,
            Rz,
            Vx,
            Vy,
            Vz
            ]
            Most canonial systems in class are in 2-D and RZ, VZ, can be set to 0

    Returns:
        np.ndarray: time derivative of state
    """

    R = state[:3]
    V = state[3:]

    # define R1 and R2 in right-handed system
    R1 = np.sqrt((R[0] - muStar) ** 2 + R[1] ** 2 + R[2] ** 2)
    R2 = np.sqrt((R[0] + 1 - muStar) ** 2 + R[1] ** 2 + R[2] ** 2)

    # define time derivatives
    dR = V
    dVx = (
        -1 * (1 - muStar) * (R[0] - muStar) / R1**3
        - muStar * (R[0] + 1 - muStar) / R2**3
        + R[0]
        + 2 * V[1]
    )
    dVy = -1 * (1 - muStar) * R[1] / R1**3 - muStar * R[1] / R2**3 + R[1] - 2 * V[0]
    dVz = -1 * (1 - muStar) * R[2] / R1**3 - muStar * R[2] / R2**3

    return np.array([*dR, dVx, dVy, dVz])
