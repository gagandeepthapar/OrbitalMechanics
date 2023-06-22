"""
relativemotion
Module concerned with relative motion and rendezvous mechanics
"""


import numpy as np
import pandas as pd

from . import astroconsts as ast


def eci_to_lvlh(target_R: np.ndarray, target_V: np.ndarray) -> np.ndarray:
    """
    Computes rotation matrix from ECI to LVLH ref. frames
    Adapted from AERO 452 Notes

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


def linearized_relative_motion():
    raise NotImplementedError
    return


def five_term_acceleration():
    raise NotImplementedError
    return


def clohessy_wiltshire_relmot(
    delR0: np.ndarray, delV0: np.ndarray, mean_mot: float, delta_t: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clohessy-Wiltshire Equations for relative motion in a circular orbit
    Adapted from Eqn 7.53 in "Orbital Mechanics for Engineering Students", Curtis

    Args:
        delR0 (np.ndarray): relative position of chaser
        delV0 (np.ndarray): relative velocity of chaser
        mean_mot (float): mean motion of bodies
        delta_t (float): time to propagate relative motion

    Returns:
        delRF (np.ndarray): relative position of chaser at time end
        delVF (np.ndarray): relative velocity of chaser at time end
    """

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
            [np.cos(nt), 2 * np.sin(nt), 0],
            [-2 * np.sin(nt), 4 * np.cos(nt) - 3, 0],
            [0, 0, np.cos(nt)],
        ]
    )

    delRF = Psi_rr @ delR0 + Psi_rv @ delV0
    delVF = Psi_vr @ delR0 + Psi_vv @ delV0

    return (delRF, delVF)


def continuous_thrust_relmot():
    raise NotImplementedError
    return


def canonical_propagation():
    raise NotImplementedError
    return
