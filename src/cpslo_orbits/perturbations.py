"""
perturbations
Module concerned with perturbations in orbit 

NOTE:
Many of the concepts encapsulated in this module will not work with the 
    previous ODE's supplied.
Users should write a new ODE function and call the corresponding perturbations.
This is due to the permutations of different perturbations users can implement 
    and the variety of inputs required.
Users should be able to use the basic two-body function as a stepping stone.

Each perturbation method will note how to implement it within an ODE function
"""

from functools import wraps
from typing import List, Union, Optional, Callable

import numpy as np
from scipy.integrate import solve_ivp

from . import astroconsts as ast
from .orbitalcore import (
    COES,
    coes_to_statevector,
    rot_x,
    rot_z,
    sind,
    cosd,
    statevector_to_coes,
    two_body,
    universal_variable_propagation,
)


def event_listener():
    """
    Custom decorator to set direction and terminal values for event handling
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.direction = -1
        wrapper.terminal = True
        return wrapper

    return decorator


"""
PERTURBATION METHODS
"""


@event_listener()
def drag_event_listener(time: float, state: Union[List, np.ndarray], *opts):
    """
    Event listener for `solve_ivp` to quit integration when alt below 100km
    """
    alt = np.linalg.norm(state[:3]) - ast.EARTH_RAD
    return alt - 100


def atmos_density(alt: float) -> float:
    """
    Method to estimate atmospheric density for LEO Drag Model
    Adapted from Appendix D.25 in Curtis

    Args:
        alt (float): altitude of satellite. Should be < 1000km

    Returns:
        rho (float): estimated atmospheric density (kg/m3)
    """

    # ...Geometric altitudes (km):
    h = [
        0,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        180,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        600,
        700,
        800,
        900,
        1000,
    ]

    # ...Corresponding densities (kg/m^3) from USSA76:
    r = [
        1.225,
        4.008e-2,
        1.841e-2,
        3.996e-3,
        1.027e-3,
        3.097e-4,
        8.283e-5,
        1.846e-5,
        3.416e-6,
        5.606e-7,
        9.708e-8,
        2.222e-8,
        8.152e-9,
        3.831e-9,
        2.076e-9,
        5.194e-10,
        2.541e-10,
        6.073e-11,
        1.916e-11,
        7.014e-12,
        2.803e-12,
        1.184e-12,
        5.215e-13,
        1.137e-13,
        3.070e-14,
        1.136e-14,
        5.759e-15,
        3.561e-15,
    ]
    # ...Scale heights (km):
    H = [
        7.310,
        5.927,
        21.652,
        60.980,
        6.427,
        6.546,
        7.360,
        8.342,
        7.583,
        6.661,
        5.533,
        5.703,
        6.782,
        9.973,
        13.243,
        16.322,
        27.974,
        34.934,
        43.342,
        49.755,
        54.513,
        58.019,
        65.654,
        76.377,
        100.587,
        147.203,
        208.020,
    ]
    # ...Handle altitudes outside of the range:

    if alt > 1000:
        alt = 1000

    elif alt < 0:
        alt = 0

    # ...Determine the interpolation interval:
    i = 0
    for j in range(26):
        if alt >= h[j] and alt < h[j + 1]:
            i = j

    if alt == 1000:
        i = 26
    # ...Exponential interpolation:

    rho = r[i] * np.exp(-(alt - h[i]) / H[i])
    return rho


def drag_perturbation(
    time: float,
    state: np.ndarray,
    mu: int = ast.EARTH_MU,
    mass: float = 10,
    surf_area: float = 1,
    coeff_drag: float = 2.2,
) -> np.ndarray:
    """
    LEO Drag model
    Adapted from Eqn. 10.12 from "Orbital Mechanics for Engineers", Curtis

    Args:
        time (float): time param for ODE call (not used)
        state (np.ndarray): state vector of satellite [km]
        mu (int): gravitational parameter of central body. Defaults to Earth.
            (not used)
        mass (float): Mass of satellite [kg]. Defaults to 10
        surf_area (float): Average surface area in ram direction [m2]. Defaults to 1
        coeff_drag (float): Coefficient of drag. Defaults to 2.2

    Returns:
        a_drag: perturbations due to drag: [ax, ay, az]
    """

    # create np arrays
    R = state[:3]
    V = state[3:]

    # calc local density
    alt = np.sqrt(R.dot(R)) - ast.EARTH_RAD
    rho = atmos_density(alt)

    # calc local velocity rel to earth
    Vrel = V - np.cross(ast.EARTH_ANGVEL, R)
    vrel = np.sqrt(Vrel.dot(Vrel))

    # compute a_drag in m/s2
    a_drag = (
        -0.5 * coeff_drag * surf_area / mass * rho * (1000 * vrel) ** 2 * (Vrel / vrel)
    )

    return a_drag / 1000


def solar_position(juliandate: float) -> np.ndarray:
    """
    Method to compute position of sun with respect to Earth
    Adapted from AERO 452 resources/Kira Abercromby Notes

    Args:
        juliandate (float): juliandate of computation

    Returns:
        r_S (np.ndarray): position of sun w.r.t. Earth
    """

    jd = juliandate

    # ...Astronomical unit (km):
    AU = 149597870.691

    # ...Julian days since J2000:
    n = jd - 2451545

    # ...Mean anomaly (deg{:
    M = 357.528 + 0.9856003 * n
    M = np.mod(M, 360)

    # ...Mean longitude (deg):
    L = 280.460 + 0.98564736 * n
    L = np.mod(L, 360)

    # ...Apparent ecliptic longitude (deg):
    lamda = L + 1.915 * sind(M) + 0.020 * sind(2 * M)
    lamda = np.mod(lamda, 360)

    # ...Obliquity of the ecliptic (deg):
    eps = 23.439 - 0.0000004 * n

    # ...Unit vector from earth to sun:
    u = np.array([cosd(lamda), sind(lamda) * cosd(eps), sind(lamda) * sind(eps)])

    # ...Distance from earth to sun (km):
    rS = (1.00014 - 0.01671 * cosd(M) - 0.000140 * cosd(2 * M)) * AU

    # ...Geocentric position vector (km):
    r_S = rS * u

    return r_S


def shadow_function(
    r_earth_sc: Union[List, np.ndarray],
    juliandate: float,
) -> bool:
    """
    Function to determine if satellite in Earth shadow or exposed to Sun
    Adapted from Algorithm 10.3 from "Orbtial Mechanics for Engineers", Curtis

     Args:
        r_earth_sc (Union[List, np.ndarray]): position of sat w.r.t. Earth
        juliandate (float): juliandate for satellite for solar position

    Returns:
        shadow (bool): True if satellite is eclipsed. False if satellite is exposed.
    """

    # conv to np arrays
    r_earth_sc = np.array(r_earth_sc)
    r_earth_sun = solar_position(juliandate)

    # get magnitude of positions
    r_esc = np.sqrt(r_earth_sc.dot(r_earth_sc))
    r_esu = np.sqrt(r_earth_sun.dot(r_earth_sun))

    # calc view angles (eqn. 10.113, 10.114)
    theta = np.arccos((r_earth_sun.dot(r_earth_sc)) / (r_esc * r_esu))
    theta1 = np.arccos(ast.EARTH_RAD / r_esc)
    theta2 = np.arccos(ast.EARTH_RAD / r_esu)

    return theta1 + theta2 < theta


def solar_radiation_perturbation(
    time: float,
    state: np.ndarray,
    mu: int = ast.EARTH_MU,
    juliandate: float = 2_451_545,
    surf_area: float = 1,
    mass: float = 10,
    coeff_ref: float = 1,
) -> np.ndarray:
    """
    Computation of perturbations due to solar radiation pressure
    Adapted from Alg. 10.3, Eqn. 10.98 from Curtis

    Args:
        r_earth_sc (Union[List, np.ndarray]): position of sat w.r.t. Earth
        juliandate (float): start date in juliandate format of computation
        surf_area (float): surface area of satellite [m2]. Defaults to 1.
        mass (float): mass of satellite [kg]. Defaults to 10
        coeff_ref (float): Reflectivity of satellite. Defaults to 1

    Returns:
        a_pert (np.ndarray): disturbance acceleration due to SRP:
            [ax, ay, az]
    """

    # extract required data from args
    r_earth_sc = state[:3]

    # convert date with time argument
    juliandate = juliandate + time / (86_400)
    r_earth_sun = solar_position(juliandate)

    # get magnitude of positions
    r_esu = np.sqrt(r_earth_sun.dot(r_earth_sun))

    # compute perturbation
    a_pert = np.array([0, 0, 0])

    # determine if sat in shade or not (shadow function)
    eclipsed = shadow_function(r_earth_sc, juliandate)

    # if eclipsed, pert is 0
    if eclipsed:
        return a_pert

    # if exposed then pert must be calc'd
    a_pert = -ast.SUN_PSR * coeff_ref * surf_area / mass * (r_earth_sun / r_esu)
    return a_pert


def oblateness_perturbation(
    time: float,
    state: np.ndarray,
    mu: int = ast.EARTH_MU,
    zonal_number: int = 2,
    R_body: int = ast.EARTH_RAD,
) -> np.ndarray:
    """
    Orbital perturbations caused by oblateness in the Earth
        (J2-J6, only Zonal components)

    Args:
        time (float): time param for ODE call. (not used)
        state (np.ndarray): state vector of satellite.
        mu (int): gravitational parameter of central body. Defaults to Earth.
        zonal_number (int): granularity of which zonal numbers to comptue (2-6 only)
        R_body (int): radius of central body. Defaults to Earth

    Returns:
        accel (np.ndarray): acceleration due to oblateness:
            [ax, ay, az]
    """

    # confirm zonal number within bounds
    if zonal_number < 2 or 6 < zonal_number:
        raise ValueError("Invalid Zonal Number. Pick 2-6, inclusive.")

    # define R_sat
    R_sat = state[:3]

    # define J2-6 Zonal defs
    def J2(
        r_sat: np.ndarray,
        r_body: int = ast.EARTH_RAD,
        mu: int = ast.EARTH_MU,
    ):
        """
        Perturbations due to oblateness from Zonal Number 2
        """
        # particular J term
        J = 1.08262668355e-3

        # rename variables
        rs = r_sat
        r = np.linalg.norm(r_sat)
        rb = r_body

        # eqns based on curtis,
        mult = -3 * J * mu * rb**2 / (2 * r**5)

        ax = mult * rs[0] * (1 - (5 * rs[2] ** 2 / r**2))
        ay = mult * rs[1] * (1 - (5 * rs[2] ** 2 / r**2))
        az = mult * rs[2] * (3 - (5 * rs[2] ** 2 / r**2))

        # combine accel model
        J2a = np.array([ax, ay, az])
        return J2a

    def J3(
        r_sat: np.ndarray,
        r_body: int = ast.EARTH_RAD,
        mu: int = ast.EARTH_MU,
    ):
        """
        Perturbations due to oblateness from Zonal Number 3
        """
        # particular J term
        J = -2.53265648533e-6

        # rename variables
        rs = r_sat
        r = np.linalg.norm(r_sat)
        rb = r_body

        # eqns based on curtis,
        ax = (
            -5
            * J
            * mu
            * rb**3
            * rs[0]
            / (2 * r**7)
            * (3 * rs[2] - (7 * rs[2] ** 3 / r**2))
        )
        ay = (
            -5
            * J
            * mu
            * rb**3
            * rs[1]
            / (2 * r**7)
            * (3 * rs[2] - (7 * rs[2] ** 3 / r**2))
        )
        az = (
            -5
            * J
            * mu
            * rb**3
            / (2 * r**7)
            * (6 * rs[2] ** 2 - (7 * rs[2] ** 4 / r**2) - (3 * r**2 / 5))
        )

        # combine accel model
        J3a = np.array([ax, ay, az])
        return J3a

    def J4(
        r_sat: np.ndarray,
        r_body: int = ast.EARTH_RAD,
        mu: int = ast.EARTH_MU,
    ):
        """
        Perturbations due to oblateness from Zonal Number 4
        """
        # particular J term
        J = -1.61962159137e-6

        # rename variables
        rs = r_sat
        r = np.linalg.norm(r_sat)
        rb = r_body

        # eqns based on curtis,
        ax = (
            15
            * J
            * mu
            * rb**4
            * rs[0]
            / (8 * r**7)
            * (1 - (14 * rs[2] ** 2 / r**2) + (21 * rs[2] ** 4 / r**4))
        )
        ay = (
            15
            * J
            * mu
            * rb**4
            * rs[1]
            / (8 * r**7)
            * (1 - (14 * rs[2] ** 2 / r**2) + (21 * rs[2] ** 4 / r**4))
        )
        az = (
            15
            * J
            * mu
            * rb**4
            * rs[2]
            / (8 * r**7)
            * (5 - (70 * rs[2] ** 2 / (3 * r**2)) + (21 * rs[2] ** 4 / r**4))
        )

        # combine accel model
        J4a = np.array([ax, ay, az])
        return J4a

    def J5(
        r_sat: np.ndarray,
        r_body: int = ast.EARTH_RAD,
        mu: int = ast.EARTH_MU,
    ):
        """
        Perturbations due to oblateness from Zonal Number 5
        """
        # particular J term
        J = -2.27296082869e-7

        # rename variables
        rs = r_sat
        r = np.linalg.norm(r_sat)
        rb = r_body

        # eqns based on curtis,
        ax = (
            3
            * J
            * mu
            * rb**5
            * rs[0]
            * rs[2]
            / (8 * r**9)
            * (35 - (210 * rs[2] ** 2 / r**2) + (231 * rs[2] ** 4 / r**4))
        )
        ay = (
            3
            * J
            * mu
            * rb**5
            * rs[1]
            * rs[2]
            / (8 * r**9)
            * (35 - (210 * rs[2] ** 2 / r**2) + (231 * rs[2] ** 4 / r**4))
        )
        az = 3 * J * mu * rb**5 * rs[2] * rs[2] / (8 * r**9) * (
            105 - (315 * rs[2] ** 2 / r**2) + (231 * rs[2] ** 4 / r**4)
        ) - 15 * J * mu * rb**5 / (8 * r**7)

        # combine accel model
        J5a = np.array([ax, ay, az])
        return J5a

    def J6(
        r_sat: np.ndarray,
        r_body: int = ast.EARTH_RAD,
        mu: int = ast.EARTH_MU,
    ):
        """
        Perturbations due to oblateness from Zonal Number 6
        """
        J = 5.40681239107e-7

        # rename variables
        rs = r_sat
        r = np.linalg.norm(r_sat)
        rb = r_body

        # eqns based on curtis,
        ax = (
            -J
            * mu
            * rb**6
            * rs[0]
            / (16 * r**9)
            * (
                35
                - (945 * rs[2] ** 2 / r**2)
                + (3465 * rs[2] ** 4 / r**4)
                - (3003 * rs[2] ** 6 / r**6)
            )
        )
        ay = (
            -J
            * mu
            * rb**6
            * rs[1]
            / (16 * r**9)
            * (
                35
                - (945 * rs[2] ** 2 / r**2)
                + (3465 * rs[2] ** 4 / r**4)
                - (3003 * rs[2] ** 6 / r**6)
            )
        )
        az = (
            -J
            * mu
            * rb**6
            * rs[2]
            / (16 * r**9)
            * (
                245
                - (2205 * rs[2] ** 2 / r**2)
                + (4851 * rs[2] ** 4 / r**4)
                - (3003 * rs[2] ** 6 / r**6)
            )
        )

        # combine accel model
        J6a = np.array([ax, ay, az])
        return J6a

    # compile functions and compute as needed
    zonal_comps = [J2, J3, J4, J5, J6]
    zonal_number -= 1
    a_pert = np.array([0.0, 0.0, 0.0])

    for idx in range(zonal_number):
        a_pert += zonal_comps[idx](R_sat, R_body, mu)

    return a_pert


def n_body_perturbation(
    time: float,
    state: np.ndarray,
    mu: int,
    r_earth_body: np.ndarray,
    mu_body: int,
) -> np.ndarray:
    """
    Compute perturbation on orbit due to n-body effects
    Works for near-bodies (Moon) and far bodies (Sun, other planets)
    Adapted from Eqn. 10.117-10.131 from "Orbital Mechanics for Engineers", Curtis

    Args:
        time (float): time param for ODE call. (not used)
        state (np.ndarray): state vector of satellite.
        mu (int): gravitational parameter of central body.
        r_earth_body (Union[List, np.ndarray]): Position of body w.r.t. Earth
        mu_body (int): gravitational parameter of external body

    Returns:
        a_pert (np.ndarray): disturbance acceleration due to n-body effects:
            [ax, ay, az]
    """

    # extract data from args
    r_earth_sc = state[:3]
    r_earth_body = np.array(r_earth_body)
    r_body_sc = r_earth_body - r_earth_sc

    # calc magnitude of positions
    r_eb = np.sqrt(r_earth_body.dot(r_earth_body))
    r_sb = np.sqrt(r_body_sc.dot(r_body_sc))
    a_pert = np.array([0, 0, 0])

    # if body is sufficiently far away then need to rewrite relations
    # digital computer
    if np.isclose(r_sb / r_eb, 1.0, 1e-3, 1e-3):
        # eqn. 10.130, 10.131, F.3, F.4 in Curtis
        q = r_earth_sc.dot((2 * r_earth_body - r_earth_sc)) / (r_eb**2)
        Fq = (q**2 - 3 * q + 3) * q / (1 + (1 - q) ** (1.5))
        a_pert = mu_body / (r_sb**3) * (Fq * r_earth_body - r_earth_sc)

    # else can use standard notation
    else:
        a_pert = mu_body * (r_body_sc / (r_sb**3) - r_earth_body / (r_eb**3))

    return a_pert


"""
PROPAGATION TECHNIQUES

Modified versions of two-body propagation to include perturbations.
Each technique has an argument, `funcs` that is a list of tuples.
Each tuple requires the function and list of arguments
"""


def cowells_method(
    time: float,
    state: np.ndarray,
    mu: int = ast.EARTH_MU,
    perturbs: Optional[List[tuple[Callable, tuple]]] = None,
) -> np.ndarray:
    """
    Cowell's Method for Orbit Propagation.
    Very similar to two-body where disturbance acclerations are added onto two-body calc
    Adapted from Chapter 10.2 from "Orbital Mechanics for Engineers", Curtis

    Args:
        time (float): time param for ODE call
        state (np.ndarray): state vector of orbit:
            [Rx, Ry, Rz, Vx, Vy, Vz]
        mu (int): gravitational parameter of central body. Defaults to EARTH_MU.
        perturbs (Optional[List[tuple[callable, tuple]]]): List of perturbations
            Formatted as a *list* of tuples where the pair contains the function
            and then its associated arguments aside from time, state, mu. I.e.,:
                perturbs=[
                    (oblateness_perturbation, (zonal_num, r_body)),
                    (solar_radiation_perturbation, (juliandate, surf_area,...)),
                    ...
                    ]

    Returns:
        a_total (np.ndarray): acceleration of spacecraft after
            accounting for perturbations
    """
    # compute standard two_body
    a_two_body = two_body(time, state, mu)
    a_pert = np.array([0.0, 0.0, 0.0])

    # compute total perturbational acceleration
    if perturbs is not None:
        for func, args in perturbs:
            a_pert += func(time, state, mu, *args)

    # add peturbations into two_body and return
    a_total = a_two_body + np.array([0.0, 0.0, 0.0, *a_pert])
    return a_total


def enckes_ode(
    time: float,
    state: np.ndarray,
    mu: int,
    state_osc: np.ndarray,
    perturbs: Optional[List[tuple[Callable, tuple]]] = None,
) -> np.ndarray:
    """
    Encke's Method for ODE call
    Adapted from Appendix D.42 from "Orbital Mechanics for Engineers", Curtis

    Args:
        time (float): time param for ODE
        state (np.ndarray): state of orbit deviation:
            [delRx, delRy, delRz, delVx, delVy, delVz]
        mu (int): gravitational parameter for central body
        state_osc (np.ndarray): Osculating/reference orbit
        perturbs (Optional[List[tuple[callable, tuple]]]): List of perturbations
            Formatted as a *list* of tuples where the pair contains the function
            and then its associated arguments aside from time, state, mu. I.e.,:
                perturbs=[
                    (oblateness_perturbation, (zonal_num, r_body)),
                    (solar_radiation_perturbation, (juliandate, surf_area,...)),
                    ...
                    ]

    Returns:
        d_del (np.ndarray): delV, delA for orbit deviations
    """
    # unpack data
    del_r = state[:3]
    del_v = state[3:]
    R_osc = state_osc[:3]

    # calc perturbed orbit
    Rpp = R_osc + del_r

    r_osc = np.sqrt(R_osc.dot(R_osc))
    rpp = np.sqrt(Rpp.dot(Rpp))

    # compute perturbations
    a_perts = np.array([0.0, 0.0, 0.0])
    if perturbs is not None:
        for func, args in perturbs:
            a_perts += func(time, state_osc, mu, *args)

    # compute total pert
    F = 1 - (r_osc / rpp) ** 3
    del_a = -mu / r_osc**3 * (del_r - F * Rpp) + a_perts

    return np.array([*del_v, *del_a])


def solve_enckes_method(
    state: np.ndarray,
    tspan: Union[List, np.ndarray],
    mu: int = ast.EARTH_MU,
    perturbs: Optional[List[tuple[Callable, tuple]]] = None,
    events: Optional[List[Callable]] = None,
    steps: int = 100,
) -> np.ndarray:
    """
    Encke's Method for Orbit Propagation.
    Uses Two-Body orbit as reference (Osculating) and recrtifies based on
        changes in position, velocity due to perturbations
    Adapted from Chapter 10.1 from "Orbital Mechanics for Engineers", Curtis
    No need to pass into ODE

    Args:
        state (np.ndarray): state vector of orbit:
            [Rx, Ry, Rz, Vx, Vy, Vz]
        timespan (List, np.ndarray): timespan of propagation
        mu (int): gravitational parameter of central body. Defaults to EARTH_MU.
        perturbs (Optional[List[tuple[callable, tuple]]]): List of perturbations
            Formatted as a *list* of tuples where the pair contains the function
            and then its associated arguments aside from time, state, mu. I.e.,:
                perturbs=[
                    (oblateness_perturbation, (zonal_num, r_body)),
                    (solar_radiation_perturbation, (juliandate, surf_area,...)),
                    ...
                    ]

    Returns:
        state_history (np.ndarray): state history of orbit and perturbations
    """

    # set time params
    orbit = statevector_to_coes(state)
    del_t = orbit.T / steps

    # set initial conditions
    R0 = state[:3]
    V0 = state[3:]
    t0 = tspan[0]
    t = t0 + del_t / 2

    state_history = np.array([[*R0, *V0]])

    while t <= tspan[1] + del_t / 2:
        delstate = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        d_delstate = solve_ivp(
            enckes_ode,  # call enckes ode
            [t0, t],  # set timespan
            delstate,  # set initial state (all 0s)
            atol=1e-8,  # set tolerances
            rtol=1e-8,
            args=(  # pass in arguments
                mu,
                np.array([*R0, *V0]),
                perturbs,
            ),
            events=events,  # set events
        )

        # compute osculating elements
        state_osc = universal_variable_propagation(R0, V0, t - t0)
        R_osc = state_osc[:3]
        V_osc = state_osc[3:]

        # extract data
        d_delstate = d_delstate["y"].T

        # Rectify
        R0 = R_osc + d_delstate[-1, :3]
        V0 = V_osc + d_delstate[-1, 3:]
        new_state = [*R0, *V0]
        t0 = t

        # save state history
        state_history = np.append(state_history, np.array([new_state]), axis=0)
        t = t + del_t

    return state_history


def variation_of_params(
    time: float,
    state: np.ndarray,
    mu: int = ast.EARTH_MU,
    perturbs: Optional[List[tuple[Callable, tuple]]] = None,
) -> np.ndarray:
    """
    Variation of Parameters Method for Orbit Propagation.
    Propogates the orbital elements as opposed to the state vector
        Still requires conversion to state vector for perturbation calc
    Adapted from Chapter 10.7 from "Orbital Mechanics for Engineers", Curtis

    Args:
        time (float): time param for ODE call
        state (np.ndarray): 6-element state vector of orbit:
            [ecc, inc_rad, raan_rad, arg_peri_rad, theta_rad, h_km2s]

        mu (int): gravitational parameter of central body. Defaults to EARTH_MU.
        perturbs (Optional[List[tuple[callable, tuple]]]): List of perturbations
            Formatted as a *list* of tuples where the pair contains the function
            and then its associated arguments aside from time, state, mu. I.e.,:
                perturbs=[
                    (oblateness_perturbation, (zonal_num, r_body)),
                    (solar_radiation_perturbation, (juliandate, surf_area,...)),
                    ...
                    ]

    Returns:
        a_total (np.ndarray): acceleration of spacecraft after
            accounting for perturbations
    """

    # pack state array
    ecc = state[0]
    inc = np.mod(state[1], 2 * np.pi)
    raan = np.mod(state[2], 2 * np.pi)
    arg = np.mod(state[3], 2 * np.pi)
    theta = np.mod(state[4], 2 * np.pi)
    h = state[5]
    coes = COES(ecc, inc, raan, arg, theta, h=h)

    # ensure params are non-zero to avoid failure
    if coes.inc_rad == 0:
        coes.inc_rad = 1e-8

    if coes.ecc == 0:
        coes.ecc = 1e-8

    # calc helper params
    r = coes.h**2 / (coes.mu * (1 + coes.ecc * np.cos(coes.theta_rad)))

    # conv to statevector for pert calcs
    state_sv: np.ndarray = coes_to_statevector(coes)

    # calc perturbation accels
    a_pert = np.array([0.0, 0.0, 0.0])

    if perturbs is not None:
        for func, args in perturbs:
            a_pert += func(time, state_sv, mu, *args)

    # rotate a_pert from ECI to RSW frame
    q_bar_Xx = rot_z(coes.arg_peri_rad) @ rot_x(coes.inc_rad) @ rot_z(coes.raan_rad)
    q_bar_Xr = rot_z(coes.theta_rad) @ q_bar_Xx

    a_pert_RSW = q_bar_Xr @ a_pert

    # unpack a_pert into components
    [a_pr, a_ps, a_pw] = a_pert_RSW

    # begin d/dt of orbital elements; Eqn. 10.84 from Curtis
    d_ecc = (
        coes.h / mu * np.sin(coes.theta_rad) * a_pr
        + 1
        / (mu * coes.h)
        * ((coes.h**2 + mu * r) * np.cos(coes.theta_rad) + mu * coes.ecc * r)
        * a_ps
    )
    d_inc = r / coes.h * np.cos(coes.arg_peri_rad + coes.theta_rad) * a_pw
    d_raan = (
        r
        / (coes.h * np.sin(coes.inc_rad))
        * np.sin(coes.arg_peri_rad + coes.theta_rad)
        * a_pw
    )
    d_arg = (
        -1
        / (coes.ecc * coes.h)
        * (
            coes.h**2 / mu * np.cos(coes.theta_rad) * a_pr
            - (r + coes.h**2 / mu) * np.sin(coes.theta_rad) * a_ps
        )
        - (r * np.sin(coes.arg_peri_rad + coes.theta_rad))
        / (coes.h * np.tan(coes.inc_rad))
        * a_pw
    )
    d_theta = coes.h / r**2 + 1 / (coes.ecc * coes.h) * (
        coes.h**2 / mu * np.cos(coes.theta_rad) * a_pr
        - (r + coes.h**2 / mu) * np.sin(coes.theta_rad) * a_ps
    )
    d_h = r * a_ps

    # return d_state in correct form:
    # [ecc, inc_rad, raan_rad, arg_peri_rad, theta_rad, h_km2s]
    return np.array([d_ecc, d_inc, d_raan, d_arg, d_theta, d_h])
