"""
observations 
Core module for observations and filtering including some conversions 
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from . import astroconsts as ast
from .orbitalcore import (
    COES,
    juliandate,
    local_sidereal_time,
    rot_y,
    rot_z,
    stumpff_C,
    stumpff_S,
)


from pymap3d import eci2aer


"""
Observation dataclasses 
"""


@dataclass
class Site:
    """
    Dataclass to store site-specific information. Useful for references

    Args:
        lat (float): latitude of site [deg]
        long (float): longitude of site [deg]
        alt (float)
        date (datetime)


    """

    lat: float
    long: float
    alt: float
    date: datetime

    def __post_init__(self) -> None:
        self.jd = juliandate(self.date)
        self.lst = local_sidereal_time(self.date, self.long)
        self.ecef_pos = self.site_location_ecef(self.lat, self.long, self.alt)
        self.eci_pos = self.site_location_eci(self.lat, self.lst, self.alt)

    @staticmethod
    def site_location_ecef(geod_lat: float, long: float, alt: float) -> np.ndarray:
        """
        Calculate ECEF vector of the site
        Adapted from AERO 557 Notes

        Args:
            geod_lat (float): geodetic latitude [deg]
            long (float): longitude [deg]
            alt (float): altitude [m]

        Returns:
            np.ndarray: ECEF vector of the site
        """

        # update units of arguments
        geod_lat = np.deg2rad(geod_lat)
        long = np.deg2rad(long)
        alt = alt / 1_000

        Re = 6378
        Rp = 6357
        f = 1 - Rp / Re

        Cearth = Re / np.sqrt(
            1.0 - ((2.0 * f - f**2) * np.sin(geod_lat) * np.sin(geod_lat))
        )
        Searth = (1 - (2 * f - f**2)) * Cearth

        i_hat = (Cearth + alt) * np.cos(geod_lat) * np.cos(long)
        j_hat = (Cearth + alt) * np.cos(geod_lat) * np.sin(long)
        k_hat = (Searth + alt) * np.sin(geod_lat)

        return np.array([i_hat, j_hat, k_hat])

    @staticmethod
    def site_location_eci(geod_lat: float, lst: float, h_site: float) -> np.ndarray:
        """Calculate ECI vector of the site
            Adapted from AERO 557 W1L1 Notes

        Args:
            geod_lat (float): geodetic latitude [deg]
            lst (float): local sidereal time [deg]
            h_site (float): elevation [km]

        Returns:
            np.ndarray: ECI vector of the site
        """

        geod_lat = np.deg2rad(geod_lat)
        lst = np.deg2rad(lst)

        Re = 6378 * 1_000
        Rp = 6357 * 1_000
        f = (Re - Rp) / Re

        ij_den = np.sqrt(1 - (2 * f - f**2) * np.sin(geod_lat) * np.sin(geod_lat))
        ij_coeff = (Re / ij_den + h_site) * np.cos(geod_lat)
        i_hat = ij_coeff * np.cos(lst)
        j_hat = ij_coeff * np.sin(lst)

        k_coeff = Re * (1 - f) ** 2 / ij_den + h_site
        k_hat = k_coeff * np.sin(geod_lat)

        return np.array([i_hat, j_hat, k_hat]) / 1_000


@dataclass
class Observation:
    time: datetime
    ra: float
    dec: float
    site: Site

    def __post_init__(self) -> None:
        self.ra = np.deg2rad(self.ra)
        self.dec = np.deg2rad(self.dec)
        self.jd = juliandate(self.time)

        # update site with specific observation time
        self.site = Site(self.site.lat, self.site.long, self.site.alt, self.time)

        self.rho_hat = np.array(
            [
                np.cos(self.dec) * np.cos(self.ra),
                np.cos(self.dec) * np.sin(self.ra),
                np.sin(self.dec),
            ]
        )


def ra_dec_to_vec(ra: float, dec: float) -> np.ndarray:
    """
    Derive pointing vector from ra, dec

    Args:
        ra (float): right ascension [deg]
        dec (float): declination [deg]

    Returns:
        np.ndarray: pointing direction
    """

    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    a = np.cos(ra) * np.cos(dec)
    b = np.sin(ra) * np.cos(dec)
    c = np.sin(dec)

    return np.array([a, b, c])


def az_el_to_ra_dec(
    azimuth: float, elevation: float, local_sidereal_time: float, latitude: float
) -> np.ndarray:
    """
    Calculate Topocentric Right Ascension/Declination from Site azimuth, elevation,
        local sidereal time, latitude
    Adapted from "Fundamentals of Astrodynamics", D. Vallado, Algorithm 28

    Args:
        azimuth (float): azimuth look angle from site [deg]
        elevation (float): elevation look angle from site [deg]
        local_sidereal_time (float): local sidereal time of site [deg]
        latitude (float): latitude of site [deg]

    Returns:
        np.ndarray: right ascension, declination [deg]
    """

    # convert to radians
    beta = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    lat = np.deg2rad(latitude)
    lst = np.deg2rad(local_sidereal_time)

    sin_dec = np.sin(el) * np.sin(lat) + np.cos(el) * np.cos(lat) * np.cos(
        beta
    )  # np.sin(declination)
    dec = np.arcsin(sin_dec)  # declination

    sin_lha = -np.sin(beta) * np.cos(el) / np.cos(dec)
    cos_lha = np.cos(lat) * np.sin(el) - np.sin(lat) * np.cos(beta) * np.cos(el)

    lha = np.arctan2(sin_lha, cos_lha)
    ra = lst - lha  # right ascension
    ra = np.mod(ra, 2 * np.pi)

    return np.array([np.rad2deg(ra), np.rad2deg(dec)])  # convert to deg and return


def razel(site: Site, sat_pos: np.ndarray) -> tuple:
    """
    convert site and topocentric information into az/el values
    adapted from dr. kira abercromby, aero 557 "r2elaz" notes

    args:
        site (site): site object containing lst, lat, lon values
        rho (np.ndarray): topocentric directional range to satellite

    returns:
        rho: range [km]
        beta: azimuth angle [deg]
        el: elevation angle [deg]
    """

    az, el, r = eci2aer(
        sat_pos[0] * 1000,
        sat_pos[1] * 1000,
        sat_pos[2] * 1000,
        site.lat,
        site.long,
        site.alt,
        site.date,
    )

    return r / 1000, az, el


def site_track_ecef(
    site: Site, rho: float, azimuth: float, elevation: float
) -> np.ndarray:
    """
    Generate the R_ECEF for satellite using site information

    Args:
        site (Site): observation site
        rho (float): slant range
        azimuth (float): azimuth [deg]
        elevation (float); elevation [deg]

    Returns:
        R (np.ndarray): Sat Position
    """
    el = np.deg2rad(elevation)
    beta = np.deg2rad(azimuth)

    rho_sez = np.array(
        [
            -rho * np.cos(el) * np.cos(beta),
            rho * np.cos(el) * np.sin(beta),
            rho * np.sin(el),
        ]
    )

    rho_ecef = (
        rot_z(np.deg2rad(-site.long))
        @ rot_y(-(np.pi / 2 - np.deg2rad(site.lat)))
        @ rho_sez
    )
    r_ecef = rho_ecef + site.ecef_pos

    return r_ecef


"""
Observation Methods
"""


def gibbs(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> np.ndarray:
    """
    Gibbs method to calculate velocity vector from 3 position vectors
    Adapted from Algorithm 54 from "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        r1 (np.ndarray): Position (ECI) at time 1
        r2 (np.ndarray): Position (ECI) at time 2
        r3 (np.ndarray): Position (ECI) at time 3

    Returns:
        np.ndarray: Velocity Vector at time 2
    """

    z12 = np.cross(r1, r2)
    z23 = np.cross(r2, r3)
    z31 = np.cross(r3, r1)

    # coplanar checks
    alpha_cop = np.arcsin(z23.dot(r1) / (np.linalg.norm(z23) * np.linalg.norm(r1)))

    if alpha_cop > np.deg2rad(3):
        alpha_12 = np.arccos(r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
        alpha_23 = np.arccos(r2.dot(r3) / (np.linalg.norm(r2) * np.linalg.norm(r3)))
        raise ValueError(
            "Vectors are not coplanar! 12: {}; 23: {}".format(
                np.round(alpha_12, 3), np.round(alpha_23, 3)
            )
        )

    N = np.linalg.norm(r1) * z23 + np.linalg.norm(r2) * z31 + np.linalg.norm(r3) * z12
    D: np.NDArray[float] = z12 + z23 + z31
    S = (
        (np.linalg.norm(r2) - np.linalg.norm(r3)) * r1
        + (np.linalg.norm(r3) - np.linalg.norm(r1)) * r2
        + (np.linalg.norm(r1) - np.linalg.norm(r2)) * r3
    )

    B = np.cross(D, r2)
    l_g = np.sqrt(ast.EARTH_MU / (np.linalg.norm(N) * np.linalg.norm(D)))
    v_2 = l_g / np.linalg.norm(r2) * B + l_g * S

    return v_2


def herrick_gibbs(
    r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, jd1: float, jd2: float, jd3: float
) -> np.ndarray:
    """
    Herrick-Gibbs method to calculate velocity vector from 3 position vectors and time
        of observation
    Adapted from Algorithm 55 "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        r1 (np.ndarray): Position (ECI) at time 1
        r2 (np.ndarray): Position (ECI) at time 2
        r3 (np.ndarray): Position (ECI) at time 3
        jd1 (float): JD of observation 1
        jd2 (float): JD of observation 2
        jd3 (float): JD of observation 3

    Returns:
        np.ndarray: Velocity Vector at time 2
    """

    t31 = jd3 - jd1
    t32 = jd3 - jd2
    t21 = jd2 - jd1

    v2_A = -t32 * (1 / (t21 * t31) + ast.EARTH_MU / (12 * np.linalg.norm(r1) ** 3))
    v2_B = (t32 - t21) * (
        1 / (t21 * t32) + ast.EARTH_MU / (12 * np.linalg.norm(r2) ** 3)
    )
    v2_C = t21 * (1 / (t32 * t31) + ast.EARTH_MU / (12 * np.linalg.norm(r3) ** 3))

    return v2_A * r1 + v2_B * r2 + v2_C * r3


def find_c2_c3(psi: float) -> tuple:
    """
    Find C2, C3 from psi for Universal Variable
    Adapted from Algorithm 1 "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        psi (float): psi

    Returns:
        c2 (float): c2
        c3 (float): c3
    """

    if psi > 1e-6:
        c2 = (1 - np.cos(np.sqrt(psi))) / psi
        c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / np.sqrt(psi**3)

        return c2, c3

    if psi < -1e-6:
        c2 = (1 - np.cosh(np.sqrt(-psi))) / psi
        c3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / np.sqrt((-psi) ** 3)

        return c2, c3

    return 1 / 2, 1 / 6


"""
INITIAL ORBIT DETERMINATION METHODS
"""


def gauss_iod(
    observation_1: Observation,
    observation_2: Observation,
    observation_3: Observation,
    *,
    extended: bool = True,
    max_it: int = 100_000,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Gauss Method of Initial Orbit Determination
    Adapted from Algorithm 52 from "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        observation_1 (Observation): Observation 1; contains site, angles, time info
        observation_2 (Observation): Observation 2; contains site, angles, time info
        observation_3 (Observation): Observation 3; contains site, angles, time info
        extended (bool, optional): Flag to use extended Gauss IOD Method
            (more computation but much more accurate). Defaults to True.

    Returns:
        np.ndarray: Position and Velocity Vector of observed object at time 2
    """

    # assert Observations passed in
    for obs in [observation_1, observation_2, observation_3]:
        if type(obs) is not Observation:
            raise TypeError("Need to pass in Observation objects")

    def __cramer_inverse(L: np.ndarray) -> np.ndarray:
        """Alternate method in computing inverse of matrix; Cramer's Rule

        Args:
            L (np.ndarray): matrix to invert

        Returns:
            np.ndarray: inverted matrix
        """

        Lx1 = L[0][0]
        Lx2 = L[1][0]
        Lx3 = L[2][0]

        Ly1 = L[0][1]
        Ly2 = L[1][1]
        Ly3 = L[2][1]

        Lz1 = L[0][2]
        Lz2 = L[1][2]
        Lz3 = L[2][2]

        X = np.array(
            [
                [Ly2 * Lz3 - Ly3 * Lz2, -Ly1 * Lz3 + Ly3 * Lz1, Ly1 * Lz2 - Ly2 * Lz1],
                [-Lx2 * Lz3 + Lx3 * Lz2, Lx1 * Lz3 - Lx3 * Lz1, -Lx1 * Lz2 + Lx2 * Lz1],
                [Lx2 * Ly3 - Lx3 * Ly2, -Lx1 * Ly3 + Lx3 * Ly1, Lx1 * Ly2 - Lx2 * Ly1],
            ]
        )

        return X / np.linalg.det(L)

    def __mod_uni_var(
        R_2: np.ndarray, V_2: np.ndarray, delT: float, *, max_it: int = 10_000
    ) -> tuple:
        """
        Modified Universal Variable to return LaGrange Coefficients
        Adapted from Alg. 8 "Fundamentals of Astrodynamics and Applications", Vallado

        Args:
            R_2 (np.ndarray): Position at time 2
            V_2 (np.ndarray): Velocity at time 2
            delT (float): time to propagate

        Returns:
            f: F LaGrange Coefficient
            g: G LaGrange Coefficient
        """

        r = np.linalg.norm(R_2)
        v = np.linalg.norm(V_2)

        # calculate initial conditions
        alpha = -(v**2) / ast.EARTH_MU + 2 / r

        x_0 = 0
        # elliptical/circular case
        if alpha >= 1e-6:
            x_0 = np.sqrt(ast.EARTH_MU) * delT * alpha

        # parabolic case
        elif np.abs(alpha) < 1e-6:
            hB = np.cross(R_2, V_2)
            p = hB.dot(hB) / ast.EARTH_MU

            cot2s = 3 * np.sqrt(ast.EARTH_MU / p**3) * delT
            s = np.arctan(1 / cot2s) / 2

            tan3w = np.tan(s)
            w = np.arctan(tan3w ** (1 / 3))

            x_0 = np.sqrt(p) * 2 / np.tan(2 * w)

        # hyperbolic case
        elif alpha < -1e-6:
            a = 1 / alpha

            num = -2 * ast.EARTH_MU * alpha * delT
            den = R_2.dot(V_2) + np.sign(delT) * np.sqrt(-ast.EARTH_MU * a) * (
                1 - r * alpha
            )

            x_0 = np.sign(delT) * np.sqrt(-a) * np.log(num / den)

        diff = 1
        count = 0
        c2 = 0
        c3 = 0
        x_r = 0
        while np.abs(diff) > 1e-6 and count < max_it:
            psi = x_0**2 * alpha
            c2, c3 = find_c2_c3(psi)

            x_r = (
                x_0**2 * c2
                + R_2.dot(V_2) / np.sqrt(ast.EARTH_MU) * x_0 * (1 - psi * c3)
                + r * (1 - psi * c2)
            )

            num = (
                np.sqrt(ast.EARTH_MU) * delT
                - x_0**3 * c3
                - (R_2.dot(V_2)) / np.sqrt(ast.EARTH_MU) * x_0**2 * c2
                - r * x_0 * (1 - psi * c3)
            )
            x_1 = x_0 + num / x_r

            diff = x_0 - x_1
            count += 1

            x_0 = x_1

        f = 1 - x_0**2 * c2 / r
        g = delT - x_0**3 * c3 / np.sqrt(ast.EARTH_MU)
        fDot = np.sqrt(ast.EARTH_MU) * x_0 * (psi * c3 - 1) / (x_r * r)
        gDot = 1 - x_0**2 * c2 / x_r

        if np.abs(f * gDot - fDot * g) > 1 + 1e-6:
            raise ValueError("bad lagrange: {}".format(f * gDot - fDot * g))

        return f, g

    # rename arguments
    oA = observation_1
    oB = observation_2
    oC = observation_3

    # calculate timing parameters
    tau1 = (oA.jd - oB.jd) * 86_400
    tau3 = (oC.jd - oB.jd) * 86_400

    a1 = tau3 / (tau3 - tau1)
    a1u = tau3 * ((tau3 - tau1) ** 2 - tau3**2) / (6 * (tau3 - tau1))

    a3 = -tau1 / (tau3 - tau1)
    a3u = -tau1 * ((tau3 - tau1) ** 2 - tau1**2) / (6 * (tau3 - tau1))

    # calculate angles only matrix
    L = np.column_stack([oA.rho_hat, oB.rho_hat, oC.rho_hat])
    L_inv = __cramer_inverse(L)

    # calculate site position matrix
    R = np.column_stack([oA.site.eci_pos, oB.site.eci_pos, oC.site.eci_pos])
    M = L_inv @ R

    # calculate d params
    d1 = M[1][0] * a1 - M[1][1] + M[1][2] * a3
    d2 = M[1][0] * a1u + M[1][2] * a3u

    C = oB.rho_hat.dot(oB.site.eci_pos)
    rs2 = np.linalg.norm(oB.site.eci_pos)

    # set coefficients of 8th order polynomial
    poly_coeff = [
        1,
        0,
        -(d1**2 + 2 * C * d1 + rs2**2),
        0,
        0,
        -2 * ast.EARTH_MU * (C * d2 + d1 * d2),
        0,
        0,
        -(ast.EARTH_MU**2 * d2**2),
    ]

    # root solve and determine real (positive) root
    r2 = []
    r2_candidate = np.roots(poly_coeff)

    for r in r2_candidate:
        if np.imag(r) != 0:  # discard roots with imaginary parts
            continue
        r = np.real(r)
        if r < 0:  # discard roots with negative real parts
            continue

        r2.append(r)  # append to solution list

    if len(r2) > 1:
        raise IndexError("Too many real solutions found")

    r2 = r2[0]

    # calculate c parameters
    u = ast.EARTH_MU / (r2**3)
    c1 = a1 + a1u * u
    c2 = -1
    c3 = a3 + a3u * u

    c_rho = M @ np.array([-c1, -c2, -c3])

    # solve for slant ranges
    rho_1 = c_rho[0] / c1
    rho_2 = c_rho[1] / c2
    rho_3 = c_rho[2] / c3

    # solve for position vectors across all three observations
    r1 = rho_1 * oA.rho_hat + oA.site.eci_pos
    r2 = rho_2 * oB.rho_hat + oB.site.eci_pos
    r3 = rho_3 * oC.rho_hat + oC.site.eci_pos

    # use gibbs method to solve for velocity
    delTheta = np.zeros(2)
    for i, pos in enumerate([r1, r3]):
        delTheta[i] = np.arccos(
            pos.dot(r2) / (np.linalg.norm(pos) * np.linalg.norm(r2))
        )

    # if small separation use herrick gibbs
    if np.sum(delTheta) <= np.deg2rad(3):
        v2 = herrick_gibbs(r1, r2, r3, oA.jd, oB.jd, oC.jd)

    else:
        v2 = gibbs(r1, r2, r3)

    if extended:
        count = 0
        diff1 = 1
        diff2 = 1
        diff3 = 1

        oldRho1 = rho_1
        oldRho2 = rho_2
        oldRho3 = rho_3

        f = np.zeros(2)
        g = np.zeros(2)

        # calculate lagrange coefficients
        f[0], g[0] = __mod_uni_var(r2, v2, tau1)
        f[1], g[1] = __mod_uni_var(r2, v2, tau3)

        while count < max_it and diff1 > tol and diff2 > tol and diff3 > tol:
            ff1, gg1 = __mod_uni_var(r2, v2, tau1)
            ff3, gg3 = __mod_uni_var(r2, v2, tau3)

            # average the LaGrange Coefficients
            f[0] = (f[0] + ff1) / 2
            f[1] = (f[1] + ff3) / 2
            g[0] = (g[0] + gg1) / 2
            g[1] = (g[1] + gg3) / 2

            # recalculate c parameters
            c1 = g[1] / (f[0] * g[1] - f[1] * g[0])
            c2 = -1
            c3 = -g[0] / (f[0] * g[1] - f[1] * g[0])

            c_rho = M @ np.array([-c1, -c2, -c3])

            # solve for slant ranges
            rho_1 = c_rho[0] / c1
            rho_2 = c_rho[1] / c2
            rho_3 = c_rho[2] / c3

            # solve for position vectors across all three observations
            r1 = rho_1 * oA.rho_hat + oA.site.eci_pos
            r2 = rho_2 * oB.rho_hat + oB.site.eci_pos
            r3 = rho_3 * oC.rho_hat + oC.site.eci_pos

            # use gibbs method to solve for velocity
            for i, pos in enumerate([r1, r3]):
                delTheta[i] = np.arccos(
                    pos.dot(r2) / (np.linalg.norm(pos) * np.linalg.norm(r2))
                )

            # if small separation use herrick gibbs
            if np.sum(delTheta) <= np.deg2rad(3):
                v2 = herrick_gibbs(r1, r2, r3, oA.jd, oB.jd, oC.jd)

            else:
                v2 = gibbs(r1, r2, r3)

            # check slant range diff
            diff1 = np.abs(rho_1 - oldRho1)
            diff2 = np.abs(rho_2 - oldRho2)
            diff3 = np.abs(rho_3 - oldRho3)

            oldRho1 = rho_1
            oldRho2 = rho_2
            oldRho3 = rho_3

            count += 1

    return np.append(r2, v2)


def double_r_iod(
    observation_1: Observation,
    observation_2: Observation,
    observation_3: Observation,
    *,
    max_it: int = 1_000,
    traj_type: int = 1,
) -> np.ndarray:
    """
    Double-R Iteration for initial orbit determination
    Adapted from Alg. 53 "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        observation_1 (Observation): Observation at time 1
        observation_2 (Observation): Observation at time 2
        observation_3 (Observation): Observation at time 3
        max_it (int, optional): Max number of iterations. Defaults to 1_000.
        type (int, optional): Trajectory type;
            1 => prograde, -1 => retrograde. Defaults to 1.

    Raises:
        TypeError: if arguments are not Observations

    Returns:
        np.ndarray: State vector of orbit at time 2
    """

    # assert Observations passed in
    for obs in [observation_1, observation_2, observation_3]:
        if type(obs) is not Observation:
            raise TypeError("Need to pass in Observation objects")

    # check trajectory type
    if traj_type not in [1, -1]:
        raise ValueError("Invalid Trajectory Type: +1 for prograde -1 for retrograde")

    def __dr_iteration(
        c: np.ndarray,
        r: np.ndarray,
        tau1: float,
        tau3: float,
        oA: Observation,
        oB: Observation,
        oC: Observation,
        tm: int = 1,
    ) -> tuple:
        """
        Iterative portion of the Double R IOD Method
        Adapted from:
            Algorithm 53 from "Fundamentals of Astrodynamics and Applications", Vallado
            "AERO 557doubleR.m", Kira Abercromby; Cal Poly SLO AERO 557

        Args:
            c (np.ndarray): c values (C1, C3)
            R (np.ndarray): norm of positions (3rd entry usually 0)
            tau1 (float): obs1_time - obs2_time [sec]
            tau3 (float): obs3_time - obs2_time [sec]
            oA (Observation): observation 1
            oB (Observation): observation 2
            oC (Observation): observation 3

        Returns:
            F1 (float): value for diff eq
            F2 (float): value for diff eq
            f (float): lagrange coefficient
            g (float); lagrange coefficient
            rv_2 (np.ndarray): position at time 2
            rv_3 (np.ndarray): position at time 3
        """
        # print(r)

        rho = np.zeros(3)
        r_v = np.zeros((3, 3))
        e_cos_v = np.zeros(3)

        def cos_v_jk(j: np.ndarray, k: np.ndarray) -> float:
            """
            Custom cos function for iteration
            """
            return j.dot(k) / (np.linalg.norm(j) * np.linalg.norm(k))

        def sin_v_jk(j: np.ndarray, k: np.ndarray) -> float:
            """
            Custom sun function for iteration
            """
            return tm * np.sqrt(1 - (cos_v_jk(j, k) ** 2))

        for i, obs in enumerate([oA, oB]):
            rho[i] = 0.5 * (
                -c[i]
                + np.sqrt(
                    c[i] ** 2 - 4 * (np.linalg.norm(obs.site.eci_pos) ** 2 - r[i] ** 2)
                )
            )
            r_v[i] = rho[i] * obs.rho_hat + obs.site.eci_pos
            r[i] = np.linalg.norm(r_v[i])

        W = np.cross(r_v[0], r_v[1]) / (r[0] * r[1])

        rho[2] = np.dot(-oC.site.eci_pos, W) / (oC.rho_hat.dot(W))
        r_v[2] = rho[2] * oC.rho_hat + oC.site.eci_pos
        r[2] = np.linalg.norm(r_v[2])

        c_del_v21 = cos_v_jk(r_v[1], r_v[0])
        c_del_v32 = cos_v_jk(r_v[2], r_v[1])
        c_del_v31 = cos_v_jk(r_v[2], r_v[0])

        s_del_v21 = sin_v_jk(r_v[1], r_v[0])
        s_del_v32 = sin_v_jk(r_v[2], r_v[1])
        s_del_v31 = sin_v_jk(r_v[2], r_v[0])

        if np.arccos(c_del_v31) > np.pi:
            c1 = (r[1] * s_del_v32) / (r[0] * s_del_v31)
            c3 = (r[1] * s_del_v21) / (r[2] * s_del_v31)
            p = (c1 * r[0] + c3 * r[2] - r[1]) / (c1 + c3 - 1)

        elif np.arccos(c_del_v31) < np.pi:
            c1 = (r[0] * s_del_v31) / (r[1] * s_del_v32)
            c3 = (r[0] * s_del_v21) / (r[2] * s_del_v32)
            p = (c3 * r[2] - c1 * r[1] + r[0]) / (-c1 + c3 + 1)

        else:
            raise ValueError("cos(v31) is Invalid!: {}".format(c_del_v31))

        for i, arr in enumerate(r):
            e_cos_v[i] = p / arr - 1

        if np.arccos(c_del_v21) != np.pi:
            e_sin_v2 = (-c_del_v21 * e_cos_v[1] + e_cos_v[0]) / s_del_v21
        else:
            e_sin_v2 = (c_del_v32 * e_cos_v[1] - e_cos_v[2]) / s_del_v31

        e2 = e_cos_v[1] ** 2 + e_sin_v2**2
        e = np.sqrt(e2)
        a = p / (1 - e**2)

        # elliptical case
        if np.sqrt(e2) < 1:  # or np.sqrt(e2) >= 1:
            n = np.sqrt(ast.EARTH_MU / (a**3))

            S = r[1] * np.sqrt(1 - e2) * e_sin_v2 / p
            C = r[1] * (e2 + e_cos_v[1]) / p

            s_del_e32 = (
                r[2] * s_del_v32 / np.sqrt(a * p) - r[2] * (1 - c_del_v32) * S / p
            )
            c_del_e32 = 1 - (r[1] * r[2]) * (1 - c_del_v32) / (a * p)
            s_del_e21 = (
                r[0] * s_del_v21 / np.sqrt(a * p) + r[0] * (1 - c_del_v21) * S / p
            )
            c_del_e21 = 1 - (r[1] * r[0]) * (1 - c_del_v21) / (a * p)

            e32 = np.arccos(c_del_e32)
            e21 = np.arccos(c_del_e21)

            del_m32 = e32 + 2 * S * (np.sin(e32 / 2) ** 2) - C * s_del_e32
            del_m12 = -e21 + 2 * S * (np.sin(e21 / 2) ** 2) + C * s_del_e21

            F1 = tau1 - del_m12 / n
            F2 = tau3 - del_m32 / n

            f = 1 - a / r[1] * (1 - c_del_e32)
            g = tau3 - np.sqrt(a**3 / ast.EARTH_MU) * (e32 - s_del_e32)

        # hyperbolic case
        else:
            n = np.sqrt(ast.EARTH_MU / (-(a**3)))
            Sh = r[1] * np.sqrt(e2 - 1) * e_sin_v2 / p
            Ch = r[1] * (e2 + e_cos_v[1]) / p
            sinhF32 = (
                r[2] * s_del_v32 / np.sqrt(-a * p) - r[2] * (1 - c_del_v32) * Sh / p
            )
            F32 = np.log(sinhF32 + np.sqrt(sinhF32 + 1))

            sinhF21 = (
                r[0] * s_del_v21 / np.sqrt(-a * p) + r[0] * (1 - c_del_v32) * Sh / p
            )
            F21 = np.log(sinhF21 + np.sqrt(sinhF21**2 + 1))

            M32 = -F32 + 2 * Sh * np.sinh(F32 / 2) ** 2 + Ch * sinhF32
            M12 = F21 + 2 * Sh * np.sinh(F21 / 2) ** 2 + Ch * sinhF21
            F1 = tau1 - M12 / n
            F2 = tau3 - M32 / n
            f = 1 - (-a) / r[1] * (1 - np.cosh(F32))
            g = tau3 - np.sqrt((-a) ** 3 / ast.EARTH_MU) * (F32 - sinhF32)

        return F1, F2, f, g, r_v[2], r_v[1]

    oA = observation_1
    oB = observation_2
    oC = observation_3

    # tau values [sec]
    tau1 = (oA.jd - oB.jd) * 86_400
    tau3 = (oC.jd - oB.jd) * 86_400

    # set C values
    c = np.zeros(2)
    for i, obs in enumerate([oA, oB]):
        c[i] = np.dot(2 * obs.rho_hat, obs.site.eci_pos)

    # set up iteration parameters
    dr_count = 0
    r = np.array([2 * ast.EARTH_RAD, 2.01 * ast.EARTH_RAD, 0])  # initial guess of orbit
    err = 1
    g = 0
    while err > 0.0001 and dr_count < max_it:
        # run main iteration
        F1, F2, f, g, r_v3, r_v2 = __dr_iteration(c, r, tau1, tau3, oA, oB, oC)

        Q = np.linalg.norm([F1, F2])
        ddr1 = 0.005 * r[0]
        ddr2 = 0.005 * r[1]

        dr1 = np.array([r[0] + ddr1, r[1], 0])
        dr2 = np.array([r[0], r[1] + ddr2, 0])

        # run modified iterations
        F1dr1, F2dr1, _, _, _, _ = __dr_iteration(c, dr1, tau1, tau3, oA, oB, oC)
        F1dr2, F2dr2, _, _, _, _ = __dr_iteration(c, dr2, tau1, tau3, oA, oB, oC)

        # jacobian of F parameters
        delF1dr1 = (F1dr1 - F1) / ddr1
        delF2dr1 = (F2dr1 - F2) / ddr1
        delF1dr2 = (F1dr2 - F1) / ddr2
        delF2dr2 = (F2dr2 - F2) / ddr2

        # update orbit based on jacobian
        delta = delF1dr1 * delF2dr2 - delF2dr1 * delF1dr2
        delta_1 = delF2dr2 * F1 - delF1dr2 * F2
        delta_2 = delF1dr1 * F2 - delF2dr1 * F1
        delta_r1 = -delta_1 / delta
        delta_r2 = -delta_2 / delta

        err = (np.abs(delta_r1) + np.abs(delta_r2)) / 2

        r[0] += delta_r1
        r[1] += delta_r2

        dr_count += 1

    # find velocity at time 2
    v_2 = (r_v3 - f * r_v2) / g

    return np.append(r_v2, v_2)


"""
LAMBERT/PROPAGATION SOLVERS
"""


def universal_variable_propagation_vallado(
    R_2: np.ndarray, V_2: np.ndarray, delT: float, *, max_it: int = 10_000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lambert Solver to solve for state at some t = t_0 + delta_t
    Adapted from Algorithm 8 "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        R_2 (np.ndarray): Position at time 2
        V_2 (np.ndarray): Velocity at time 2
        delT (float): time to propagate

    Returns:
        R: Position at time t = 0 + delta_t
        V: Velocity at time t = 0 + delta_t
    """

    r = np.linalg.norm(R_2)
    v = np.linalg.norm(V_2)

    # calculate initial conditions
    alpha = -(v**2) / ast.EARTH_MU + 2 / r

    # elliptical/circular case
    if alpha > 1e-6:
        x_0 = np.sqrt(ast.EARTH_MU) * delT * alpha

    # parabolic case
    elif np.abs(alpha) < 1e-6:
        hB = np.cross(R_2, V_2)
        p = hB.dot(hB) / ast.EARTH_MU

        cot2s = 3 * np.sqrt(ast.EARTH_MU / p**3) * delT
        s = np.arctan(1 / cot2s) / 2

        tan3w = np.tan(s)
        w = np.arctan(tan3w ** (1 / 3))

        x_0 = np.sqrt(p) * 2 / np.tan(2 * w)

    # hyperbolic case
    elif alpha < -1e-6:
        a = 1 / alpha

        num = -2 * ast.EARTH_MU * alpha * delT
        den = R_2.dot(V_2) + np.sign(delT) * np.sqrt(-ast.EARTH_MU * a) * (
            1 - r * alpha
        )

        x_0 = np.sign(delT) * np.sqrt(-a) * np.log(num / den)

    else:
        raise ValueError("alpha: {}".format(alpha))

    # begin iteration to calc parameters
    diff = 1
    count = 0
    c2 = 0
    c3 = 0
    x_r = 0
    psi = 0
    while np.abs(diff) > 1e-6 and count < max_it:
        psi = x_0**2 * alpha
        c2, c3 = find_c2_c3(psi)

        x_r = (
            x_0**2 * c2
            + R_2.dot(V_2) / np.sqrt(ast.EARTH_MU) * x_0 * (1 - psi * c3)
            + r * (1 - psi * c2)
        )

        num = (
            np.sqrt(ast.EARTH_MU) * delT
            - x_0**3 * c3
            - (R_2.dot(V_2)) / np.sqrt(ast.EARTH_MU) * x_0**2 * c2
            - r * x_0 * (1 - psi * c3)
        )
        x_1 = x_0 + num / x_r

        diff = x_0 - x_1
        count += 1

        x_0 = x_1

    f = 1 - x_0**2 * c2 / r
    g = delT - x_0**3 * c3 / np.sqrt(ast.EARTH_MU)
    fDot = np.sqrt(ast.EARTH_MU) * x_0 * (psi * c3 - 1) / (x_r * r)
    gDot = 1 - x_0**2 * c2 / x_r

    if np.abs(f * gDot - fDot * g) > 1 + 1e-6:
        raise ValueError("bad lagrange: {}".format(f * gDot - fDot * g))

    R = f * R_2 + g * V_2
    V = fDot * R_2 + gDot * V_2

    return R, V


def izzo_gooding_lambert_solver(
    r_v1: np.ndarray,
    r_v2: np.ndarray,
    t_flight: float,
    max_revs: int = 0,
    *,
    mu: int = ast.EARTH_MU,
    tol: float = 1e-12,
    max_it: int = 10_000,
    retrograde_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Izzo-Gooding Solution to Lambert's Problem
    Adapted from "Revisiting Lambert's Problem", Izzo (2012)
    Adapted from lambert_problem.cpp, ESA GitHub: https://github.com/esa/pykep/blob/master/src/lambert_problem.cpp

    Args:
        r_v1 (np.ndarray): position at time 1
        r_v2 (np.ndarray): position at time 2
        t_flight (float): time of flight [sec]
        max_revs (int): max number of revolutions to solve for
        mu (int, optional): Gravitational Parameter of central body. Defaults to EARTH_MU.

    Returns:
        v1 (np.ndarray): velocity at time 1
        v2 (np.ndarray): velocity at time 2
    """

    def __hyperF(z: float, tol: float) -> float:
        """
        Construct F from hypergeometric parameters

        Args:
            z (float): z-value
            tol (float): tolerance

        Returns:
            float: F
        """
        Sj = 1
        Cj = 1
        err = 1
        Cj1 = 0
        Sj1 = 0
        j = 0

        while err > tol:
            Cj1 = Cj * (3 + j) * (1 + j) / (2.5 + j) * z / (j + 1)
            Sj1 = Sj + Cj1

            err = np.abs(Cj1)
            Sj = Sj1
            Cj = Cj1
            j += 1

        return Sj

    def __x2tof2(x: float, N: int, lam: float) -> float:
        """
        LaGrange expression for time of flight

        Args:
            x (float): x
            N (int): num revs
            lam (float): lambda value

        Returns:
            float: time of flight
        """

        a = 1 / (1 - x**2)

        if a > 0:  # elliptical case
            alpha = 2 * np.arccos(x)
            beta = 2 * np.arccos(np.sqrt(lam**2 / a))

            if lam < 0:
                beta = -beta

            tof = (
                a
                * np.sqrt(a)
                * ((alpha - np.sin(alpha)) - (beta - np.sin(beta)) + 2 * np.pi * N)
            ) / 2

        else:  # hyperbolic case
            alpha = 2 * np.arccosh(x)
            beta = 2 * np.arcsinh(np.sqrt(-lam * lam / a))

            if lam < 0:
                beta = -beta

            tof = (
                -a
                * np.sqrt(-a)
                * ((beta - np.sinh(beta)) - (alpha - np.sinh(alpha)))
                / 2
            )

        return tof

    def __x2tof(x: float, N: int, lam: float) -> float:
        """
        get time of flight from x

        Args:
            x (float): x
            N (int): num revs
            lam (float): lambda value

        Returns:
            float: time of flight
        """
        battin = 0.01
        lagrange = 0.2
        dist = np.abs(x - 1)

        if dist < lagrange and dist > battin:
            return __x2tof2(x, N, lam)

        K = lam**2
        E = x**2 - 1
        rho = np.abs(E)
        z = np.sqrt(1 + K * E)

        if dist < battin:
            eta = z - lam * x
            S1 = 0.5 * (1 - lam - x * eta)
            Q = __hyperF(S1, 1e-11)

            Q = 4 / 3 * Q

            tof = (eta**3 * Q + 4 * lam * eta) / 2 + N * np.pi / rho**1.5

            return tof

        y = np.sqrt(rho)
        g = x * z - lam * E
        d = 0

        if E < 0:
            l = np.arccos(g)
            d = N * np.pi + l

        else:
            f = y * (z - lam * x)
            d = np.log(f + g)

        tof = (x - lam * x - d / y) / E

        return tof

    def __dTdx(x: float, T: float, lam: float) -> tuple:
        """
        Time derivative wrt x function

        Args:
            x (float): x to solve T at
            T (float): Time
            lam (float): lambda value

        Returns:
            tuple: first, second, third derivative of T
        """
        umx2 = 1 - x**2
        y = np.sqrt(1 - lam**2 * umx2)

        DT = 1 / umx2 * (3 * T * x - 2 + 2 * lam**3 * x / y)
        DDT = 1 / umx2 * (3 * T + 5 * x * DT + 2 * (1 - lam**2) * lam**3 / y**3)
        DDDT = (
            1
            / umx2
            * (7 * x * DDT + 8 * DT - 6 * (1 - lam**2) * lam**5 * x / (y**5))
        )

        return DT, DDT, DDDT

    def __householder_iteration(
        T: float, x_0: float, N: int, eps: float, lam: float, *, max_it: int = 100
    ) -> float:
        """
        Householder iteration to solve for iterations required per revolution case

        Args:
            T (float): Time
            x_0 (float): Initial guess
            N (int): number of revs
            eps (float): tolerance
            lam (float): lambda value
            max_it (int, optional): max number of iterations. Defaults to 100.

        Returns:
            float: number of iterations
        """
        # set up variables for iteration
        it = 0
        err = 1
        xnew = 0

        tof = 0
        delta = 0
        DT = 0
        DDT = 0
        DDDT = 0

        while err > eps and it < max_it:
            tof = __x2tof(x_0, N, lam)
            DT, DDT, DDDT = __dTdx(x_0, tof, lam)

            delta = tof - T
            DT2 = DT**2

            xnew = x_0 - delta * (DT2 - delta * DDT / 2) / (
                DT * (DT2 - delta * DDT) + DDDT * delta**2 / 6
            )
            err = np.abs(xnew - x_0)
            x_0 = xnew

            it += 1

        return it

    # calc lambda and T
    c_v = r_v2 - r_v1

    c = np.linalg.norm(c_v)
    r1 = np.linalg.norm(r_v1)
    r2 = np.linalg.norm(r_v2)
    s = 0.5 * (r1 + r2 + c)

    r1u = r_v1 / r1
    r2u = r_v2 / r2
    hu = np.cross(r1u, r2u)

    lam = np.sqrt(1 - c / s)

    if hu[2] < 0:  # transfer angle > 180deg
        lam *= -1
        t1u = np.cross(r1u, hu)
        t2u = np.cross(r2u, hu)

    else:
        t1u = np.cross(hu, r1u)
        t2u = np.cross(hu, r2u)

    t1u = t1u / np.linalg.norm(t1u)
    t2u = t2u / np.linalg.norm(t2u)

    # check retrograde motion
    if retrograde_flag:
        t1u = t1u * -1
        t2u = t2u * -1

    T = np.sqrt(2 * mu / s**3) * t_flight

    # find all x's
    # first check how many solutions exist (revolutions-wise)
    n_max = int(T / np.pi)
    t00 = np.arccos(lam) + lam * np.sqrt(1 - lam**2)
    t0 = t00 + n_max * np.pi
    t1 = 2 / 3 * (1 - lam**3)

    DT = 0
    DDT = 0
    DDDT = 0

    if n_max > 0:
        if T < t0:  # halley's iteration
            it = 0
            err = 1
            T_min = t0
            x_0 = 0
            x_1 = 0

            while err > tol and it < max_it:
                DT, DDT, DDDT = __dTdx(x_0, T_min, lam)

                if DT != 0:
                    x_1 = x_0 - DT * DDT / (DDT**2 - DT * DDDT / 2)

                err = np.abs(x_1 - x_0)

                T_min = __x2tof(x_1, n_max, lam)
                x_0 = x_1
                it += 1

            if T_min > T:
                n_max -= 1

    # select minimum number of revolutions
    n_max = np.min([n_max, max_revs])

    # allocate space for terminal velocities
    V1 = np.zeros((2 * n_max + 1, 3))
    V2 = np.zeros((2 * n_max + 1, 3))

    its = np.zeros((2 * n_max + 1, 1))
    x = np.zeros((2 * n_max + 1, 1))

    # find all solutions of x,y
    if T >= t00:
        x[0] = -(T - t00) / (T - t00 + 4)

    elif T <= t1:
        x[0] = t1 * (t1 - T) / (2 * (1 - lam**5) * T / 5) + 1

    else:
        x[0] = np.power((T / t00), 0.69314718055994529 / np.log(t1 / t00)) - 1

    its[0] = __householder_iteration(T, x[0], 0, 1e-12, lam)  # no revolution case

    # multi-rev case
    tmp = 0
    for i in range(1, n_max + 1):
        # left leg
        tmp = np.power((i * np.pi + np.pi) / (8 * T), 2 / 3)
        x[2 * i - 1] = (tmp - 1) / (tmp + 1)
        its[2 * i - 1] = __householder_iteration(T, x[2 * i - 1], i, 1e-12, lam)

        # right leg
        tmp = np.power((8 * T) / (i * np.pi), 2 / 3)
        x[2 * i] = (tmp - 1) / (tmp + 1)
        its[2 * i] = __householder_iteration(T, x[2 * i], i, 1e-12, lam)

    gamma = np.sqrt(mu * s / 2)
    rho = (r1 - r2) / c
    sigma = np.sqrt(1 - rho**2)

    # for all x and y solutions, solve for terminal velocities
    x_list = x
    for i, x in enumerate(x_list):
        y = np.sqrt(1 - lam**2 + lam**2 * x**2)
        vr1 = gamma * ((lam * y - x) - rho * (lam * y + x)) / r1
        vr2 = -gamma * ((lam * y - x) + rho * (lam * y + x)) / r2

        vt = gamma * sigma * (y + lam * x)
        vt1 = vt / r1
        vt2 = vt / r2

        V1[i] = vr1 * r1u + vt1 * t1u
        V2[i] = vr2 * r2u + vt2 * t2u

    return V1, V2


def universal_variable_lambert_solver_vallado(
    r_v1: np.ndarray,
    r_v2: np.ndarray,
    deltaT: float,
    *,
    traj_type: int = 1,
    tol: float = 1e-8,
    mu: int = ast.EARTH_MU,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Universal Variable solution to Lambert's Problem
    Adapted from Algorithm 5.2 from "Orbital Mechanics for Engineering Students", Curtis

    Args:
        r1 (np.ndarray): Position at time 1
        r2 (np.ndarray): Position at time 2
        deltaT (float): time difference
        traj_type (int, optional): Short way vs long way around. Defaults to 1.

    Returns:
        V1: Velocity Vector at time 1
        V2: Velocity Vector at time 2
    """

    # validate trajectory type
    if traj_type not in [1, -1]:
        raise ValueError("Invalid Trajectory Type. 1 for Prograde, -1 for Retrograde")

    """
    define functions for universal variable solution
    """

    def __y_z(z: float) -> float:
        """
        Universal Variable function

        Args:
            r1 (float): norm of position vector at time 1
            r2 (float): norm of position vector at time 2
            z (float): Z-Value
            A (float): A_Value

        Returns:
            float: Result
        """

        frac = (z * stumpff_S(z) - 1) / np.sqrt(stumpff_C(z))
        return r1 + r2 + A * frac

    def __F_z(z: float) -> float:
        """
        Numerator to solve for Universal Variable via Newton's Method

        Args:
            r1 (float): norm of position vector at time 1
            r2 (float): norm of position vector at time 2
            z (float): Z-Value
            t (float): Time

        Returns:
            float: Result
        """

        yz = __y_z(z)
        return (
            (yz / stumpff_C(z)) ** 1.5 * stumpff_S(z)
            + A * np.sqrt(yz)
            - np.sqrt(mu) * deltaT
        )

    def __Fp_z(z: float) -> float:
        """
        Derivative of F function to solve for Universal Variable via Newtons Method

        Args:
            r1 (float): norm of position vector at time 1
            r2 (float): norm of position vector at time 2
            z (float): Z-Value
            A (float): A-Value

        Returns:
            float: Result
        """

        if z == 0:
            yz0 = __y_z(0)
            rhs = np.sqrt(yz0) + A * np.sqrt(1 / (2 * yz0))
            fp = np.sqrt(2) * yz0**1.5 / 40 + A / 8 * rhs

        else:
            Sz = stumpff_S(z)
            Cz = stumpff_C(z)
            yz = __y_z(z)

            rhs = 3 * Sz * np.sqrt(yz) / Cz + A * np.sqrt(Cz / yz)
            mhs = 1 / (2 * z) * (Cz - 3 * Sz / (2 * Cz)) + 3 * Sz**2 / (4 * Cz)
            lhs = (yz / Cz) ** 1.5

            fp = lhs * mhs + A / 8 * rhs

        return fp

    r1 = np.sqrt(r_v1.dot(r_v1))
    r2 = np.sqrt(r_v2.dot(r_v2))

    r_cross_z = np.cross(r_v1, r_v2)[2]

    delTheta = np.arccos(r_v1.dot(r_v2) / (r1 * r2))
    if traj_type == 1 and r_cross_z < 0:
        delTheta = 2 * np.pi - delTheta
    if traj_type == -1 and r_cross_z >= 0:
        delTheta = 2 * np.pi - delTheta

    A = np.sin(delTheta) * np.sqrt(r1 * r2 / (1 - np.cos(delTheta)))

    # newton method to find z
    def z_newton(z0: float) -> float:
        """
        Call newton's iteration on F, Fp
        """
        return z0 - __F_z(z0) / __Fp_z(z0)

    # initial values for newtons method
    z_0 = 1.0
    z_1 = z_newton(z_0)
    err = np.abs(z_1 - z_0)

    # run newtons method
    while err > tol:
        z_0 = z_1
        z_1 = z_newton(z_0)
        err = np.abs(z_1 - z_0)

    z = z_1
    yz = __y_z(z)

    # solve for v1, v2 using lagrange coefficients
    f = 1 - yz / r1
    g = A * np.sqrt(yz / mu)
    gDot = 1 - yz / r2

    v_v1 = 1 / g * (r_v2 - f * r_v1)
    v_v2 = 1 / g * (gDot * r_v2 - r_v1)

    return v_v1, v_v2


def gauss_lambert_solver(
    r_v1: np.ndarray,
    r_v2: np.ndarray,
    deltaT: float,
    *,
    mu: int = ast.EARTH_MU,
    traj_type: int = 1,
    max_it: int = 1_000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gauss method to solve Lambert's Problem
    Adapted from Alg 57 "Fundamentals of Astrodynamics and Applications", Vallado

    Args:
        r_v1 (np.ndarray): Position at time 1
        r_v2 (np.ndarray): Posotion at time 2
        deltaT (float): time between time 1 and 2
        traj_type (int, optional): Trajectory Type;
            1 for Prograde -1 for Retrograde. Defaults to 1.
        max_it (int, optional): Max number of iterations. Defaults to 1_000.
        tol (float, optional): Tolerance for error. Defaults to 1e-8.

    Raises:
        ValueError: When trajectory type is invalid

    Returns:
        np.ndarray: velocity vector at time 1 and 2
    """

    # validate trajectory type
    if traj_type not in [1, -1]:
        raise ValueError("Invalid Trajectory Type. 1 for Prograde, -1 for Retrograde")

    # determine norms of position
    r1 = np.sqrt(r_v1.dot(r_v1))
    r2 = np.sqrt(r_v2.dot(r_v2))

    # cosine, sine of inter-vector angles
    c_delv = r_v1.dot(r_v2) / (r1 * r2)
    s_delv = traj_type * np.sqrt(1 - c_delv**2)
    delv = np.arccos(c_delv)

    # setting l and m parameters
    l = (r1 + r2) / (4 * np.sqrt(r1 * r2) * np.cos(delv / 2)) - 0.5
    m = mu * deltaT**2 / ((2 * np.sqrt(r1 * r2) * np.cos(delv / 2)) ** 3)

    # definition of x_1 and x_2 relative to y
    def x1_cal(y: float) -> float:
        return m / y**2 - l

    def x2_cal(x: float) -> float:
        return (
            4
            / 3
            * (
                1
                + 6 * x / 5
                + 6 * 8 * x**2 / (5 * 7)
                + 6 * 8 * 10 * x**3 / (5 * 7 * 9)
            )
        )

    def y1_cal(y: float) -> float:
        return 1 + x2_cal(x1_cal(y)) * (l + x1_cal(y))

    # start iterating to find optimal y with i iterations
    i = 0
    y_0 = 1
    y_1 = y1_cal(y_0)
    err = np.abs(y_1 - y_0)

    # iterate to find optimal y
    while err > tol and i < max_it:
        y_0 = y_1
        y_1 = y1_cal(y_0)
        err = np.abs(y_1 - y_0)

        i += 1

    # set x1, parameters from optimal y
    x1 = x1_cal(y_1)

    # elliptical case: mean eccentric anomaly
    c_del_e2 = 1 - 2 * x1
    p = (
        r1
        * r2
        * (1 - c_delv)
        / (r1 + r2 - 2 * np.sqrt(r1 * r2) * np.cos(delv / 2) * c_del_e2)
    )

    # set lagrange coefficients
    f = 1 - r1 / p * (1 - c_delv)
    g = r1 * r2 * s_delv / np.sqrt(mu * p)
    gDot = 1 - r1 / p * (1 - c_delv)

    # solve for v1, v2
    v1 = (r_v2 - f * r_v1) / g
    v2 = (gDot * r_v2 - r_v1) / g

    return v1, v2


"""
PLANETS
"""


def planetary_ephemeris(body: str, julian_date: float, *, mu: int = ast.SUN_MU) -> COES:
    """
    Calculate R and V of specified planet at julian date
    Adapted from Cal Poly SLO AERO 351: Planetary Elements2 - Dr. Abercromby

    Args:
        body (str): planet to calc state vector for
            [MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE]

        julian_date (float): julian date

    Returns:
        COES: Coes of Planet
    """
    jd = julian_date
    T = (jd - 2_451_545) / 36_525

    body = body.upper()
    match body:
        case "MERCURY":
            a = 0.387098310  # AU but in km later
            ecc = (
                0.20563175
                + 0.000020406 * T
                - 0.0000000284 * T**2
                - 0.00000000017 * T**3
            )
            inc = (
                7.004986 - 0.0059516 * T + 0.00000081 * T**2 + 0.000000041 * T**3
            )  # degs
            raan = (
                48.330893 - 0.1254229 * T - 0.00008833 * T**2 - 0.000000196 * T**3
            )  # degs
            w_hat = (
                77.456119 + 0.1588643 * T - 0.00001343 * T**2 + 0.000000039 * T**3
            )  # degs
            L = (
                252.250906
                + 149472.6746358 * T
                - 0.00000535 * T**2
                + 0.000000002 * T**3
            )  # degs

        case "VENUS":
            a = 0.723329820  # AU
            ecc = (
                0.00677188
                - 0.000047766 * T
                + 0.000000097 * T**2
                + 0.00000000044 * T**3
            )
            inc = (
                3.394662 - 0.0008568 * T - 0.00003244 * T**2 + 0.000000010 * T**3
            )  # degs
            raan = (
                76.679920 - 0.2780080 * T - 0.00014256 * T**2 - 0.000000198 * T**3
            )  # degs
            w_hat = (
                131.563707 + 0.0048646 * T - 0.00138232 * T**2 - 0.000005332 * T**3
            )  # degs
            L = (
                181.979801
                + 58517.8156760 * T
                + 0.00000165 * T**2
                - 0.000000002 * T**3
            )  # degs

        case "EARTH":
            a = 1.000001018  # AU
            ecc = (
                0.01670862
                - 0.000042037 * T
                - 0.0000001236 * T**2
                + 0.00000000004 * T**3
            )
            inc = (
                0.0000000 + 0.0130546 * T - 0.00000931 * T**2 - 0.000000034 * T**3
            )  # degs
            raan = 0.0  # degs
            w_hat = (
                102.937348 + 0.3225557 * T + 0.00015026 * T**2 + 0.000000478 * T**3
            )  # degs
            L = (
                100.466449
                + 35999.372851 * T
                - 0.00000568 * T**2
                + 0.000000000 * T**3
            )  # degs

        case "MARS":
            a = 1.523679342  # AU
            ecc = (
                0.09340062
                + 0.000090483 * T
                - 0.00000000806 * T**2
                - 0.00000000035 * T**3
            )
            inc = (
                1.849726 - 0.0081479 * T - 0.00002255 * T**2 - 0.000000027 * T**3
            )  # degs
            raan = (
                49.558093 - 0.2949846 * T - 0.00063993 * T**2 - 0.000002143 * T**3
            )  # degs
            w_hat = (
                336.060234 + 0.4438898 * T - 0.00017321 * T**2 + 0.000000300 * T**3
            )  # degs
            L = (
                355.433275
                + 19140.2993313 * T
                + 0.00000261 * T**2
                - 0.000000003 * T**3
            )  # degs

        case "JUPITER":
            a = 5.202603191 + 0.0000001913 * T  # AU
            ecc = (
                0.04849485
                + 0.000163244 * T
                - 0.0000004719 * T**2
                + 0.00000000197 * T**3
            )
            inc = (
                1.303270 - 0.0019872 * T + 0.00003318 * T**2 + 0.000000092 * T**3
            )  # degs
            raan = (
                100.464441 + 0.1766828 * T + 0.00090387 * T**2 - 0.000007032 * T**3
            )  # degs
            w_hat = (
                14.331309 + 0.2155525 * T + 0.00072252 * T**2 - 0.000004590 * T**3
            )  # degs
            L = (
                34.351484
                + 3034.9056746 * T
                - 0.00008501 * T**2
                + 0.000000004 * T**3
            )  # degs

        case "SATURN":
            a = 9.5549009596 - 0.0000021389 * T  # AU
            ecc = (
                0.05550862
                - 0.000346818 * T
                - 0.0000006456 * T**2
                + 0.00000000338 * T**3
            )
            inc = (
                2.488878 + 0.0025515 * T - 0.00004903 * T**2 + 0.000000018 * T**3
            )  # degs
            raan = (
                113.665524 - 0.2566649 * T - 0.00018345 * T**2 + 0.000000357 * T**3
            )  # degs
            w_hat = (
                93.056787 + 0.5665496 * T + 0.00052809 * T**2 - 0.000004882 * T**3
            )  # degs
            L = (
                50.077471
                + 1222.1137943 * T
                + 0.00021004 * T**2
                - 0.000000019 * T**3
            )  # degs

        case "URANUS":
            a = 19.218446062 - 0.0000000372 * T + 0.00000000098 * T**2  # AU
            ecc = (
                0.04629590
                - 0.000027337 * T
                + 0.0000000790 * T**2
                + 0.00000000025 * T**3
            )
            inc = (
                0.773196 - 0.0016869 * T + 0.00000349 * T**2 + 0.00000000016 * T**3
            )  # degs
            raan = (
                74.005947 + 0.0741461 * T + 0.00040540 * T**2 + 0.000000104 * T**3
            )  # degs
            w_hat = (
                173.005159 + 0.0893206 * T - 0.00009470 * T**2 + 0.000000413 * T**3
            )  # degs
            L = (
                314.055005
                + 428.4669983 * T
                - 0.00000486 * T**2
                - 0.000000006 * T**3
            )  # degs

        case "NEPTUNE":
            a = 30.110386869 - 0.0000001663 * T + 0.00000000069 * T**2  # AU
            ecc = 0.00898809 + 0.000006408 * T - 0.0000000008 * T**2
            inc = (
                1.769952 + 0.0002557 * T + 0.00000023 * T**2 - 0.0000000000 * T**3
            )  # degs
            raan = (
                131.784057 - 0.0061651 * T - 0.00000219 * T**2 - 0.000000078 * T**3
            )  # degs
            w_hat = (
                48.123691 + 0.0291587 * T + 0.00007051 * T**2 - 0.000000000 * T**3
            )  # degs
            L = (
                304.348665
                + 218.4862002 * T
                + 0.00000059 * T**2
                - 0.000000002 * T**3
            )  # degs

        case _:
            raise ValueError(
                "Invalid Planet. Options include"
                " MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE"
            )

    a = a * ast.AU

    h = np.sqrt(mu * a * (1 - ecc))

    L = np.mod(L, 360)
    w_hat = np.mod(w_hat, 360)

    raan = np.mod(raan, 360)
    w = np.mod(w_hat - raan, 360)

    Me = np.mod(L - w_hat, 360)

    Me = np.deg2rad(Me)
    if Me < np.pi:
        E_0 = Me - ecc

    else:
        E_0 = Me + ecc

    def f(E: float) -> float:
        """
        Part 1 of Newton Iteration
        """
        return Me - E + ecc * np.sin(E)

    def fp(E: float) -> float:
        """
        Part 2 of Newton's Iteration
        """
        return -1 + ecc * np.sin(E)

    E_1 = E_0 - f(E_0) / fp(E_0)
    err = np.abs(E_1 - E_0)

    while err > 1e-8:
        E_0 = E_1
        E_1 = E_0 - (f(E_0) / fp(E_0))
        err = np.abs(E_1 - E_0)

    ta = np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E_1 / 2)
    ta = 2 * np.arctan(ta)

    # conv to rad
    ta = np.mod(ta, 2 * np.pi)
    raan = np.deg2rad(raan)
    w = np.deg2rad(w)
    inc = np.deg2rad(inc)

    return COES(ecc, inc, raan, w, ta, h, a, mu=mu)
