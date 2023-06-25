"""
ASTROCONSTS
Astronomical Constants relevant for Orbital Mechanics
"""

# pylint: disable=pointless-string-statement
"""
TEXT MODIFIERS
"""
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DEFAULT = "\033[0m"

"""
EARTH CONSTANTS
"""
EARTH_MU = 398600  # [km/s2] Earth Gravitational Parameter
EARTH_RAD = 6378  # [km] Earth Radius
EARTH_MUSTAR = 0.01215  # [~] Earth Gravitational Parameter in Canonical Units
EARTH_ANGVEL = [0, 0, 7.2921159e-5]  # [rad/s] Angular velocity of the Earth

"""
SUN CONSTANTS
"""
SUN_MU = int(1.3271244e11)  # [km/s2] Sun gravitational parameter
SUN_PSR = 1367 / (2.998e8)  # [N/m2] Solar radiation from Sun at 1 AU
AU = 149597870.691  # [km] conversion of 1AU to km

"""
MISC PLANETARY PARAMETERS
"""
