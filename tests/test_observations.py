"""
test_observations
Test cases for Perturbations module to ensure changes don't affect results
"""
import pytest
import numpy as np
from scipy.integrate import solve_ivp

from src.cpslo_orbits import astroconsts as ast

# from src.cpslo_orbits import orbitalcore as core
from src.cpslo_orbits import orbitalcore as core
from src.cpslo_orbits import observations as obs

# from ..src import astroconsts as ast
# from ..src import orbitalcore as core
# from ..src import observations as obs

# pylint: disable=C0103


class TestSanity:
    """
    Sanity check for pytest
    """

    def test_add(self):
        """
        sanity check
        """
        test = 1 + 2
        expect = 3

        assert np.equal(test, expect)

    def test_isclose(self):
        """
        sanity check for close values
        """
        test = 1 + 2
        expect = 3.001
        assert np.isclose(test, expect, rtol=1e-3, atol=1e-3)


class TestConversions:
    """
    Test conversion methods presented
    """

    def test_site_location_ecef(self):
        assert False

    def test_site_location_eci(self):
        assert False

    def test_radec_to_vec(self):
        assert False

    def test_az_el_to_radec(self):
        assert False

    def test_razel(self):
        assert False

    def test_site_track_ecef(self):
        assert False


class TestOrbitDet:
    """
    Test Position-based Orbit Determination Solvers
    """

    def test_gibbs(self):
        assert False

    def test_herrick_gibbs(self):
        assert False


class TestIOD:
    """
    Test Initial/Observation-based Orbit Determination Solvers
    """

    def test_gauss_iod(self):
        assert False

    def test_double_r_iod(self):
        assert False


class TestLambertPropagators:
    """
    Test Lambert and Propagation Solvers presented by Vallado
    """

    def test_uni_var_prop_vallado(self):
        assert False

    def test_izzo_gooding(self):
        assert False

    def test_uni_var_lambert_vallado(self):
        assert False

    def test_gauss_lambert(self):
        assert False


class TestPlanetaryData:
    """
    Test Planetary Ephemeris function
    """

    def test_mercury(self):
        assert False

    def test_venus(self):
        assert False

    def test_earth(self):
        assert False

    def test_mars(self):
        assert False

    def test_jupiter(self):
        assert False

    def test_saturn(self):
        assert False

    def test_uranus(self):
        assert False

    def test_neptune(self):
        assert False
