"""
test_orbitalcore
Test cases for Orbital Core module to ensure changes don't affect results
"""
import pytest
import numpy as np

from ..src import astroconsts as ast
from ..src import orbitalcore as com


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


class TestGeneric:
    """
    Test `GENERIC FUNCTIONS` section of orbital core
    """

    @pytest.fixture
    def identity(self):
        """
        Return identity matrix i.e., no rotation
        """
        return np.eye(3)

    def fail_format(self, test, real) -> str:
        """
        String formatting for rotation matrix failure
        """
        return f"TEST:\n{test}\nREAL:\n{real}\nDIFF:\n{np.equal(test, real)}"

    def test_rot_x_no_angle(self, identity):
        """
        Test Rx rotation
        """
        rx0 = com.rot_x(0)
        assert np.equal(rx0, identity).all(), self.fail_format(rx0, identity)

    def test_rot_y_no_angle(self, identity):
        """
        Test Ry Rotation
        """
        ry0 = com.rot_y(0)
        assert np.equal(ry0, identity).all(), self.fail_format(ry0, identity)

    def test_rot_z_no_angle(self, identity):
        """
        Test Rz Rotation
        """
        rz0 = com.rot_z(0)
        assert np.equal(rz0, identity).all(), self.fail_format(rz0, identity)


class TestOrbitClass:
    """
    Test basic Orbit Classes
    """

    @pytest.fixture
    def basic_statevector(self):
        """
        return basic state vector:
        500km alt, circular orbit, no inclination
        """
        rx = 500 + ast.EARTH_RAD
        vy = np.sqrt(ast.EARTH_MU / rx)
        return com.StateVector(rx, 0, 0, 0, vy, 0)

    def test_statevector_to_arr(self, basic_statevector):
        rx = 500 + ast.EARTH_RAD
        vy = np.sqrt(ast.EARTH_MU / rx)
        _sv_arr = basic_statevector.to_arr()
        assert np.equal(_sv_arr, [rx, 0, 0, 0, vy, 0]).all()
        return

    def test_statevector_from_arr(self, basic_statevector):
        rx = 500 + ast.EARTH_RAD
        vy = np.sqrt(ast.EARTH_MU / rx)
        sv_arr = np.array([rx, 0, 0, 0, vy, 0])
        sv_from_arr = com.StateVector.from_arr(sv_arr)
        assert basic_statevector == sv_from_arr
        return


class TestLambert:
    """
    Test Lambert Solvers
    """

    def test_universal_variable(self):
        """
        Test for Universal Variable Propagation
        Test from Example 3.7 from "Orbital Mechanics for Engineering Students", Curtis
        """

        R0 = [7000, -12124, 0]
        V0 = [2.6679, 4.6210, 0]
        dt = 60 * 60

        RF = [-3296.8, 7413.9, 0]
        VF = [-8.2977, -0.96309, 0]
        expect = np.array([*RF, *VF])

        test = com.universal_variable_propagation(R0, V0, dt)

        assert np.isclose(test, expect, rtol=1e-3).all()

        return
