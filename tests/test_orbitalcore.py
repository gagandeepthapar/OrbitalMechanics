"""
test_orbitalcore
Test cases for Orbital Core module to ensure changes don't affect results
"""
from datetime import datetime
import pytest
import numpy as np
import pandas as pd

from src.cpslo_orbits import astroconsts as ast
from src.cpslo_orbits import orbitalcore as core

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
        return core.StateVector(rx, 0, 0, 0, vy, 0)

    def test_statevector_to_arr(self, basic_statevector):
        """
        Test equality of state vector to array
        """
        rx = 500 + ast.EARTH_RAD
        vy = np.sqrt(ast.EARTH_MU / rx)
        _sv_arr = basic_statevector.to_arr()
        assert np.equal(_sv_arr, [rx, 0, 0, 0, vy, 0]).all()

    def test_statevector_from_arr(self, basic_statevector):
        """
        Test creation of state vector from array
        """
        rx = 500 + ast.EARTH_RAD
        vy = np.sqrt(ast.EARTH_MU / rx)
        sv_arr = np.array([rx, 0, 0, 0, vy, 0])
        sv_from_arr = core.StateVector.from_arr(sv_arr)
        assert basic_statevector == sv_from_arr


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
        rx0 = core.rot_x(0)
        assert np.equal(rx0, identity).all(), self.fail_format(rx0, identity)

    def test_rot_y_no_angle(self, identity):
        """
        Test Ry Rotation
        """
        ry0 = core.rot_y(0)
        assert np.equal(ry0, identity).all(), self.fail_format(ry0, identity)

    def test_rot_z_no_angle(self, identity):
        """
        Test Rz Rotation
        """
        rz0 = core.rot_z(0)
        assert np.equal(rz0, identity).all(), self.fail_format(rz0, identity)

    def test_sphere_data(self):
        """
        Test to check if sphere data returns correct data
        """
        rad = 1
        x, y, z = core.sphere_data(rad)
        r, c = x.shape

        for i in range(r):
            for j in range(c):
                test = np.linalg.norm([x[i, j], y[i, j], z[i, j]])
                assert np.isclose(test, rad)


class TestConversion:
    """
    Test conversion methods
    """

    @pytest.fixture
    def coes_set_A(self) -> core.COES:
        """
        Sample COES from Curtis Ex. 4.3
        """
        inc = np.deg2rad(153.2)
        raan = np.deg2rad(255.3)
        arg = np.deg2rad(20.07)
        theta = np.deg2rad(28.45)
        a = 8788
        h = 58310
        ecc = 0.1712

        return core.COES(ecc, inc, raan, arg, theta, h, a)

    @pytest.fixture
    def statevector_set_A(self) -> core.StateVector:
        """
        Sample statevector from Curtis Ex. 4.3
        """
        r_vec = [-6045, -3490, 2500]
        v_vec = [-3.457, 6.618, 2.533]

        sv = core.StateVector(*r_vec, *v_vec)

        return sv

    @pytest.fixture
    def coes_set_B(self) -> core.COES:
        """
        Sample coes set from example 4.7, Curtis
        """
        h = 80_000
        ecc = 1.4
        inc = np.deg2rad(30)
        raan = np.deg2rad(40)
        arg = np.deg2rad(60)
        theta = np.deg2rad(30)

        return core.COES(ecc, inc, raan, arg, theta, h)

    @pytest.fixture
    def statevector_set_B(self) -> core.StateVector:
        """
        Sample coes set from example 4.7, Curtis
        """
        return core.StateVector(-4040, 4815, 3629, -10.39, -4.772, 1.744)

    @pytest.fixture
    def sample_two_body(self) -> pd.DataFrame:
        """
        Example 2.3 in Curtis
        """
        # given in problem statement
        r_0 = np.array([8000, 0, 6000])
        v_0 = np.array([0, 7, 0])
        sv_0 = np.array([*r_0, *v_0])
        t_span = np.array([0, 4 * 3_600])

        orbit = core.solve_two_body(sv_0, t_span)
        orbit["ALT"] = (
            np.sqrt(orbit.RX**2 + orbit.RY**2 + orbit.RZ**2) - ast.EARTH_RAD
        )
        orbit["VEL"] = np.sqrt(orbit.VX**2 + orbit.VY**2 + orbit.VZ**2)

        return orbit

    def test_juliandate_vallado(self):
        """
        Example 3.4 from Vallado
        """
        ex = datetime(1996, 10, 26, 14, 20)
        test = core.juliandate(ex)
        expect = 2_450_383.097_222_22
        assert np.isclose(test, expect)

    def test_local_sidereal_time(self):
        """
        Example 5.6 from Curtis
        """
        ex = datetime(1992, 8, 20, 12, 14)
        test = core.local_sidereal_time(ex, -104)
        expect = 48.578_787_810
        assert np.isclose(test, expect)

    def test_sv_to_coes(
        self, coes_set_A: core.COES, statevector_set_A: core.StateVector
    ):
        """
        Example 4.3 in Curtis
        """
        sv = statevector_set_A
        test = core.statevector_to_coes(sv)
        expect = coes_set_A
        assert test == expect

    def test_coes_to_sv(
        self, coes_set_B: core.COES, statevector_set_B: core.StateVector
    ):
        """
        Example 4.7 in Curtis
        """
        test = core.coes_to_statevector(coes_set_B)
        expect = statevector_set_B.to_arr()

        print(test)
        print(expect)
        assert np.isclose(test, expect, rtol=1e-3, atol=1e-3).all()

    def test_two_body_minalt(self, sample_two_body: pd.DataFrame):
        """
        Example 2.3 from Curtis
        checking min alt/vel
        """

        orbit = sample_two_body
        min_alt_state = orbit.loc[orbit.ALT.idxmin()]

        test_alt = min_alt_state.ALT
        expect_alt = 3622

        test_vel = min_alt_state.VEL
        expect_vel = 7

        assert np.isclose(test_alt, expect_alt)
        assert np.isclose(test_vel, expect_vel)

    def test_two_body_maxalt(self, sample_two_body: pd.DataFrame):
        """
        Example 2.3 from Curtis
        checking max alt/vel
        """

        orbit = sample_two_body
        min_alt_state = orbit.loc[orbit.ALT.idxmax()]

        test_alt = min_alt_state.ALT
        expect_alt = 9570

        test_vel = min_alt_state.VEL
        expect_vel = 4.39

        assert np.isclose(test_alt, expect_alt, rtol=1e-3)
        assert np.isclose(test_vel, expect_vel, rtol=1e-3)


class TestLambert:
    """
    Test Lambert Solvers
    """

    def test_basic_lambert_solver(self):
        """
        Example 5.2 in Curtis
        """
        r_0 = np.array([5_000, 10_000, 2_100])
        r_F = np.array([-14_600, 2_500, 7_000])
        del_t = 3_600

        v_0, v_F = core.lambert_problem_solver(r_0, r_F, del_t)

        print(v_0)
        print(v_F)

        expect_v0 = np.array([-5.9925, 1.9254, 3.2456])
        expect_vF = np.array([-3.3125, -4.1966, -0.38529])

        assert np.isclose(v_0, expect_v0, rtol=1e-3, atol=1e-3).all()
        assert np.isclose(v_F, expect_vF, rtol=1e-3, atol=1e-3).all()

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

        test = core.universal_variable_propagation(R0, V0, dt)

        assert np.isclose(test, expect, rtol=1e-3, atol=1e-3).all()

        return
