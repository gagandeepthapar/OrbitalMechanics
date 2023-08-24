"""
test_perturbations
Test cases for Perturbations module to ensure changes don't affect results
"""
import pytest
import numpy as np
from scipy.integrate import solve_ivp

from src.cpslo_orbits import astroconsts as ast
from src.cpslo_orbits import orbitalcore as core
from src.cpslo_orbits import perturbations as pert

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


class TestPerturbations:
    """
    Test different perturbations and propagation methods
    """

    @pytest.fixture
    def sample_orbit(self) -> core.COES:
        # given orbit
        rp = 6678
        ra = 9940
        ecc = (ra - rp) / (ra + rp)
        semi = 0.5 * (ra + rp)
        raan = np.deg2rad(45)
        inc = np.deg2rad(28)
        arg = np.deg2rad(30)
        theta = np.deg2rad(40)

        orbit = core.COES(ecc, inc, raan, arg, theta, semi_major=semi)

        return orbit

    def test_drag_cowells(self):
        """
        Example 10.1 in Curtis
        """
        # given orbit
        rp = 6593
        ra = 7317
        ecc = (ra - rp) / (ra + rp)
        semi_maj = 0.5 * (ra + rp)
        raan = np.deg2rad(340)
        inc = np.deg2rad(65.1)
        arg = np.deg2rad(58)
        theta = np.deg2rad(332)

        mass = 100
        surf_area = (1 / 2) ** 2 * np.pi
        cd = 2.2

        orbit = core.COES(ecc, inc, raan, arg, theta, semi_major=semi_maj)
        state0 = core.coes_to_statevector(orbit)
        drag_args = (mass, surf_area, cd)

        ivp_sol = solve_ivp(
            pert.cowells_method,
            [0, 120 * 86_400],
            state0,
            events=pert.drag_event_listener,
            args=(
                ast.EARTH_MU,
                [(pert.drag_perturbation, drag_args)],
            ),
            rtol=1e-6,
            atol=1e-6,
        )
        fdata = core.ivp_result_to_dataframe(ivp_sol)
        fdata["ALT"] = (
            np.sqrt(fdata.RX**2 + fdata.RY**2 + fdata.RZ**2) - ast.EARTH_RAD
        )
        fdata["DAYS"] = fdata.TIME / (24 * 3_600)

        fin_day = fdata.DAYS.iloc[-1]
        assert 100 < fin_day and fin_day < 110

    def test_j2_enckes(self, sample_orbit: core.COES):
        """
        Example 10.2 in Curtis
        """
        orbit = sample_orbit

        state0 = core.coes_to_statevector(orbit)
        j2_args = (6, ast.EARTH_RAD)

        state_hist = pert.solve_enckes_method(
            state0, [0, 48 * 3_600], perturbs=[(pert.oblateness_perturbation, j2_args)]
        )

        f_orbit = core.statevector_to_coes(state_hist[-1])

        raan0 = orbit.raan_rad
        raanF = f_orbit.raan_rad

        arg0 = orbit.arg_peri_rad
        argF = f_orbit.arg_peri_rad

        # form test params
        t_raan = (raanF - raan0) / 48
        t_arg = (argF - arg0) / 48

        e_raan = -0.172
        e_arg = 0.282

        # this is incorrect
        assert np.isclose(t_raan, e_raan, rtol=1e-3)
        assert np.isclose(t_arg, e_arg, rtol=1e-3)

    def test_vop_j2(self, sample_orbit: core.COES):
        """
        Example 10.6 in Curtis, same as 10.2
        """
        orbit = sample_orbit
        orbit_state = orbit.to_arr()[:-1]
        j2_args = (2, ast.EARTH_RAD)

        ivp_sol = solve_ivp(
            pert.variation_of_params,
            [0, 48 * 3_600],
            orbit_state,
            atol=1e-5,
            rtol=1e-5,
            args=(
                ast.EARTH_MU,
                [(pert.oblateness_perturbation, j2_args)],
            ),
        )

        labels = ["ECC", "INC", "RAAN", "ARG", "THETA", "H"]
        f_data = core.ivp_result_to_dataframe(ivp_sol, labels)

        print(f_data)

        print(orbit)
        print(f_data.iloc[-1])

        raan0 = orbit.raan_rad
        raanF = f_data.iloc[-1].RAAN

        arg0 = orbit.arg_peri_rad
        argF = f_data.iloc[-1].ARG

        t_raan = (raanF - raan0) / 48
        t_arg = (argF - arg0) / 48

        e_raan = -0.172
        e_arg = 0.282

        print(t_raan, e_raan)
        print(t_arg, e_arg)

        # this is incorrect
        assert np.isclose(t_raan, e_raan, rtol=1e-3)
        assert np.isclose(t_arg, e_arg, rtol=1e-3)
