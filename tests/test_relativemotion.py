"""
test_relativemotion
Test cases for Relative Motion module to ensure changes don't affect results
"""
import numpy as np

from src.cpslo_orbits import relativemotion as rm

# from ..src import relativemotion as rm

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
    Test Conversion and basic Relative Motion Methods
    """

    def test_eci_lvlh(self):
        """
        Example 7.1 in Curtis
        """
        # target pos, vel in example
        t_r = [-266.77, 3865.8, 5426.2]
        t_v = [-6.4836, -3.6198, 2.4156]

        test = rm.eci_to_lvlh(t_r, t_v)
        expect = np.array(
            [
                [-0.040009, 0.57977, 0.81380],
                [-0.82977, -0.47302, 0.29620],
                [0.55667, -0.66341, 0.5],
            ]
        )

        assert np.isclose(test, expect).all()

    def test_five_term_accel(self):
        """
        Example 7.1 in Curtis
        """
        # target, chaser pos/vel in ECI
        target_r = [-266.77, 3865.8, 5426.2]
        target_v = [-6.4836, -3.6198, 2.4156]
        chaser_r = [-5890.7, -2979.8, 1792.2]
        chaser_v = [0.93583, -5.2403, -5.5009]

        t_relR, t_relV, t_relA = rm.five_term_acceleration(
            target_r, target_v, chaser_r, chaser_v
        )

        e_relR = [-6701.2, 6828.3, -406.26]
        e_relV = [0.31667, 0.11199, 1.2470]
        e_relA = [-0.00022222, -0.00018074, 0.00050593]

        assert np.isclose(t_relR, e_relR, rtol=1e-3).all()
        assert np.isclose(t_relV, e_relV, rtol=1e-3).all()
        assert np.isclose(t_relA, e_relA, rtol=1e-3).all()

    def test_cw_relmotion(self):
        """
        Example 7.4 in Curtis
        """
        c_delR0 = [20, 20, 20]  # [km]
        c_delV0 = [0.00930458, -0.0467472, 0.00798343]  # [km/s] delV0+

        del_t = 8 * 3_600  # [sec]
        mean_mot = 0.00115691  # [rad/s]

        _, t_vF = rm.clohessy_wiltshire_relmot(c_delR0, c_delV0, mean_mot, del_t)
        e_vF = [-0.0257978, -0.000470870, -0.0244767]

        assert np.isclose(t_vF, e_vF, rtol=5e-3).all()


class TestPropagation:
    """
    Test propagation methods

    Note: No sufficient example exists in Curtis or Vallado to test methods
    """
