"""
test_perturbations
Test cases for Perturbations module to ensure changes don't affect results
"""

import pytest
import numpy as np

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


