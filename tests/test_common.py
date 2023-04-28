import pytest

import numpy as np
from ..src import common as com

class Test_RotationMatrices:
    eye = np.eye(3) # identity matrix => no rotation
    fail_format = lambda self, test, real: "TEST:\n{}\
                                            \nREAL:\n{}\
                                            \nDIFF:\n{}".format(test, real, np.equal(test,real))    # basic format to print matrices
    def test_Rx_no_angle(self):
        rx0 = com.Rx(0)
        assert np.equal(rx0, self.eye).all(), self.fail_format(rx0, self.eye)
            
    def test_Ry_no_angle(self):
        ry0 = com.Ry(0)
        assert np.equal(ry0, self.eye).all(), self.fail_format(ry0, self.eye)

    def test_Rz_no_angle(self):
        rz0 = com.Rz(0)
        assert np.equal(rz0, self.eye).all(), self.fail_format(rz0, self.eye)
