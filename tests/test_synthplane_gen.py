import numpy as np
import pytest
import sys
# setting path
sys.path.append('..')
from remove_phase_ramp import synth_plane


def test_data_type():
    slope_c = 1
    slope_r = -1
    n_cycle_c = 3.9
    n_cycle_r = 7.7
    n_columns = n_rows = 20
    xx_m, yy_m = np.meshgrid(np.arange(n_columns), np.arange(n_rows))
    # - Parameters
    with pytest.raises(TypeError):
        synth_complex = synth_plane(slope_c, slope_r, 7.6, n_rows, n_cycle_c, n_cycle_r, xx_m, yy_m)

    with pytest.raises(TypeError):
        synth_complex = synth_plane(slope_c, slope_r, n_columns, n_rows, n_cycle_c, n_cycle_r,
                                    xx_m, np.zeros([np.arange(n_columns)]))
