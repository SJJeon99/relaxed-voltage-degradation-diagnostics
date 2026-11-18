import numpy as np
from scipy.interpolate import interp1d

try:
    from .load_OCP import (
        function_OCPn_smooth_extrapolated,
        function_OCPp_smooth_extrapolated,
    )
except ImportError:
    from src.load_OCP import (
        function_OCPn_smooth_extrapolated,
        function_OCPp_smooth_extrapolated,
    )





def get_OCV_inv_extrapolated(x0,y0,Q_NE,Q_PE):
    x0_fresh, y0_fresh = x0,y0
    Q_NE_fresh = Q_NE
    Q_PE_fresh = Q_PE
    
    Q_vec = np.linspace(-50,150,20000)
    x_vec = x0_fresh + Q_vec/Q_NE_fresh
    y_vec = y0_fresh - Q_vec/Q_PE_fresh
    Vn_vec = function_OCPn_smooth_extrapolated(x_vec)
    Vp_vec = function_OCPp_smooth_extrapolated(y_vec)
    OCV_vec = Vp_vec - Vn_vec
    return interp1d(OCV_vec, Q_vec, fill_value = 'extrapolate', bounds_error = False)



def get_OCV_extrapolated(x0, y0, Q_NE, Q_PE, V_max=4.2):
    x0_fresh, y0_fresh = x0, y0
    Q_NE_fresh = Q_NE
    Q_PE_fresh = Q_PE

    Q_vec = np.linspace(0, 130, 20000)
    x_vec = x0_fresh + Q_vec / Q_NE_fresh
    y_vec = y0_fresh - Q_vec / Q_PE_fresh

    Vn_vec = function_OCPn_smooth_extrapolated(x_vec)
    Vp_vec = function_OCPp_smooth_extrapolated(y_vec)
    OCV_vec = Vp_vec - Vn_vec

    idx_over = np.where(OCV_vec >= V_max)[0]
    if len(idx_over) > 0:
        idx_max = idx_over[0]
        Q_cell_fresh = Q_vec[idx_max]
        x0_fresh + Q_cell_fresh / Q_NE_fresh
        y0_fresh - Q_cell_fresh / Q_PE_fresh
    else:
        Q_cell_fresh = Q_vec[-1]
        x_vec[-1]
        y_vec[-1]

    return interp1d(Q_vec, OCV_vec, fill_value='extrapolate')



