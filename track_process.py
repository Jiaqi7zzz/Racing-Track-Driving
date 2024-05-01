import numpy as np
import trajectory_planning_helpers as tph

def prep_track(reftrack: np.ndarray,
               reg_smooth_opts: dict,
               stepsize_opts: dict
               ) -> tuple:
    '''
    reftrack:               传入的初始路径[x, y, width_left, width_right]
    reg_smooth_opts:        样条逼近的参数
    stepsize_opts:          样条逼近前后的步长参数

    return:
    reftrack_interp:        插值后的路径，格式与reftrack同
    normvec_interp:         参考线上规范化的法向量，包括 x 坐标和 y 坐标
    a_interp:               LES系数
    coeffs_x:               样条系数矩阵
    coeffs_y:               样条系数矩阵
    '''
    reftrack_interp = tph.spline_approximation. \
        spline_approximation(track= reftrack,
                             k_reg= reg_smooth_opts["k_reg"],
                             s_reg= reg_smooth_opts["s_reg"],
                             stepsize_prev= stepsize_opts["stepsize_prev"],
                             stepsize_reg= stepsize_opts["stepsize_reg"],
                             debug= False)
    
    closed_reftrack = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))
    coeffs_x, coeffs_y, a_interp, normvec_interp = tph.calc_splines.cal_splines(path= closed_reftrack)
    
    # NOTES: return value is not close
    return  reftrack_interp, normvec_interp, a_interp, coeffs_x, coeffs_y