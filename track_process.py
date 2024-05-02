import numpy as np
import trajectory_planning_helpers as tph
import math

def import_track(file_path: str,
                 imp_opts: dict,
                 wid_veh: float,
                ) -> np.ndarray:
    '''
    file_path:               路径文件路径
    imp_opt:                 导入路径的参数
    wid_veh:                 车辆宽度

    return:
    reftrack:                导入的初始路径[x, y, width_left, width_right]
    '''
    csv_data = np.loadtxt(file_path, comments='#', delimiter=',')
    
    # 获取坐标和轨道宽度
    if np.shape(csv_data)[1] > 0:
        mid_line = csv_data[:, 0:2]
        wid_track_r = csv_data[:,2]
        wid_track_l = csv_data[:,3]
    else:
        raise IOError("No track data found in file {}".format(file_path))

    reftrack = np.column_stack((mid_line, wid_track_l, wid_track_r))
    
    # 翻转路径（用于测试）
    if imp_opts["flip_imp_track"]:
        reftrack = np.flipud(reftrack)
        
    # 判断路径最小宽度是否大于车辆宽度
    wid_track_min = np.amin(wid_track_r + wid_track_l)
    
    if wid_track_min < wid_veh:
        print("WARNING: Minimum track width %.2fm is close to or smaller than vehicle width!" % np.amin(wid_track_min))
    
    return reftrack
        
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
                             stepsize_prep= stepsize_opts["stepsize_prep"],
                             stepsize_reg= stepsize_opts["stepsize_reg"],
                             debug= False)
    
    closed_reftrack = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))
    coeffs_x, coeffs_y, a_interp, normvec_interp = tph.calc_splines.cal_splines(path= closed_reftrack)
    
    # NOTES: return value is not close
    return  reftrack_interp, normvec_interp, a_interp, coeffs_x, coeffs_y

def check_track(reftrack: np.ndarray, 
                reftrack_normvec: np.ndarray, 
                trajectory: np.ndarray, 
                v_max: float, 
                length_veh: float, 
                width_veh: float, 
                curvlim: float) -> tuple:
    
    # 计算边界
    bound_r = reftrack[:, :2] + reftrack_normvec * np.expand_dims(reftrack[:, 2], 1)
    bound_l = reftrack[:, :2] - reftrack_normvec * np.expand_dims(reftrack[:, 3], 1)
    
    # 检查赛道边界是否与车辆边界
    bound_r_temp = np.column_stack((bound_r, np.zeros((bound_r.shape[0], 2))))
    bound_l_temp = np.column_stack((bound_l, np.zeros((bound_l.shape[0], 2))))
    
    bound_r_interp = interp_track(reftrack = bound_r_temp, 
                                  stepsize_approx = 1.0)
    bound_l_interp = interp_track(reftrack = bound_l_temp, 
                                  stepsize_approx = 1.0)
    
    # 计算每个车辆边界到边界的最小距离
    min_dists = calc_min_bound_dists(trajectory = trajectory, 
                                     bound_r = bound_r_interp, 
                                     bound_l = bound_l_interp, 
                                     length_veh = length_veh, 
                                     width_veh = width_veh)
    if min_dists < 1.0:
        print("WARNING: Vehicle is close to the track boundaries!")
        
    # 检查轨迹曲率是否大于曲率限制
    if np.amax(np.abs(trajectory[:, 4])) > curvlim:
        print("WARNING: Curvature limit is exceeded: %.3frad/m" % np.amax(np.abs(trajectory[:, 4])))
        
    # 检查是否超过最大速度限制
    if np.any(trajectory[:, 5] > v_max + 0.1):
        print("WARNING: Maximum speed limit is exceeded!")
        
    return bound_r, bound_l

def interp_track(reftrack: np.ndarray, 
                 stepsize_approx: float = 1.0) -> np.ndarray:
    
    reftrack_cl = np.vstack((reftrack, reftrack[0]))
    
    el_lengths = np.sqrt(np.sum(np.power(np.diff(reftrack_cl[:,:2], axis=0), 2), axis=1))
    
    # 起点到每个元素的距离总和
    dists_cum = np.cumsum(el_lengths)
    dists_cum = np.insert(dists_cum, 0, 0.0)
    
    no_points_interp = math.ceil(dists_cum[-1] / stepsize_approx) + 1
    dists_interp = np.linspace(0.0, dists_cum[-1], no_points_interp)
    
    # 插值闭合轨迹点
    reftrack_interp_cl = np.zeros((no_points_interp, 4))
    reftrack_interp_cl[:, 0] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 0])
    reftrack_interp_cl[:, 1] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 1])
    reftrack_interp_cl[:, 2] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 2])
    reftrack_interp_cl[:, 3] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 3])
    
    reftrack_interp = reftrack_interp_cl[:-1]
    
    return reftrack_interp

def calc_min_bound_dists(trajectory: np.ndarray, 
                         bound_r: np.ndarray, 
                         bound_l: np.ndarray, 
                         length_veh: float, 
                         width_veh: float) -> tuple:
    
    bounds = np.vstack((bound_r, bound_l))
    
    # 计算航向角为0时静止的车辆边界坐标
    fl = np.array([-width_veh / 2, length_veh / 2])
    fr = np.array([width_veh / 2, length_veh / 2])
    rl = np.array([-width_veh / 2, -length_veh / 2])
    rr = np.array([width_veh / 2, -length_veh / 2])
    
    
    min_dists = np.zeros(trajectory.shape[0])
    mat_rot = np.zeros((2, 2)) # 旋转矩阵
    
    # 遍历轨迹
    for i in range(trajectory.shape[0]):
        mat_rot[0, 0] = math.cos(trajectory[i, 3])
        mat_rot[0, 1] = -math.sin(trajectory[i, 3])
        mat_rot[1, 0] = math.sin(trajectory[i, 3])
        mat_rot[1, 1] = math.cos(trajectory[i, 3])
        
        fl_temp = trajectory[i, 1:3] + np.matmul(mat_rot, fl)
        fr_temp = trajectory[i, 1:3] + np.matmul(mat_rot, fr)
        rl_temp = trajectory[i, 1:3] + np.matmul(mat_rot, rl)
        rr_temp = trajectory[i, 1:3] + np.matmul(mat_rot, rr)
        
        # 计算最小边界
        fl__mindist = np.sqrt(np.power(bounds[:, 0] - fl_temp[0], 2) + np.power(bounds[:, 1] - fl_temp[1], 2))
        fr__mindist = np.sqrt(np.power(bounds[:, 0] - fr_temp[0], 2) + np.power(bounds[:, 1] - fr_temp[1], 2))
        rl__mindist = np.sqrt(np.power(bounds[:, 0] - rl_temp[0], 2) + np.power(bounds[:, 1] - rl_temp[1], 2))
        rr__mindist = np.sqrt(np.power(bounds[:, 0] - rr_temp[0], 2) + np.power(bounds[:, 1] - rr_temp[1], 2))
        
        min_dists[i] = np.amin((fl__mindist, fr__mindist, rl__mindist, rr__mindist))
    
    return min_dists