import numpy as np
import trajectory_planning_helpers as tph
from track_process import import_track
from track_process import prep_track
from track_process import check_track
from min_time import opt_time
import pandas as pd
import matplotlib.pyplot as plt
import result_plot

# 设置路径导入选项
import_track_opts = {"flip_imp_track": False,
                    }

pars = {"curv_cal_opts":
            {"stepsize_curv_preview": 2.0,
             "stepsize_curv_review": 2.0,
             "stepsize_psi_preview": 1.0,
             "stepsize_psi_review": 1.0},
        "stepsize_opts": 3.0,
        "opt_params":
            {"r_delta": 10.0,
             "r_f": 0.01,},
        "veh_params":
            {"g": 9.81,
             "mass": 1200.0,
             "mu": 1.0,
             "dragcoeff": 0.75,
             "liftcoeff_front": 0.45,
             "liftcoeff_rear": 0.75,
             "cog_z": 0.38,
             "wheelbase": 3.0,# wheelbase??
             "k_roll": 0.5,
             "width_front": 1.6,
             "width_rear": 1.6,
             "eps_front": -0.1,
             "eps_rear": -0.1,
             "B_front": 10.0,
             "B_rear": 10.0,
             "C_front": 2.5,
             "C_rear": 2.5,
             "E_front": 1.0,
             "E_rear": 1.0,
             "f_z0": 3000.0,
             "k_brake_front": 0.6,
             "k_drive_front": 0.6,
             "I_z": 1200.0,
             "wheelbase_front": 1.6,
             "wheelbase_rear": 1.4,
             "delta_max": 0.35,
             "delta_min": -0.35,
             "f_drive_max": 7000.0,
             "f_drive_min": -7000.0,
             "v_max": 70.0,
             "width": 3.4,
             "max_power": 230000.0,
             "t_delta": 0.2,
             "t_brake": 0.05,
             "t_drive": 0.05,
             "curvlim": 0.12,
             }}

reg_smooth_opts = {"k_reg": 3.0,
                   "s_reg": 10.0}

stepsize_opts = {"stepsize_prep": 1.0,
               "stepsize_reg": 3.0,
               "stepsize_interp_after_opt": 2.0}

track_file_path = "./data/tracks/berlin_2018.csv"

reftrack_init = import_track(file_path = track_file_path,
                             imp_opts = import_track_opts,
                             wid_veh = pars["veh_params"]["width"])

reftrack_interp, normvec_interp, a_interp, coeffs_x, coeffs_y = prep_track(reftrack= reftrack_init, 
                                                                           reg_smooth_opts= reg_smooth_opts, 
                                                                           stepsize_opts= stepsize_opts)
alpha_opt, v_opt = opt_time(reftrack= reftrack_interp,
                             coeffs_x= coeffs_x,
                             coeffs_y= coeffs_y,
                             pars= pars)

raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
    spline_lengths_opt, el_lengths_opt_interp = tph.creat_raceline.creat_raceline(refline = reftrack_interp[:,:2],
                                                                                  normvectors = normvec_interp,
                                                                                  alpha = alpha_opt,
                                                                                  stepsize_interp = stepsize_opts["stepsize_interp_after_opt"])

# 计算航向和曲率
psi_opt, kappa_opt, = tph.calc_head_curv_an.calc_head_curv_an(coeffs_x = coeffs_x_opt, 
                                                              coeffs_y = coeffs_y_opt, 
                                                              ind_spls = spline_inds_opt_interp, 
                                                              t_spls = t_vals_opt_interp)

# 插值：计算速度
s_splines = np.cumsum(spline_lengths_opt)
s_splines = np.insert(s_splines, 0, 0.0)
vx_profile_opt = np.interp(s_points_opt_interp, s_splines[:-1], v_opt)

# 计算加速度
vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
ax_profile_opt = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                     el_lengths=el_lengths_opt_interp,
                                                     eq_length_output=False)

# 计算圈速
t_profile = tph.calc_t_profile.calc_t_profile(vx_profile=vx_profile_opt,
                                              ax_profile=ax_profile_opt,
                                              el_lengths=el_lengths_opt_interp)
print("INFO: Estimated laptime: %.2fs" % t_profile[-1])

# 绘制速度，加速度图
s_points = np.cumsum(el_lengths_opt_interp[:-1])
s_points = np.insert(s_points, 0, 0.0)

plt.plot(s_points, vx_profile_opt)
plt.plot(s_points, ax_profile_opt)

plt.grid()
plt.xlabel("distance in m")
plt.legend(["vx in m/s", "ax in m/s2"])

plt.show()

# 数据处理
trajectory_opt = np.column_stack((s_points_opt_interp, 
                                  raceline_interp, 
                                  psi_opt, 
                                  kappa_opt, 
                                  vx_profile_opt, 
                                  ax_profile_opt))
spline_data_opt = np.column_stack((spline_inds_opt_interp, 
                                   coeffs_x_opt, 
                                   coeffs_y_opt))

# 使轨迹闭合
traj_cl = np.vstack((trajectory_opt, trajectory_opt[0,:]))
traj_cl[-1, 0] = np.sum(spline_data_opt[:, 0]) 

# 检查轨迹
bound_r, bound_l = check_track(reftrack = reftrack_interp, 
                               reftrack_normvec = normvec_interp, 
                               trajectory = trajectory_opt, 
                               v_max = pars["veh_params"]["v_max"], 
                               length_veh = pars["veh_params"]["length"], 
                               width_veh = pars["veh_params"]["width"], 
                               curvlim = pars["veh_params"]["curvlim"])

# 导出轨迹
export_traj_opt = pd.DataFrame(trajectory_opt, columns = ['s', 'raceline', 'psi', 'kappa', 'vx', 'ax'])
export_traj_opt.to_csv("./data/trajectory.csv", index = False)

result_plot(width_veh = pars["veh_params"]["width"],
            refline = reftrack_interp[:, :2],
            bound_r = bound_r,
            bound_l = bound_l,
            trajectory = trajectory_opt)