import numpy as np
import trajectory_planning_helpers as tph
from min_time import opt_time
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from utils.result_plot import result_plots
from utils.track_process import import_track, prep_track, check_track
import json

itertimes = 1000
warnings.filterwarnings("ignore")
# 设置路径导入选项
import_track_opts = {"flip_imp_track": False,
                    }

with open('./params/params.json', 'r') as f:
    pars = json.load(f)

reg_smooth_opts = {"k_reg": 3,
                   "s_reg": 10}

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

# print(reftrack_interp, coeffs_x, coeffs_y)

alpha_opt, v_opt = opt_time(reftrack= reftrack_interp,
                             coeffs_x= coeffs_x,
                             coeffs_y= coeffs_y,
                             pars= pars,
                             itertimes= itertimes)


raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
    spline_lengths_opt, el_lengths_opt_interp = tph.create_raceline.create_raceline(refline = reftrack_interp[:,:2],
                                                                                  normvectors = normvec_interp,
                                                                                  alpha = alpha_opt,
                                                                                  stepsize_interp = stepsize_opts["stepsize_interp_after_opt"])

""" raceline_mintime = reftrack_interp[:,:2] + np.expand_dims(alpha_opt, 1) * normvec_interp

w_tr_right_mintime = reftrack_interp[:,2] - alpha_opt
w_tr_left_mintime = reftrack_interp[:,3] + alpha_opt

racetrack_mintime = np.column_stack((raceline_mintime, w_tr_right_mintime, w_tr_left_mintime))

raceline_interp, normvec_interp, a_interp, _, _ = prep_track(reftrack= racetrack_mintime, 
                                                       reg_smooth_opts= reg_smooth_opts,
                                                       stepsize_opts= stepsize_opts)

w_tr_tmp = 0.5 * pars["opt_params"]["w_tr_reopt"] * np.ones(reftrack_interp.shape[0])
racetrack_mintime_reopt = np.column_stack((reftrack_interp[:,:2], w_tr_tmp, w_tr_tmp))

print(np.shape(reftrack_interp))
print(np.shape(racetrack_mintime_reopt))
print(np.shape(normvec_interp))

alpha_opt = tph.opt_min_curv.opt_min_curv(reftrack = racetrack_mintime_reopt[:-1,:], 
                                          normvectors = normvec_interp, 
                                          A = a_interp, 
                                          kappa_bound = pars["veh_params"]["curvlim"],
                                          w_veh = pars["veh_params"]["width_front"], 
                                          print_debug = False, 
                                          plot_debug = False) """


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

# plt.show()
plt.savefig("./fig/vx_ax.png")

# 数据处理
trajectory_opt = np.column_stack((s_points_opt_interp, 
                                  raceline_interp, 
                                  psi_opt, 
                                  kappa_opt, 
                                  vx_profile_opt, 
                                  ax_profile_opt))
#spline_data_opt = np.column_stack((spline_inds_opt_interp,coeffs_x_opt, coeffs_y_opt))

# 使轨迹闭合
traj_cl = np.vstack((trajectory_opt, trajectory_opt[0,:]))
traj_cl[-1, 0] = np.sum(spline_inds_opt_interp) 

# 检查轨迹
bound_r, bound_l = check_track(reftrack = reftrack_interp, 
                               reftrack_normvec = normvec_interp, 
                               trajectory = trajectory_opt, 
                               v_max = pars["veh_params"]["v_max"], 
                               length_veh = pars["veh_params"]["length"], 
                               width_veh = pars["veh_params"]["width"], 
                               curvlim = pars["veh_params"]["curvlim"])

# 导出轨迹
export_traj_opt = pd.DataFrame(trajectory_opt, columns = ['s', 'raceline_x','raceline_y', 'psi', 'kappa', 'vx', 'ax'])
export_traj_opt.to_csv("./data/trajectory.csv", index = False)

result_plots(width_veh = pars["veh_params"]["width"],
            refline = reftrack_interp[:, :2],
            bound_r = bound_r,
            bound_l = bound_l,
            trajectory = trajectory_opt)