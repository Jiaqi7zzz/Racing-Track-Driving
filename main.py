import numpy as np
from track_process import prep_track
from min_time import opt_time
import pandas as pd
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
             }}
reg_smooth_opts = {"k_reg": 3.0,
                   "s_reg": 10.0}
stepsize_opts = {"stepsize_prep": 1.0,
               "stepsize_reg": 3.0,
               "stepsize_interp_after_opt": 2.0}
reftrack_init = [] # read the reference track from csv
reftrack, _, _, coeffs_x, coeffs_y = prep_track(reftrack= reftrack_init, 
                                                 reg_smooth_opts= reg_smooth_opts, 
                                                 stepsize_opts= stepsize_opts)
opt_time(reftrack= reftrack,
         coeffs_x= coeffs_x,
         coeffs_y= coeffs_y,
         pars= pars)

