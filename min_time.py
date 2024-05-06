import numpy as np
import casadi as ca
import trajectory_planning_helpers as tph
import pandas as pd
import tqdm

def opt_time(reftrack:      np.ndarray,
             coeffs_x:      np.ndarray,
             coeffs_y:      np.ndarray,
             pars:          dict
             )       ->     tuple:
    
    # ------------------------------------------------------------------------ #
    # --------------------------- TRACK PROCESSING ------------------------------------- #
    # ------------------------------------------------------------------------ #

    # number of all the points
    points_num = reftrack.shape[0]
    discrete_points = np.arange(points_num)
    a_interp = None
    
    # spline length and heading&kappa
    reftrack_length : np.ndarray = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x , coeffs_y=coeffs_y)
    
    reftrack_kappa = tph.calc_head_curv_num.calc_head_curv_num(path=reftrack[:,:2],
                                                               el_lengths=reftrack_length,
                                                               is_closed=True,
                                                               stepsize_curv_preview=pars["curv_calc_opts"]["stepsize_curv_preview"],
                                                               stepsize_curv_review=pars["curv_calc_opts"]["stepsize_curv_review"],
                                                               stepsize_psi_preview=pars["curv_calc_opts"]["stepsize_psi_preview"],
                                                               stepsize_psi_review=pars["curv_calc_opts"]["stepsize_psi_review"],
                                                               calc_curv=True)[1]
    # close the track
    # reftrack_kappa.append(reftrack_kappa[0])
    closed_reftrack_kappa = np.append(reftrack_kappa, reftrack_kappa[0])
    # discrete_points.append(discrete_points[0])
    closed_discrete_points = np.append(discrete_points, points_num)
    w_tr_left = np.append(reftrack[:, 2], reftrack[0, 2])
    w_tr_right = np.append(reftrack[:, 3], reftrack[0, 3])

    '''
    step opts:
        step_size : 每一步的间隔
        steps : (0, 1, ..., points_num)
        intervals : 需要解决几段间隔 这里由于是闭合的 
                    因此以三个点为例 这里就会有三段间隔 长度为四的数组
    '''

    step_size           = pars["stepsize_opts"] # 取3.0
    steps               = np.arange(points_num + 1)
    intervals           = points_num
    # s_opt               = np.asarray(closed_discrete_points) * step_size

    # 定义了三个插值器 可以根据点的信息得到对应的曲率/赛道左右宽度
    kappa_interp        = ca.interpolant('kappa_interp', 'linear', [steps], closed_reftrack_kappa)
    w_tr_left_interp    = ca.interpolant('w_tr_left_interp', 'linear', [steps], w_tr_left)
    w_tr_right_interp   = ca.interpolant('w_tr_right_interp', 'linear', [steps], w_tr_right)

    # ------------------------------------------------------------------------ #
    # --------------------------- GAUSS-LEGENDER COLLOCATION --------------------------------- #
    # ------------------------------------------------------------------------ #

    d   = 3
    tau = np.append(0, ca.collocation_points(d, 'legendre'))
    B_r = np.zeros(d + 1)
    D   = np.zeros(d + 1)
    C   = np.zeros((d + 1, d + 1))

    for i in range(d + 1):
        p = np.poly1d([1])
        for j in range(d + 1):
            if i != j:
                p *= np.poly1d([1, -tau[j]]) / (tau[i] - tau[j])
        
        D[i] = p(1.0) # 连续性方程

        p_der = np.polyder(p) # 多项式的导数
        for j in range(d + 1):
            C[i, j] = p_der(tau[j])

        p_int = np.polyint(p) # 拉格朗日正交基的积分
        B_r[i] = p_int(1.0)

    # ------------------------------------------------------------------------ #
    # --------------------------- STATE VARIABLES ------------------------------------- #
    # ------------------------------------------------------------------------ #
    
    # we here choose n, xi, v, beta, omega_z as state variables
    # 以下所有变量均表示为：单位向量 * 标量，均与论文中对应，详见figure1

    # n 为横向误差，左为正，单位m
    n_n = ca.SX.sym('n_n')
    n_s = 5.0
    n = n_n * n_s

    # xi 为航向误差，单位rad
    xi_n = ca.SX.sym('xi_n')
    xi_s = 1.0
    xi = xi_n * xi_s

    # v 为纵向速度，单位m/s
    v_n = ca.SX.sym('v_n')
    v_s = 50
    v = v_n * v_s

    # beta 为侧滑角，单位rad
    beta_n = ca.SX.sym('beta_n')
    beta_s = 0.5
    beta = beta_n * beta_s

    # omega_z 为横摆角速度，单位rad/s
    omega_z_n = ca.SX.sym('omega_z_n')
    omega_z_s = 1.0
    omega_z = omega_z_n * omega_z_s

    # x_s是状态变量的尺度大小，x是状态变量的向量形式
    x_s = np.array([n_s, xi_s, v_s, beta_s, omega_z_s])
    x = ca.vertcat(n_n, xi_n, v_n, beta_n, omega_z_n)

    # ------------------------------------------------------------------------ #
    # --------------------------- CONTROL VARIABLES ------------------------------------- #
    # ------------------------------------------------------------------------ #

    # control variables contain delta, f_drive, f_brake, Gamma_y

    # delta 为方向盘转角，单位rad
    delta_n = ca.SX.sym('delta_n')
    delta_s = 0.5
    delta = delta_n * delta_s

    # f_drive 为驱动力，单位N
    f_drive_n = ca.SX.sym('f_drive_n')
    f_drive_s = 7500.0
    f_drive = f_drive_n * f_drive_s

    # f_brake 为制动力，单位N
    f_brake_n = ca.SX.sym('f_brake_n')
    f_brake_s = 20000.0
    f_brake = f_brake_n * f_brake_s

    # Gamma_y 为动态车轮载荷变化，单位N
    gamma_y_n = ca.SX.sym('gamma_y_n')
    gamma_y_s = 5000.0
    gamma_y = gamma_y_n * gamma_y_s

    # u_s是控制变量的尺度大小，u是控制变量的向量形式
    u_s = np.array([delta_s, f_drive_s, f_brake_s, gamma_y_s])
    u = ca.vertcat(delta_n, f_drive_n, f_brake_n, gamma_y_n)

    # ------------------------------------------------------------------------ #
    # --------------------------- MODEL EQUATIONS ------------------------------------- #
    # ------------------------------------------------------------------------ #

    '''
    弯曲率(kappa)是参考线的曲率，表示道路的弯曲程度。
    拖拽力(f_xdrag)是与速度(v)的平方成正比的阻力。
    滚动阻力(f_xroll)是与车辆质量、重力加速度和轮胎的滚动阻力系数(c_roll)成正比的阻力。根据车辆的前后轴重量分布，这个阻力在前后轮之间进行了分配。
    静态轮胎法向力(f_zstat)是车辆重力按照前后轴重量分布均匀分配在轮胎上的力。
    动态轮胎法向力(f_zlift)是与速度的平方成正比的气动升力。
    动态轮胎法向力(f_zdyn)是由于加速度和制动力产生的重量转移效应。
    所有轮胎法向力的总和(f_z)是静态法向力、动态法向力和气动升力的总和。
    滑移角(alpha)是轮胎实际滑移角和纵向/横向运动的关系。
    横向轮胎力(f_y)是与法向力、滑移角和轮胎参数有关的横向力。
    纵向轮胎力(f_x)是与驱动力、制动力、滚动阻力和轮胎参数有关的纵向力。
    纵向加速度(ax)是纵向轮胎力的总和除以车辆质量。
    横向加速度(ay)是横向轮胎力的总和除以车辆质量。
    '''
    gravity = pars["veh_params"]["g"]
    mass = pars["veh_params"]["mass"]
    # front_proportion = pars["veh_params"]["frontwheel"] / pars["veh_params"]["wheelsbase"]
    # rear_proportion = pars["veh_params"]["rearwheel"] / pars["veh_params"]["wheelsbase"]

    kappa = ca.SX.sym('kappa')
    f_xdrag = pars["veh_params"]["dragcoeff"] * v ** 2

    f_xroll = mass * gravity * pars["veh_params"]["c_roll"] # 滚动阻力系数

    f_xroll_fl = 0.5 * f_xroll * pars["veh_params"]["wheelbase_rear"] / pars["veh_params"]["wheelbase"]
    f_xroll_fr = 0.5 * f_xroll * pars["veh_params"]["wheelbase_rear"] / pars["veh_params"]["wheelbase"]
    f_xroll_rl = 0.5 * f_xroll * pars["veh_params"]["wheelbase_front"] / pars["veh_params"]["wheelbase"]
    f_xroll_rr = 0.5 * f_xroll * pars["veh_params"]["wheelbase_front"] / pars["veh_params"]["wheelbase"]

    f_zstat = mass * gravity
    f_zstat_fl = 0.5 * f_zstat * pars["veh_params"]["wheelbase_rear"] / pars["veh_params"]["wheelbase"]
    f_zstat_fr = 0.5 * f_zstat * pars["veh_params"]["wheelbase_rear"] / pars["veh_params"]["wheelbase"]
    f_zstat_rl = 0.5 * f_zstat * pars["veh_params"]["wheelbase_front"] / pars["veh_params"]["wheelbase"]
    f_zstat_rr = 0.5 * f_zstat * pars["veh_params"]["wheelbase_front"] / pars["veh_params"]["wheelbase"]

    f_zlift_fl = 0.5 * pars["veh_params"]["liftcoeff_front"] * v ** 2
    f_zlift_fr = 0.5 * pars["veh_params"]["liftcoeff_front"] * v ** 2
    f_zlift_rl = 0.5 * pars["veh_params"]["liftcoeff_rear"] * v ** 2
    f_zlift_rr = 0.5 * pars["veh_params"]["liftcoeff_rear"] * v ** 2

    # 加速度和制动力产生的轴向转移效应
    f_tot = f_drive + f_brake - f_xdrag - f_xroll

    f_zdyn_fl = (-0.5 * pars["veh_params"]["cog_z"] / pars["veh_params"]["wheelbase"] * f_tot
                 - pars["veh_params"]["k_roll"] * gamma_y)
    f_zdyn_fr = (-0.5 * pars["veh_params"]["cog_z"] / pars["veh_params"]["wheelbase"] * f_tot
                 + pars["veh_params"]["k_roll"] * gamma_y)
    f_zdyn_rl = (0.5 * pars["veh_params"]["cog_z"] / pars["veh_params"]["wheelbase"] * f_tot
                 - (1.0 - pars["veh_params"]["k_roll"]) * gamma_y)
    f_zdyn_rr = (0.5 * pars["veh_params"]["cog_z"] / pars["veh_params"]["wheelbase"] * f_tot
                 + (1.0 - pars["veh_params"]["k_roll"]) * gamma_y)
    
    f_z_fl = f_zstat_fl + f_zlift_fl + f_zdyn_fl
    f_z_fr = f_zstat_fr + f_zlift_fr + f_zdyn_fr
    f_z_rl = f_zstat_rl + f_zlift_rl + f_zdyn_rl
    f_z_rr = f_zstat_rr + f_zlift_rr + f_zdyn_rr

    # wheelbase_front是底盘中心到前轴的距离 width_front是前轴宽度
    alpha_fl = delta - ca.atan((v * ca.sin(beta) + omega_z * pars["veh_params"]["wheelbase_front"]) /
                                (v * ca.cos(beta) - 0.5 * pars["veh_params"]["width_front"] * omega_z))
    alpha_fr = delta - ca.atan((v * ca.sin(beta) + omega_z * pars["veh_params"]["wheelbase_front"]) /
                                (v * ca.cos(beta) + 0.5 * pars["veh_params"]["width_front"] * omega_z))
    alpha_rl = ca.atan((-v * ca.sin(beta) + omega_z * pars["veh_params"]["wheelbase_rear"]) /
                                (v * ca.cos(beta) - 0.5 * pars["veh_params"]["width_rear"] * omega_z))
    alpha_rr = ca.atan((-v * ca.sin(beta) + omega_z * pars["veh_params"]["wheelbase_rear"]) /
                                (v * ca.cos(beta) + 0.5 * pars["veh_params"]["width_rear"] * omega_z))
    
    # 轮胎横向力(魔术公式)
    # f_y_fl = pars["opt_params"]["mu"] * f_z_fl * (1 + pars["veh_params"]["eps_front"] * f_z_fl / pars["veh_params"]["f_z0"]) * ca.sin()
    
    f_y_fl = (pars["veh_params"]["mu"] * f_z_fl * (1 + pars["veh_params"]["eps_front"] * f_z_fl / pars["veh_params"]["f_z0"])
              * ca.sin(pars["veh_params"]["C_front"] * ca.atan(pars["veh_params"]["B_front"] * alpha_fl - pars["veh_params"]["E_front"]
                                                 * (pars["veh_params"]["B_front"] * alpha_fl - ca.atan(pars["veh_params"]["B_front"] * alpha_fl)))))
    f_y_fr = (pars["veh_params"]["mu"] * f_z_fr * (1 + pars["veh_params"]["eps_front"] * f_z_fr / pars["veh_params"]["f_z0"])
              * ca.sin(pars["veh_params"]["C_front"] * ca.atan(pars["veh_params"]["B_front"] * alpha_fr - pars["veh_params"]["E_front"]
                                                 * (pars["veh_params"]["B_front"] * alpha_fr - ca.atan(pars["veh_params"]["B_front"] * alpha_fr)))))
    f_y_rl = (pars["veh_params"]["mu"] * f_z_rl * (1 + pars["veh_params"]["eps_rear"] * f_z_rl / pars["veh_params"]["f_z0"])
              * ca.sin(pars["veh_params"]["C_rear"] * ca.atan(pars["veh_params"]["B_rear"] * alpha_rl - pars["veh_params"]["E_rear"]
                                                * (pars["veh_params"]["B_rear"] * alpha_rl - ca.atan(pars["veh_params"]["B_rear"] * alpha_rl)))))
    f_y_rr = (pars["veh_params"]["mu"] * f_z_rr * (1 + pars["veh_params"]["eps_rear"] * f_z_rr / pars["veh_params"]["f_z0"])
              * ca.sin(pars["veh_params"]["C_rear"] * ca.atan(pars["veh_params"]["B_rear"] * alpha_rr - pars["veh_params"]["E_rear"]
                                                * (pars["veh_params"]["B_rear"] * alpha_rr - ca.atan(pars["veh_params"]["B_rear"] * alpha_rr)))))
    

    # 轮胎纵向力 equ. 4a 4b
    f_x_fl = 0.5 * f_drive * pars["veh_params"]["k_drive_front"] + 0.5 * f_brake * pars["veh_params"]["k_brake_front"] - f_xroll_fl
    f_x_fr = 0.5 * f_drive * pars["veh_params"]["k_drive_front"] + 0.5 * f_brake * pars["veh_params"]["k_brake_front"] - f_xroll_fr
    f_x_rl = 0.5 * f_drive * (1 - pars["veh_params"]["k_drive_front"]) + 0.5 * f_brake * (1 - pars["veh_params"]["k_brake_front"]) - f_xroll_rl
    f_x_rr = 0.5 * f_drive * (1 - pars["veh_params"]["k_drive_front"]) + 0.5 * f_brake * (1 - pars["veh_params"]["k_brake_front"]) - f_xroll_rr

    # 加速度
    a_x = (f_x_rl + f_x_rr + (f_x_fl + f_x_fr) * ca.cos(delta) - (f_y_fl + f_y_fr) * ca.sin(delta)
          - f_xdrag) / mass
    
    a_y = ((f_x_fl + f_x_fr) * ca.sin(delta) + f_y_rl + f_y_rr + (f_y_fl + f_y_fr) * ca.cos(delta)) / mass


    # ------------------------------------------------------------------------ #
    # --------------------------- DERIVATIVES ------------------------------------- #
    # ------------------------------------------------------------------------ #
    
    sf = (1.0 - n * kappa) / (v * ca.cos(xi + beta))
    dn = sf * v * ca.sin(xi + beta)
    dxi = sf * omega_z - kappa


    dv = (sf / mass) * ((f_x_rl + f_x_rr) * ca.cos(beta) + (f_x_fl + f_x_fr) * ca.cos(delta - beta)
                        + (f_y_rl + f_y_rr) * ca.sin(beta) - (f_y_fl + f_y_fr) * ca.sin(delta - beta)
                        - f_xdrag * ca.cos(beta))
    dbeta = sf * (-omega_z + (-(f_x_rl + f_x_rr) * ca.sin(beta) + (f_x_fl + f_x_fr) * ca.sin(delta - beta)
                              + (f_y_rl + f_y_rr) * ca.cos(beta) + (f_y_fl + f_y_fr) * ca.cos(delta - beta)
                              + f_xdrag * ca.sin(beta)) / (mass * v))
    
    domega_z = (sf / pars["veh_params"]["I_z"]) * ((f_x_rr - f_x_rl) * pars["veh_params"]["width_rear"] / 2
                                    - (f_y_rl + f_y_rr) * pars["veh_params"]["wheelbase_rear"]
                                    + ((f_x_fr - f_x_fl) * ca.cos(delta)
                                       + (f_y_fl - f_y_fr) * ca.sin(delta)) * pars["veh_params"]["width_front"] / 2
                                    + ((f_y_fl + f_y_fr) * ca.cos(delta)
                                       + (f_x_fl + f_x_fr) * ca.sin(delta)) * pars["veh_params"]["width_front"])
    
    dx = ca.vertcat(dn, dxi, dv, dbeta, domega_z) / x_s

    # ------------------------------------------------------------------------ #
    # --------------------------- BOUNDARIES ------------------------------------- #
    # ------------------------------------------------------------------------ #

    delta_min       = pars["veh_params"]["delta_min"] / delta_s
    delta_max       = pars["veh_params"]["delta_max"] / delta_s
    f_drive_min     = 0.0
    f_drive_max     = pars["veh_params"]["f_drive_max"] / f_drive_s
    # f_brake_min     = 0.0
    # f_brake_max     = pars["veh_params"]["f_brake_max"] / f_brake_s
    f_brake_min     = -pars["veh_params"]["f_drive_max"] / f_brake_s
    f_brake_max     = 0.0
    gamma_y_min     = -np.inf
    gamma_y_max     = np.inf

    v_min           = 1.0 / v_s
    v_max           = pars["veh_params"]["v_max"] / v_s
    beta_min        = -0.5 * np.pi / beta_s
    beta_max        = 0.5 * np.pi / beta_s
    omega_z_min     = -0.5 * np.pi / omega_z_s
    omega_z_max     = 0.5 * np.pi / omega_z_s
    xi_min          = -0.5 * np.pi / xi_s
    xi_max          = 0.5 * np.pi / xi_s

    v_0 = 20 / v_s # initial guess

    # ------------------------------------------------------------------------ #
    # --------------------------- FUNCTIONS ------------------------------------- #
    # ------------------------------------------------------------------------ #

    f_dynamic = ca.Function('f_dynamic', [x, u, kappa], [dx, sf], ['x', 'u', 'kappa'], ['dx', 'sf'])
    f_fx      = ca.Function('f_fx', [x, u], [f_x_fl, f_x_fr, f_x_rl, f_x_rr], ['x', 'u'], ['f_x_fl', 'f_x_fr', 'f_x_rl', 'f_x_rr'])
    f_fy      = ca.Function('f_fy', [x, u], [f_y_fl, f_y_fr, f_y_rl, f_y_rr], ['x', 'u'], ['f_y_fl', 'f_y_fr', 'f_y_rl', 'f_y_rr'])
    f_fz      = ca.Function('f_fz', [x, u], [f_z_fl, f_z_fr, f_z_rl, f_z_rr], ['x', 'u'], ['f_z_fl', 'f_z_fr', 'f_z_rl', 'f_z_rr'])
    f_a       = ca.Function('f_a' , [x, u], [a_x, a_y], ['x', 'u'], ['a_x', 'a_y'])
    

    # ------------------------------------------------------------------------ #
    # --------------------------- NLP ------------------------------------- #
    # ------------------------------------------------------------------------ #

    # 初始化NLP向量
    # w是解变量，g是约束函数，J是目标函数
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    # 初始化输出向量
    x_opt = []
    u_opt = []
    ax_opt = []
    ay_opt = []
    ec_opt = []
    dt_opt = []
    tf_opt = []

    # regularization
    delta_p = []
    F_p = []

    # 5 indicates for the num of state variables
    Xk = ca.MX.sym('X0', 5)
    w.append(Xk)
    # Safety Ensurance equ.17
    b_safety = 0.0
    n_min = (-w_tr_right_interp(0) + pars["veh_params"]["width"] / 2 + b_safety) / n_s
    n_max = (w_tr_left_interp(0) - pars["veh_params"]["width"] / 2 - b_safety) / n_s

    n_min = n_min.__float__()
    n_max = n_max.__float__()
    
    # 初始化解变量和输出变量
    lbw.append([n_min, xi_min, v_min, beta_min, omega_z_min])
    ubw.append([n_max, xi_max, v_max, beta_max, omega_z_max])
    w0.append([0.0, 0.0, v_0, 0.0, 0.0])
    x_opt.append(Xk * x_s)
    # step_size 就是优化的步长
    # h = np.diff(s_opt)
    h = np.array([step_size] * intervals)

    for k in range(intervals):
        # print(len(w), len(w0))
        # 在每一段间隔内
        Uk = ca.MX.sym('U_' + str(k), 4) # 4是控制变量的维度
        w.append(Uk)
        lbw.append([delta_min, f_drive_min, f_brake_min, gamma_y_min])
        ubw.append([delta_max, f_drive_max, f_brake_max, gamma_y_max])
        w0.append([0.0, 0.0, 0.0, 0.0])

        Xc = []
        # Xc 是各插值点的状态
        for i in range(d):
            Xkj = ca.MX.sym('X_' + str(k) + '_' + str(i), 5)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-np.inf] * 5)
            ubw.append([np.inf] * 5)
            # lbw.append([n_min, xi_min, v_min, beta_min, omega_z_min])
            # ubw.append([n_max, xi_max, v_max, beta_max, omega_z_max])
            w0.append([0.0, 0.0, v_0, 0.0, 0.0])
        
        Xk_end = D[0] * Xk # 连续性方程
        
        sf_opt = []
        '''
        导数怎么求的不太懂
        '''
        for i in range(1, d + 1):
            # xp是状态变量的导数，表示状态变量的导数在当前插值点的估计值
            xp = C[0, i] * Xk
            for j in range(d):
                xp += C[j + 1, i] * Xc[j]
            
            # 当前插值点的曲率
            kappa_col = kappa_interp(k + tau[j])

            # f_dynamic计算[x, u, kappa] -> [dx, sf]
            dx_temp, sf_temp = f_dynamic(Xc[i - 1], Uk, kappa_col)
            # 状态转移方程，添加约束：dx * delta x = xp
            g.append(h[k] * dx_temp - xp)
            lbg.append([0.0] * 5)
            ubg.append([0.0] * 5)
            # 连续性条件
            Xk_end = Xk_end + D[i] * Xc[i - 1] 
            # 约束即为时间
            J += B_r[i] * sf_temp * h[k]
            sf_opt.append(B_r[i] * sf_temp * h[k])
        
        time = 0
        for i in range(d):
            time += sf_opt[i]
        dt_opt.append(time)
        ec_opt.append(Xk[2] * v_s * Uk[1] * f_drive_s * dt_opt[-1])

        # 每一个interval的最后，开启下一段的内容

        Xk = ca.MX.sym('X_' + str(k + 1), 5)
        w.append(Xk)
        n_min = (-w_tr_right_interp(k + 1) + pars["veh_params"]["width"] / 2.0) / n_s
        n_max = (w_tr_left_interp(k + 1) - pars["veh_params"]["width"] / 2.0) / n_s
        n_min = n_min.__float__()
        n_max = n_max.__float__()

        lbw.append([n_min, xi_min, v_min, beta_min, omega_z_min])
        ubw.append([n_max, xi_max, v_max, beta_max, omega_z_max])
        w0.append([0.0, 0.0, v_0, 0.0, 0.0])
        g.append(Xk_end - Xk)
        lbg.append([0.0] * 5)
        ubg.append([0.0] * 5)

        # tire forces and accelerations
        # x/y/z indicates for direction
        # fl/fr/rl/rr indicates for the type of tire, k for intervals
        f_x_flk, f_x_frk, f_x_rlk, f_x_rrk = f_fx(Xk, Uk)
        f_y_flk, f_y_frk, f_y_rlk, f_y_rrk = f_fy(Xk, Uk)
        f_z_flk, f_z_frk, f_z_rlk, f_z_rrk = f_fz(Xk, Uk)        
        a_x_k, a_y_k = f_a(Xk, Uk)

        # equ.8
        g.append(((f_y_flk + f_y_frk) * ca.cos(Uk[0] * delta_s) + f_y_rlk + f_y_rrk
                  + (f_x_flk + f_x_frk) * ca.sin(Uk[0] * delta_s))
                 * pars["veh_params"]["cog_z"] / ((pars["veh_params"]["width_front"] + pars["veh_params"]["width_rear"]) / 2) - Uk[3] * gamma_y_s)
        lbg.append([0.0])
        ubg.append([0.0])

        mu_fl = pars["veh_params"]["mu"]
        mu_fr = pars["veh_params"]["mu"]
        mu_rl = pars["veh_params"]["mu"]
        mu_rr = pars["veh_params"]["mu"]
        
        # equ.10
        g.append(((f_x_flk / (mu_fl * f_z_flk)) ** 2 + (f_y_flk / (mu_fl * f_z_flk)) ** 2))
        g.append(((f_x_frk / (mu_fr * f_z_frk)) ** 2 + (f_y_frk / (mu_fr * f_z_frk)) ** 2))
        g.append(((f_x_rlk / (mu_rl * f_z_rlk)) ** 2 + (f_y_rlk / (mu_rl * f_z_rlk)) ** 2))
        g.append(((f_x_rrk / (mu_rr * f_z_rrk)) ** 2 + (f_y_rrk / (mu_rr * f_z_rrk)) ** 2))
        lbg.append([0.0] * 4)
        ubg.append([1.0] * 4)

        # equ. 11
        g.append(Xk[2] * Uk[1])
        lbg.append([-np.inf])
        ubg.append([pars["veh_params"]["max_power"] / (f_drive_s * v_s)])



        # equ.12
        # g.append(Uk[1] * Uk[2])
        # lbg.append([0.0])
        # ubg.append([0.0])
        g.append(Uk[1] * Uk[2])
        lbg.append([-20000.0 / (f_drive_s * f_brake_s)])
        # lbg.append([0.0])
        ubg.append([0.0])

        # equ.14
        if k > 0:
            sigma = (1 - kappa_interp(k) * Xk[0] * n_s) / (Xk[2] * v_s)
            g.append((Uk - w[1 + (k - 1) * 5]) / (h[k - 1] * sigma))
            lbg.append([delta_min / (pars["veh_params"]["t_delta"]), -np.inf, f_brake_min / (pars["veh_params"]["t_brake"]), -np.inf])
            ubg.append([delta_max / (pars["veh_params"]["t_delta"]), f_drive_max / (pars["veh_params"]["t_drive"]), np.inf, np.inf])

        # 此处不限制最大加速度

        # 正则化项
        delta_p.append(Uk[0] * delta_s)
        F_p.append(Uk[1] * f_drive_s / 10000.0 + Uk[2] * f_brake_s / 10000.0)

        x_opt.append(Xk * x_s)
        u_opt.append(Uk * u_s)
        tf_opt.extend([f_x_flk, f_y_flk, f_z_flk, f_x_frk, f_y_frk, f_z_frk])
        tf_opt.extend([f_x_rlk, f_y_rlk, f_z_rlk, f_x_rrk, f_y_rrk, f_z_rrk])
        ax_opt.append(a_x_k)
        ay_opt.append(a_y_k)
    
    # 所有循环结束，Xk为最后一项
    # 初始条件等于终止条件
    g.append(w[0] - Xk)
    # lbg.append([0.0 for i in range(5)])
    # ubg.append([0.0 for i in range(5)])
    lbg.append([0.0, 0.0, 0.0, 0.0, 0.0])
    ubg.append([0.0, 0.0, 0.0, 0.0, 0.0])


    # regularization matrix
    # equ.23
    reg_matrix = np.eye(intervals)
    for i in range(intervals - 1):
        reg_matrix[i, i + 1] = -1.0
    reg_matrix[-1, 0] = -1.0

    delta_p = ca.vertcat(*delta_p)
    F_p = ca.vertcat(*F_p)
    reg_delta = ca.mtimes(ca.MX(reg_matrix), delta_p)
    reg_delta = ca.dot(reg_delta, reg_delta)
    reg_F = ca.mtimes(ca.MX(reg_matrix), F_p)
    reg_F = ca.dot(reg_F, reg_F)

    # equ.24
    J += pars["opt_params"]["r_delta"] * reg_delta + pars["opt_params"]["r_F"] * reg_F

    
    # 在casadi中，需要将所有向量展平成一维向量
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    x_opt = ca.vertcat(*x_opt)
    u_opt = ca.vertcat(*u_opt)
    tf_opt = ca.vertcat(*tf_opt)
    ax_opt = ca.vertcat(*ax_opt)
    ay_opt = ca.vertcat(*ay_opt)
    ec_opt = ca.vertcat(*ec_opt)
    dt_opt = ca.vertcat(*dt_opt)

    nlp_prob = {'f': J, 'x': w, 'g': g}


    opts_setting = {"expand": True, "ipopt.max_iter": 100, "ipopt.tol": 1e-7}
    # opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # print(g.shape, lbg.shape, ubg.shape, w.shape, lbw.shape, ubw.shape)
    res = solver(x0 = w0, lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)

    print("Solved")
    for i in range(200):
        print(res['x'][i])
    
    
    # f_solution = ca.Function("f_solution", [w], [x_opt, u_opt, tf_opt, ax_opt, ay_opt, dt_opt], 
    #                          ['w'], ['x_opt', 'u_opt', 'tf_opt', 'ax_opt', 'ay_opt', 'dt_opt'])
    # x_opt, u_opt, tf_opt, dt_opt, ax_opt, ay_opt = f_solution(res['x'])

    # f_sol = ca.Function("f_sol", [w], [x_opt, u_opt, dt_opt, tf_opt, ax_opt, ay_opt, ec_opt], 
    #                     ['w'], ['x_opt', 'u_opt', 'dt_opt', 'tf_opt', 'ax_opt', 'ay_opt', 'ec_opt'])
    # x_opt, u_opt, dt_opt = f_sol(res['x'])

    f_sol = ca.Function('f_sol', [w], [x_opt, u_opt, tf_opt, dt_opt, ax_opt, ay_opt, ec_opt],
                        ['w'], ['x_opt', 'u_opt', 'tf_opt', 'dt_opt', 'ax_opt', 'ay_opt', 'ec_opt'])

    x_opt, u_opt, tf_opt, dt_opt, ax_opt, ay_opt, ec_opt = f_sol(res['x'])

    x_opt = np.reshape(x_opt, (-1, 5))
    u_opt = np.reshape(u_opt, (-1, 4))
    t_opt = np.hstack((0.0, np.cumsum(dt_opt)))

    x_opt = pd.DataFrame(x_opt, columns = ['n', 'xi', 'v', 'beta', 'omega_z'])
    u_opt = pd.DataFrame(u_opt, columns = ['delta', 'f_drive', 'f_brake', 'gamma_y'])
    t_opt = pd.DataFrame(t_opt)

    x_opt.to_csv('./data/x_opt.csv', index = False)
    u_opt.to_csv('./data/u_opt.csv', index = False)
    t_opt.to_csv('./data/t_opt.csv', index = False)
    
    # '-'存疑
    return -x_opt.iloc[:-1,0], x_opt.iloc[:-1,2]