import numpy as np
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph

def result_plots(width_veh: float,
                 refline: np.ndarray,
                 bound_r: np.ndarray,
                 bound_l: np.ndarray,
                 trajectory: np.ndarray) -> None:
    
    # 计算车辆边界点
    normvec__opt = tph.calc_normal_vectors.\
            calc_normal_vectors(trajectory[:, 3])

    veh_bound_r = trajectory[:, 1:3] + normvec__opt * width_veh / 2
    veh_bound_l = trajectory[:, 1:3] - normvec__opt * width_veh / 2
    
    plt.figure()
    plt.plot(refline[:, 0], refline[:, 1], "k--", linewidth=0.7)
    plt.plot(veh_bound_r[:, 0], veh_bound_r[:, 1], "c", linewidth=0.5)
    plt.plot(veh_bound_l[:, 0], veh_bound_l[:, 1], "c", linewidth=0.5)
    plt.plot(bound_r[:, 0], bound_r[:, 1], "k-", linewidth=0.7)
    plt.plot(bound_l[:, 0], bound_l[:, 1], "k-", linewidth=0.7)
    plt.plot(trajectory[:, 1], trajectory[:, 2], "r-", linewidth=0.7)
    
    # 作曲率图
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 4])
    plt.grid()
    plt.xlabel("distance in m")
    plt.ylabel("curvature in rad/m")
#     plt.show()
    plt.savefig("./fig/result_plot.png")
    