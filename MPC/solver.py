import numpy as np
import casadi as ca
import sys
sys.path.append('../')
from min_time import f_dynamic


N = 10
T = 0.2
pars = {}

# states variables
n = ca.SX.sym('n')
xi = ca.SX.sym('xi')
v = ca.SX.sym('v')
beta = ca.SX.sym('beta')
omega_z = ca.SX.sym('omega_z')
# kappa = ca.SX.sym('kappa')
states = ca.vertcat(*[n, xi, v, beta, omega_z])
num_states = states.shape[0]

# control variables
delta = ca.SX.sym('delta')
f_drive = ca.SX.sym('f_drive')
f_brake = ca.SX.sym('f_brake')
gamma_y = ca.SX.sym('gamma_y')
controls = ca.vertcat(*[delta, f_drive, f_brake, gamma_y])
num_controls = controls.shape[0]

# 滚动优化矩阵
X = ca.SX.sym('X', num_states, N + 1)
U = ca.SX.sym('U', num_controls, N)
P = ca.SX.sym('P', N + 2) # reference

# 状态转移函数
# f = ca.Function('f', [states, controls], [f_dot], ['states', 'controls'], ['f_dot'])
# f = ca.Function('f', [states, controls], [dx], ['states', 'controls'], ['dx'])
# f_dynamic = ca.Function('f_dynamic', [x, u, kappa], [dx, sf], ['x', 'u', 'kappa'], ['dx', 'sf'])


X[:, 0] = P[:, -1]
for i in range(N):
    rhs_temp = f(X[:, i], U[:, i])
    X[:, i + 1] = X[:, i] + T * rhs_temp

Q = np.array()
R = np.array()

J = 0
for i in range(N):
    J += ca.mtimes([(X[:, i] - P[:, i]).T, Q, (X[:, i] - P[:, i])]) + ca.mtimes([(U[:, i]).T, R, (U[:, i])])

# 一些约束
g = []