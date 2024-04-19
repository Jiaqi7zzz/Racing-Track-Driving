import casadi as ca

'''
Use of Ipopt under CasADi Environment
'''
x1 = ca.MX.sym('x1')
x2 = ca.MX.sym('x2')
f = x1**2 + x2**2
g = [x1 + x2 - 10]

nlp = {'x': ca.vertcat(x1, x2), 'f': f, 'g': ca.vertcat(*g)}
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

res = solver(x0=[0, 0], lbx=[-ca.inf, -ca.inf], ubx=[ca.inf, ca.inf], lbg=[0], ubg=[0])

x_opt = res['x']
print('x_opt:', x_opt)