from solver import *
import pandas as pd

data = pd.read_csv('../data/x_opt.csv')
x_opt = data.to_numpy()

n_ref       = x_opt[:, 0]
xi_ref      = x_opt[:, 1]
v_ref       = x_opt[:, 2]
beta_ref    = x_opt[:, 3]
omega_z_ref = x_opt[:, 4]
