'''
Created on Jan 8, 2013

@author: chito
'''
from sympy import symbols
import numpy as np
from numpy import diag, eye

import nlmodel
from nlmodel import IN_NUM, ST_NUM, ST_V

# SET POINT: this has to be considered in the online implementation
x_sp = np.zeros(ST_NUM)  # all setpoints are zero, unless otherwise stated
x_sp[ST_V] = 1.  # m/s

xd = nlmodel.full()
param = nlmodel.param()
p6 = symbols('p6')
param[p6] = 0.  # epsilon not necessary for linearization
Ac_sym, Bc_sym, u_sp = nlmodel.compute_linear_system_matrices(xd, param, x_sp)
Ac = np.array(Ac_sym).astype('float64')
Bc = np.array(Bc_sym).astype('float64')
u_sp = np.array(u_sp).astype('float64')
#Bc[ST_V, IN_M] =/ 2.  # Take away the u0**2, the model is affine in inputs  
dt = 0.005
N = 10
Q = eye(ST_NUM) * 1e1
R = eye(IN_NUM) * 1e-3
P = 'auto'

u2_b = np.deg2rad(15)
u1_lb = 0.
u1_ub = 0.2
u_lb = [[u1_lb - u_sp[0]], [-u2_b - u_sp[1]]]
u_ub = [[u1_ub - u_sp[0]], [u2_b - u_sp[1]]]

if __name__ == '__main__':
    # SET POINT: this has to be considered in the online implementation
    x_sp = np.zeros(ST_NUM)  # all setpoints are zero, unless otherwise stated
    x_sp[ST_V] = 1.  # m/s
    
    xd = nlmodel.full()
    param = nlmodel.param()
    p6 = symbols('p6')
    param[p6] = 0.  # epsilon not necessary for linearization
    Ac_sym, Bc_sym, u_sp = nlmodel.compute_linear_system_matrices(xd, param, x_sp)
    #x0d, x1d = nlmodel.in_aff()
    #Aci, Bci = compute_linear_system_matrices(x0d, x1d, x0_sp, x1_sp)
    
    print('Aco:', Ac_sym)
    #print('Aci:', Aci)
    print('Bco:', Bc_sym)
    #print('Bci:', Bci)
    print('u_sp', u_sp)
    