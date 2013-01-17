from sys import path

from sympy.matrices import Matrix
from sympy import symbols
import numpy as np
from scipy.integrate import ode 

path.append('../setup')
import nlmodel
from nlmodel import X, U, ST_F, ST_L, ST_A, ST_V, ST_W, IN_M, IN_S

def f(t, x, u, Y):
    f_xu = Y.subs([(X[ST_F], x[ST_F]), (X[ST_L], x[ST_L]), (X[ST_A], x[ST_A]),
                   (X[ST_V], x[ST_V]), (X[ST_W], x[ST_W]), 
                   (U[IN_M], u[IN_M]), (U[IN_S], u[IN_S])])
    return np.array(f_xu).astype('float64')

def J(t, x, u, DY):
    J_xu = DY.subs([(X[ST_F], x[ST_F]), (X[ST_L], x[ST_L]), (X[ST_A], x[ST_A]),
                    (X[ST_V], x[ST_V]), (X[ST_W], x[ST_W]), 
                    (U[IN_M], u[IN_M]), (U[IN_S], u[IN_S])])
    return np.array(J_xu).astype('float64')

class GTCarSim(object):
    '''
    
    '''
   
    def __init__(self, dt):
        '''
        
        '''
        Y = nlmodel.full()
        param = nlmodel.param()
        
        self.dt = dt
        self.Y = Y.subs(param)
        self.DY = self.Y.jacobian(X)
        #self.DF = self.Y.jacobian(Matrix([X[ST_V], X[ST_W]]))
    
    def sim_zoh(self, x0, u):
        t0 = 0.
        r = ode(f).set_integrator('dopri5')#, atol=[1e-1, 1e-3], rtol=[1e-1, 1e-1])
        #r = ode(f, J).set_integrator('vode', with_jacobian=True)
        #r = ode(f).set_integrator('vode', with_jacobian=False)
        #r.set_jac_params(u, self.DY)
        r.set_f_params(u, self.Y)
        r.set_initial_value(x0, t0)
        r.integrate(self.dt)
        return r.y