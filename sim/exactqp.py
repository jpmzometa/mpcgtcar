import struct

import numpy as np
from cvxopt import matrix, solvers

class ExactQPSolverInc(object):
    
    def __init__(self, qpx):
        solvers.options['show_progress'] = False
        self.qp = solvers.qp
        self.P = matrix(qpx.HoL)
        I_u = np.eye(qpx.HOR_INPUTS)
        self.G = matrix(np.concatenate([-I_u, I_u]))
        self.h = matrix(np.concatenate([-qpx.u_lb, qpx.u_ub]))
        self.cvx2np = np.vectorize(lambda x: struct.unpack('d', x))
        self.u_opt = np.zeros([qpx.HOR_INPUTS, 1])
    
    def solve_problem(self, qpx):
        q = matrix(qpx.gxoL)
        sol = self.qp(self.P, q, G=self.G, h=self.h)
        self.u_opt = self.cvx2np(sol['x'])