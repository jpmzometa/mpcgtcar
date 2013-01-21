"""Main simulation file of MPC controller."""

from sys import path

import numpy as np
import matplotlib.pyplot as plt
import muaompc

import gtcar1sim
import exactqp
path.append('../setup')
import gtcar1
from nlmodel import IN_NUM, ST_NUM, CT_NUM, ST_V, CT_B

ROWS, COLS = (0, 1)
dt = gtcar1.dt
    
def sim_closed_loop_orig_stab(mpc, cvx=None):
    # stabilization of the origin
    samp = 1000
    u_t = np.zeros([IN_NUM, samp])
    x_t = np.zeros([ST_NUM, samp+1])
    c_t = np.zeros([CT_NUM, samp+1])
    # noise
    w_t = ((np.random.rand(ST_NUM, samp + 1) - 0.5) * 
           [[1e-1], [1e-1], [1e-3], [1e-2], [1e-3]])

    # controller setpoints
    x_sp = gtcar1.x_sp
    u_sp = gtcar1.u_sp
    c_sp = np.zeros(CT_NUM)
    c_sp[CT_B] = x_sp[ST_V]  # band velocity is equal to the state setpoint
    
    car = gtcar1sim.GTCarSim(dt)
    x_t[:, 0] = np.zeros(ST_NUM)
    print('Starting closed-loop simulation...')
    for k in range(samp):
        x = x_t[:,k] - x_sp

        if cvx is None:  # mpc approximated solution
            mpc.ctl.solve_problem(x)
            u_opt = mpc.ctl.u_opt
        else:  # cvxopt exact solution
            mpc.ctl.form_qp(x)
            cvx.solve_problem(mpc.ctl.qpx)
            u_opt = cvx.u_opt
            
        c_t[:,k] = c_sp
        u_t[:,k] = u_sp + u_opt[0:IN_NUM].flatten()
        x_t[:,k+1] = car.sim_zoh(x_t[:,k], u_t[:,k], c_t[:,k]) + w_t[:,k]
        
        if not (k % (samp/10)):
            print(k, ', x_t:', x_t[:,k])
            print('     x:', x.T)
            print('     u:', u_opt[0:IN_NUM].T)
            
    u_tp = np.concatenate([u_t, np.zeros([IN_NUM, 1])], COLS)
    print('Saving data to file')
    fname = 'closed_loop_sim_'
    if cvx is None:
        fname += 'muao'
    else:
        fname += 'cvx'
    np.save(fname, np.concatenate([x_t, u_tp]))   

def plot_sim(name):
    xu = np.load(name + '.npy')
    t = np.cumsum(dt * np.ones(xu.shape[COLS]))
    plt.subplot(221)
    plt.plot(t, xu[3,:])
    plt.subplot(222)
    plt.plot(t, xu[4,:])
    plt.subplot(223)
    plt.plot(t, xu[5,:])
    plt.subplot(224)
    plt.plot(t, xu[6,:])
    plt.show()
    
if __name__ == '__main__':
    use_cvx = False
    mpc = muaompc.ltidt.setup_mpc_problem('gtcar1')
    ctl = mpc.ctl
    ctl.conf.in_iter = 1
    ctl.conf.warmstart = 1
    if use_cvx:
        cvx = exactqp.ExactQPSolverInc(ctl.qpx)
        sim_closed_loop_orig_stab(mpc, cvx)
        plot_sim('closed_loop_sim_cvx')
    else:
        mpc.ctl.conf.in_iter = 10
        sim_closed_loop_orig_stab(mpc)
        plot_sim('closed_loop_sim_muao')
    