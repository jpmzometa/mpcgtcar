from sys import path

import numpy as np
import matplotlib.pyplot as plt
import muaompc

import gtcar1sim
import exactqp
path.append('../setup')
import gtcar1
from nlmodel import IN_NUM, ST_NUM, ST_F

ROWS, COLS = (0, 1)
dt = gtcar1.dt

def sim_open_loop():
    u_t0 = np.zeros([1, 200])
    u_t1 = np.ones([1, 600])
    u_t01 = np.concatenate([u_t0, u_t1, u_t0], COLS)
    u_t = np.concatenate([u_t01*0.15, u_t01*-0.2464])
    x_t = np.zeros([2, u_t.shape[COLS]+1])
    car = gtcar1sim.GTCarSim(dt)
    x_t[:, 0] = np.array([0.0, 0])
    for k, uk in enumerate(u_t.T):
        x_t[:,k+1] = car.sim_zoh(x_t[:,k], uk)
    u_tp = np.concatenate([u_t, np.zeros([2, 1])], COLS)
    print('saving data to file')
    np.save('open_loop_sim', np.concatenate([x_t, u_tp]))
    
def sim_closed_loop(mpc, cvx=None):
    samp = 400
    u_t = np.zeros([IN_NUM, samp])
    x_t = np.zeros([ST_NUM, samp+1])
    # noise
    w_t = ((np.random.rand(ST_NUM, samp + 1) - 0.5) * 
           [[1e-1], [1e-1], [1e-3], [1e-2], [1e-3]])
    # The car position relative to the band in the forward direction
    x_b_ST_F = (np.random.rand(1, samp + 1) - 0.5) * 1e-1
    # controller
    x_sp = gtcar1.x_sp
    u_sp = gtcar1.u_sp
    # TODO:
    # 1. The reference trajectory x_ref[ST_F] should be the integration of
    #    the reference velocity x_sp[ST_V] (in theory). However, better
    #    will be to the x_ref[ST_F] = x[ST_F] + (the X pos. relative to band) 
    # start sim
    car = gtcar1sim.GTCarSim(dt)
    x_t[:, 0] = np.zeros(ST_NUM)
    print('Starting closed-loop simulation...')
    for k in range(samp):
        x_sp[ST_F] = x_t[ST_F, k]
        x = x_t[:,k] - x_sp
        x[ST_F] += x_b_ST_F[0, k]  # simulate the relative forward position
        if cvx is None:  # mpc approximated solution
            mpc.ctl.solve_problem(x)
            u_opt = mpc.ctl.u_opt
        else:  # cvxopt exact solution
            mpc.ctl.form_qp(x)
            cvx.solve_problem(mpc.ctl.qpx)
            u_opt = cvx.u_opt
            
        u_t[:,k] = u_sp + u_opt[0:IN_NUM].flatten()
        x_t[:,k+1] = car.sim_zoh(x_t[:,k], u_t[:,k]) + w_t[:,k]
        
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

def main():
    car = gtcar1sim.GTCarSim(dt)
    x0 = np.array([0.1, 0])
    u = np.array([0.15, -0.2])
    y = car.sim_zoh(x0, u)
    print(y) 
    
if __name__ == '__main__':
    #main()
    #sim_open_loop()
    
    mpc = muaompc.ltidt.setup_mpc_problem('gtcar1')
    ctl = mpc.ctl
    ctl.conf.in_iter = 10
    ctl.conf.warmstart = 1
    cvx = exactqp.ExactQPSolverInc(ctl.qpx)
    sim_closed_loop(mpc, cvx)
    plot_sim('closed_loop_sim_cvx')
    
    