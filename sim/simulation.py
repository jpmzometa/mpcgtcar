from sys import path

import numpy as np
import matplotlib.pyplot as plt

import muaompc
import gtcar1sim
path.append('../setup')
import gtcar1
from nlmodel import IN_NUM, ST_NUM

ROWS, COLS = (0, 1)
dt = 0.004

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
    
def sim_closed_loop():
    samp = 1000
    u_t = np.zeros([IN_NUM, samp])
    x_t = np.zeros([ST_NUM, samp+1])
    # noise
    w_t = (np.random.rand(ST_NUM, samp + 1) * 
           [[0], [0], [0], [1e-2], [1e-3]])
    # controller
    mpc = muaompc.ltidt.setup_mpc_problem('gtcar1')
    ctl = mpc.ctl
    ctl.conf.in_iter = 10
    ctl.conf.warmstart = 1
    x_sp = gtcar1.x_sp
    # start sim
    car = gtcar1sim.GTCarSim(dt)
    x_t[:, 0] = np.zeros(ST_NUM)
    print('Starting closed-loop simulation...')
    for k in range(samp):
        x = x_t[:,k] - x_sp
        ctl.solve_problem(x)
        u_t[:,k] = (ctl.u_opt[0:IN_NUM] + np.array([[0], [-0.2464]])).flatten()
        x_t[:,k+1] = car.sim_zoh(x_t[:,k], u_t[:,k]) + w_t[:,k]
        if not (k % (samp/10)):
            print(k, ', x_t', x_t[:,k])
            print('     x:', x.T)
            print('     u:', ctl.u_opt[0:IN_NUM].T)
            
    u_tp = np.concatenate([u_t, np.zeros([IN_NUM, 1])], COLS)
    print('Saving data to file')
    np.save('closed_loop_sim', np.concatenate([x_t, u_tp]))   

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
    dt = 0.004
    car = gtcar1sim.GTCarSim(dt)
    x0 = np.array([0.1, 0])
    u = np.array([0.15, -0.2])
    y = car.sim_zoh(x0, u)
    print(y) 
    
if __name__ == '__main__':
    #main()
    #sim_open_loop()
    sim_closed_loop()
    plot_sim('closed_loop_sim')