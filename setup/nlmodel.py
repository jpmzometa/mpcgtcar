"""Nonlinear models of gt-car-1, based on [1]. 

[1] Modellierung und Identifikation eines autonomen Fahrzeugs, 
Studienarbeit, Andy Schröder."""

from sympy import symbols, cos, sin, solve
from sympy.matrices import Matrix
# IN_ = inputs
# M = moment applied to the driving motor (back)  
# S = input applied to the steering servomotor 
IN_M, IN_S, IN_NUM = (0, 1, 2)
# ST_ = states
# F = car forward position relative to band origin (global coordinates)
# L = lateral car position relative to band origin (global coordinates)
# A = angle relative to the band forward axis (global coordinates), 
# V = car forward speed in car local coordinates, 
# W = car angular speed in global/local coordinates
ST_F, ST_L, ST_A, ST_V, ST_W, ST_NUM = (0, 1, 2, 3, 4, 5)
# CT_ = constant
# B = band speed
CT_B, CT_NUM = (0, 1) 
# inputs
U = Matrix(symbols('u:' + str(IN_NUM)))
# states
X = Matrix(symbols('x:' + str(ST_NUM)))
# constants
C = Matrix(symbols('c:' + str(CT_NUM)))

def param():
    """The parameters for the original model in [1]"""
    p0, p1, p2, p3, p4, p5, p6 = symbols('p:7')
    p = {p0:44.3837, p1:-1.555, p2:-1.2697, 
         p3:1.2282, p4:0.2464, p5:-20.0, p6:0.01}
    return p;

def orig():
    """The original model in [1]"""
    #parameters
    p0, p1, p2, p3, p4, p5, p6 = symbols('p:7')
    # differential equation
    x4d = p0*(U[IN_M]**2) + p1*X[ST_V] + p2*(X[ST_V]**2)*(X[ST_W]**2)
    x5d = p3*(U[IN_S] + p4) + p5 * (X[ST_W]/(X[ST_V] + p6))
    return (x4d, x5d)

def full():
    """The original model in [1], expanded to consider the position 
    of the car in the band"""
    Vb = symbols('Vb')  # band speed
    x1d = X[ST_V] * cos(X[ST_A]) - C[CT_B]  # velocity in forward direction
    x2d = X[ST_V] * sin(X[ST_A])  # velocity in lateral direction
    x3d = X[ST_W]  # the angular velocity
    x4d, x5d = orig()  # cars forward and angular speed (local coordinates)
    return Matrix([x1d, x2d, x3d, x4d, x5d])

def compute_linear_system_matrices(xd, param, x_sp):
    """
    inputs:
    xd: the state space equation dx/dt = f(x) in symbolic form
    param: parameters to substitute in xd
    x_sp: setpoint around to which linearize
    """
 
    Y = xd.subs(param)
    DYx = Y.jacobian(X)
    DYu = Y.jacobian(U)
    # state setpoint
    x_sp = {X[ST_F]:x_sp[ST_F], X[ST_L]:x_sp[ST_L], X[ST_A]:x_sp[ST_A],
            X[ST_V]:x_sp[ST_V], X[ST_W]:x_sp[ST_W]}
    
    Y_sp = Y.subs(x_sp)
    # only use the equations that explicitily contain U
    u_sp = solve([Y_sp[ST_V], Y_sp[ST_W]])
    
    if isinstance(u_sp, list):
        for sol in u_sp:  # the term u[IN_M]**2 (if any) has two solutions 
            if sol[U[IN_M]] > 0:  # take the positive one
                u_sp = sol

    Ac = DYx.subs(x_sp)
    Bc = DYu.subs(u_sp)
    u_spl = []
    for k in range(IN_NUM):
        u_spl.append(u_sp[U[k]])  # make a list from the dictionary u_sp
        
    return (Ac, Bc, u_spl)
