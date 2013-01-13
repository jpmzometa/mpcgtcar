"""Nonlinear models of gt-car-1, based on [1]. 

[1] Modellierung und Identifikation eines autonomen Fahrzeugs, 
Studienarbeit, Andy Schröder."""

from sympy import symbols, cos, sin, solve
from sympy.matrices import Matrix
# INPUTS: M=moment, S=steering
IN_M, IN_S, IN_NUM = (0, 1, 2)
# STATES: X=absolute position in x (not relative to the band), Y=in y, 
# A=angle relative to the band (global coordinates), 
# V=car speed in car coordinates, W=car angular speed in global coordinates
ST_X, ST_Y, ST_A, ST_V, ST_W, ST_NUM = (0, 1, 2, 3, 4, 5) 
# inputs
U = Matrix(symbols('u:' + str(IN_NUM)))
# states
X = Matrix(symbols('x:' + str(ST_NUM)))

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
    x1d = X[ST_V] * cos(X[ST_A])  # velocity in x direction
    x2d = X[ST_V] * sin(X[ST_A])  # velocity in y direction
    x3d = X[ST_W]  # the angular velocity
    x4d, x5d = orig()
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
    x_sp = {X[ST_X]:x_sp[ST_X], X[ST_Y]:x_sp[ST_Y], X[ST_A]:x_sp[ST_A],
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
    return (Ac, Bc)


def _in_aff():
    """TEST: Model in [1] but affine in the inputs."""
    # Observation dictates that it is the same to linearize the original model
    # and simply divide by two the corresponding term of B matrix
    # i.e. the one that relates u0 to x0d
    # parameters
    p1a = -5.3  # taken such that at setpoint 1 m/s is similar to orig. model
    p2 = -1.555
    p3 = -1.2697
    p4 = 1.2282
    p5 = 0.2464
    p6 = -20.0
    p7 = 0.  # epsilon = 0.01, not used for setpoint != 0 
    
    # inputs
    u0, u1 = symbols('u:2')
    # states
    x0, x1 = symbols('x:2')
    # differential equation
    x0d = p1a*p2*(u0) + p2*x0 + p3*(x0**2)*(x1**2)
    x1d = p4*(u1 + p5) + p6 * (x1/(x0 + p7))
    # set-point
    return (x0d, x1d)    