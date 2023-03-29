"""
One-dimensional shallow ice approximation (SIA) model.

Internal function rhs_1d computes the time-derivative of
surface elevation using first-order finite volume methods.

Function solve_SIA integrates the SIA model forward in time
and returns the ice thickness.
"""
import time
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

# CONSTANTS
g = 9.81                    # Gravity; m/s2
rho = 910                   # Ice density: kg/m3
n = 3                       # Ice flow exponent: -

# PARAMETERS
A = 2.4e-24                 # Ice flow coefficient (Pa-3.s-1)
Gamma = 0.007/(365*86400)   # Mass balance gradient (s-1)
zELA = 1400


def rhs_1d(t, h, zb, dx, Gamma=Gamma, zELA=zELA,
    bcs=('no-flux', 'no-flux'), b='linear', A=A, ub=None):
    """
    Calculate right hand side (dh/dt) of one-dimensional ice flow model.

    Inputs:
    --------
    t : time (seconds). Not used; included for compatibility with time stepping
    h : (N,) array. Ice thickness
    zb : (N,) array. Bed elevation
    dx : float. Grid spacing

    Options:
    --------
    Gamma = 0.007/(365*86400). Mass-balance gradient or constant mass balance
    if b='constant'.

    zELA = 1400. Equilibrium line altitude.

    bcs = ('no-flux', 'no-flux'). Left and right boundary conditions, one of
    'no-flux' or 'free-flux'.

    b = 'linear'. Specify 'linear' or 'constant' mass-balance parameter Gamma.

    A = 2.4e-24. Ice-flow constant.

    ub = 0. Sliding velocity. Float or (N,) array

    Returns:
    --------
    hprime : (N,) array. Time derivative of ice thickness.
    hprime = -div(q) + bdot
    """
    n_center = len(h)
    n_edge = n_center + 1
    
    zs = zb + h
    # dzdx defined on edges
    dzdx_center = (zs[1:] - zs[:-1])/dx

    # k defined on centers
    k_center = -(2*A)*((rho*g)**n)*(h**(n+2))/(n+2)
    q_edge = np.zeros(n_edge)
    q_edge[1:-1] = (k_center[1:] + k_center[:-1])/2 * dzdx_center**n

    # Compute sliding contribution
    if ub is None:
        ub = np.zeros(n_center)
    ub_edge = np.zeros(n_edge)
    ub_edge[1:-1] = ub[:-1]
    h_edge = np.zeros(n_edge)
    h_edge[1:-1] = h[:-1]
    ubh = ub_edge * h_edge

    # Calculate mass balance
    if b=='linear':
        bdot = Gamma*(zs - zELA)
    else:
        bdot = Gamma*np.ones(n_center)

    if bcs[0]=='no-flux':
        # No flux boundary conditions
        q_edge[0] = 0 
        ubh[0] = 0      # No sliding on boundary
    elif bcs[0]=='free-flux':
        # q_edge[0] = k_center[0]*(dzdx_center[0])**n
        q_edge[0] = q_edge[1]
        q_edge[0] += bdot[0]*dx*np.sign(q_edge[0]) + ub[0]*h[0]
        ubh[0] = ubh[1] # Extrapolate sliding to the first cell

    if bcs[1]=='no-flux':
        q_edge[-1] = 0
        ubh[-1] = 0     # No sliding on no-flux boundary
    elif bcs[1]=='free-flux':
        # q_edge[-1] = k_center[-1]*dzdx_center[-1]**n
        q_edge[-1] = q_edge[-2]
        q_edge[-1] += bdot[1]*dx*np.sign(q_edge[-1]) + ub[-1]*h[-1]
        ubh[-1] = ub[-1]*h[-1]  # Sliding on last cell

    q_edge = q_edge + ubh

    # Time derivative is flux divergence + mass balance
    hprime = -(q_edge[1:] - q_edge[:-1])/dx + bdot
    return hprime

def solve_SIA(tt, xc, h, zb, Gamma=Gamma, zELA=zELA, method='odeRK', **kwargs):
    """
    Integrate SIA model forward in time.
    
    Inputs:
    --------
    t : (M,) array. Time increments to compute ice thickness.

    xc : (N,) array. Cell-center coordinates.

    h : (N,) array. Ice thickness at cell centers.

    zb : (N,) array. Bed elevation at cell centers.

    Options:
    --------
    Gamma = 0.007/(365*86400). Mass-balance gradient or constant mass balance
    if b='constant'.

    zELA = 1400. Equilibrium line altitude.

    method = 'odeRK'. Time-stepping method. One of:
        - 'odeRK': Explicit four-step RK method
        - 'BE': Implicit first-order backwards Euler
        - 'CN': Implicit second-order Crank-Nicolson
    
    kwargs : passed to rhs_1d.

    Returns:
    --------
    H : (M, N) array. Ice thickness.
    """
    tstart = time.time()
    t = tt[0]
    tend = tt[-1]
    dt = tt[1] - tt[0]

    nsteps = int(tend/dt + 1)
    dx = xc[1] - xc[0]
    H = np.zeros((nsteps, len(xc)))
    H[0, :] = h
    i = 1

    # Time stepping parameters
    c2 = 1/2
    c3 = 1/2
    c4 = 1

    b1 = 1/6
    b2 = 1/3
    b3 = 1/3
    b4 = 1/6

    a21 = 0.5
    a31 = 0
    a32 = 0.5

    rhs_fun = lambda t, y: rhs_1d(t, y, zb, dx, Gamma=Gamma, zELA=zELA,**kwargs)
    while t<tend:
        h_old = h
        if method=='odeRK':
            k1 = rhs_fun(t, h)
            k2 = rhs_fun(t + c2*dt, h + dt*a21*k1)
            k3 = rhs_fun(t + c3*dt, h + dt*a31*k1 + dt*a32*k2)
            k4 = rhs_fun(t + c4*dt, h + dt*k3)

            dhdt = b1*k1 + b2*k2 + b3*k3 + b4*k4
            subset = h + dhdt<0
            dhdt[subset] = -h[subset]/dt
            h_new = h + dt*dhdt

        elif method=='BE':
            # Implicit method
            # Define function g which we will find the root of
            # h_new = scipy.optimize.brentq(g, h, maxiter=100)
            g = lambda z: z - h - dt*rhs_fun(t, z)
            h_new = scipy.optimize.newton(g, h, tol=1e-6, maxiter=50)
            # dhdt = rhs_fun(t, h_new)
            h_new[h_new<0] = 0

        elif method=='CN':
            g = lambda z: z - h - 0.5*dt*(rhs_fun(t, z) + rhs_fun(t, h))
            h_new = scipy.optimize.newton(g, h, tol=1e-6, maxiter=50)
            h_new[h_new<0] = 0


        elif method=='CN':
            pass
        h = h_new
        H[i, :] = h
        i+=1
        t+=dt

    dhdt = (h_new - h_old)/dt
    print('Maximum dh/dt (m/year):')
    print(np.max(np.abs(dhdt))*86400*365)
    tend = time.time()
    dtime = tend - tstart
    print('Elapsed time: ', dtime)
    return H
