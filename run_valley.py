import numpy as np
from matplotlib import pyplot as plt

from SIA import solve_SIA


## THINGS THAT SHOULDN'T CHANGE

# DOMAIN
L = 50e3                            # Domain length (m)
N = 100                             # Number of grid centre points
dx = L/N                            # Grid spacing (m)
xc = np.arange(dx/2, L+dx/2, dx)    # Cell centre coordinates
xe = np.arange(0, L+dx, dx)         # Cell edge coordinates

xc = np.arange(dx/2, L+dx/2, dx)    # Cell centre coordinates
xe = np.arange(0, L+dx, dx)         # Cell edge coordinates

# BOUNDARY CONDITIONS
bcs = ('no-flux', 'no-flux')        # Boundary conditions: No flux

# MASS BALANCE
# Specify constant mass balance. Gamma is interpreted as the
# constant mass balance (m w.e./a) if balance is 'constant', and
# interpreted as the mass-balance gradient (m w.e./a/m) is
# balance is 'linear'.
balance = 'linear'                  # 'constant' or 'linear'

# BED ELEVATION
b0 = 2000                           # Reference bed elevation (m)
zELA = 1600                         # Equilibrium line altitude. Not used
                                    # if balance is 'constant'
zb = b0 - 1/20*xc                   # Flat bed elevation


# INITIAL THICKNESS
h0 = 400*np.ones(xc.shape)
h0[xc>30e3] = 0

# INTEGRATION parameters
t0 = 0                              # Start time (seconds)
tend = 500*365*86400                # End time (seconds)
dt = 30*86400                       # Time step (seconds). This can be large
                                    # since we use implicit timestepping
tt = np.arange(t0, tend+dt, dt)     # Solution time array

## THINGS THAT CAN CHANGE

# SLIDING
# Basal sliding velocity should be between 0 and ~200 m/year,
# or 0 and 200/365/86400 m/s. Two examples are given below:

# No sliding
# ub = np.zeros(xc.shape)

# Spatially uniform sliding
ub = 50*np.ones(xc.shape)/365/86400

# Sliding below a specified x-position
# ub = np.zeros(xc.shape); ub[40:] = 100/365/86400


# MASS BALANCE
# Mass balance gradient should be close to 0.007 m w.e./m/year,
# or 0.007/365/86400
# Gamma = 0.010/365/86400
Gamma = 0.007/(365*86400)           # Mass-balance gradient (m w.e./s/m)
# Gamma = 0.005/365/86400

# FLOW-LAW COEFFICIENT
# Flow-law coefficient should be near 2.4e-24.
# A = 2.4e-23
A = 2.4e-24                         # Cuffey & Paterson (2010) recommended
# A = 4.8e-25

## Run the SIA solver
H, U = solve_SIA(tt, xc, h0, zb, zELA=zELA,
    Gamma=Gamma, bcs=bcs, b=balance, ub=ub, A=A)

## Plot the solution
fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.plot(xc/1e3, zb, color='k', label='Bed')
ax1.plot(xc/1e3, zb + H[0, :], label='t = 0 a', color='steelblue')
ax1.plot(xc/1e3, zb + H[-1, :], label='t = 500 a', color='goldenrod')
ax1.legend()
ax1.grid()
ax1.set_xlabel('x (km)')
ax1.set_ylabel('h (m)')
ax1.text(-0.075, 1.05, 'a', transform=ax1.transAxes, fontsize=12)
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 2500])

ax2.plot(xc/1e3, U[-1, :]*365*86400, label='t = 500 a', color='goldenrod')
ax2.grid()
ax2.set_xlabel('x (km)')
ax2.set_ylabel('v (m/a)')
ax2.text(-0.075, 1.05, 'b', transform=ax2.transAxes, fontsize=12)
ax2.set_xlim([0, 50])
ax2.set_ylim([0, 300])

plt.tight_layout()


fig2, ax3 = plt.subplots()
rho = 910
tot_mass = rho*np.nansum(H*dx, axis=1)
ax3.plot(tt/365/86400, tot_mass/tot_mass[-1])
ax3.grid()
ax3.set_xlabel('Time (years)')
ax3.set_ylabel('Relative mass (-)')
ax3.text(-0.075, 1.05, 'a', transform=ax3.transAxes, fontsize=12)


plt.show()

fig1.savefig('valley_ub.png', dpi=600)
