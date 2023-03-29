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

# BOUNDARY CONDITIONS
bcs = ('free-flux', 'free-flux')    # Boundary conditions: No flux

# MASS BALANCE
# Specify constant mass balance. Gamma is interpreted as the
# constant mass balance (m w.e./a) if balance is 'constant', and
# interpreted as the mass-balance gradient (m w.e./a/m) if
# balance is 'linear'.
balance = 'constant'                # 'constant' or 'linear'

# BED ELEVATION
b0 = 1200                           # Reference bed elevation (m)
zb = b0 - 0*xc                      # Flat bed elevation

zELA = 1550                         # Equilibrium line altitude. Not used
                                    # if balance is 'constant'

# INITIAL THICKNESS
h0 = 300*np.sin(np.pi*xc/50e3) + 600

# INTEGRATION
t0 = 0                              # Start time (seconds)
tend = 500*365*86400                # End time (seconds)
dt = 30*86400                       # Time step (seconds). This can be large
                                    # since we use implicit timestepping
tt = np.arange(t0, tend+dt, dt)     # Solution time array


## THINGS THAT CAN CHANGE
# mass_balance = 0.25/86400/365
# mass_balance = 0.5/(86400*365)      # Mass-balance (m w.e./a)
mass_balance = 1/86400/365

# FLOW-LAW COEFFICIENT
# Flow-law coefficient should be near 2.4e-24. Decreasing A
# below about 2.4e-25 may cause the glacier to extend out of the domain
# A = 4.8e-24
# A = 2.4e-24                         # Cuffey & Paterson (2010) recommended
A = 4.8e-25


## Run the SIA solver
H, U = solve_SIA(tt, xc, h0, zb, zELA=zELA,
        Gamma=mass_balance, bcs=bcs, b=balance, A=A)

## Plot the solution
fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.plot(xc/1e3, H[0, :], label='t = 0 a', color='steelblue')
ax1.plot(xc/1e3, H[-1, :], label='t = 500 a', color='goldenrod')
ax1.legend()
ax1.grid()
ax1.set_xlabel('x (km)')
ax1.set_ylabel('h (m)')
ax1.text(-0.075, 1.05, 'a', transform=ax1.transAxes, fontsize=12)
ax1.set_xlim([0, 50])
ax1.set_ylim([500, 1200])

ax2.plot(xc/1e3, U[-1, :]*365*86400, label='t = 500 a', color='goldenrod')
ax2.grid()
ax2.set_xlabel('x (km)')
ax2.set_ylabel('v (m/a)')
ax2.text(-0.075, 1.05, 'b', transform=ax2.transAxes, fontsize=12)
ax2.set_xlim([0, 50])
ax2.set_ylim([-50, 50])


plt.tight_layout()


rho = 910
tot_mass = rho*np.sum(H*dx, axis=1)
fig2, ax3 = plt.subplots()
ax3.plot(tt/365/86400, tot_mass/tot_mass[-1])
ax3.grid()
ax3.set_xlabel('Time (years)')
ax3.set_ylabel('Relative mass (-)')
ax3.text(-0.075, 1.05, 'a', transform=ax3.transAxes, fontsize=12)

plt.show()
fig1.savefig('constant_balance.png', dpi=600)
