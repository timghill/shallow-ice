
import numpy as np
from matplotlib import pyplot as plt

from SIA import solve_SIA

## First set geometry, parameters, and integration time

# DOMAIN
L = 100e3                           # Domain length (m)
N = 100                             # Number of grid centre points
dx = L/N

xc = np.arange(dx/2, L+dx/2, dx)    # Cell centre coordinates
xe = np.arange(0, L+dx, dx)         # Cell edge coordinates

# MASS BALANCE
# Specify constant mass balance. Gamma is interpreted as the
# constant mass balance (m w.e./a) if balance is 'constant', and
# interpreted as the mass-balance gradient (m w.e./a/m) is
# balance is 'linear'.
Gamma = 0.007/(365*86400)           # Mass-balance (m w.e./a)
balance = 'linear'                  # 'constant' or 'linear'

b0 = 1200                           # Reference bed elevation (m)
zELA = b0 + 450                     # Equilibrium line altitude. Not used
                                    # if balance is 'constant'
zb = b0 - 0*xc                      # Flat bed elevation

# INITIAL THICKNESS
h0 = 500*np.sin(np.pi*xc/L) + 0

# INTEGRATION
t0 = 0                              # Start time (seconds)
tend = 500*365*86400                # End time (seconds)
dt = 30*86400                       # Time step (seconds). This can be large
                                    # since we use implicit timestepping
                                    
tt = np.arange(t0, tend+dt, dt)     # Solution time array

bcs = ('free-flux', 'free-flux')    # Boundary conditions: No flux

## Run the SIA solver
H1 = solve_SIA(tt, xc, h0, zb, zELA=zELA,
    Gamma=Gamma, bcs=bcs, b=balance)

## Plot the solution
fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.plot(xc/1e3, H1[0, :], label='t = 0 a')
ax1.plot(xc/1e3, H1[-1, :], label='t = 500 a')
ax1.legend()
ax1.grid()
ax1.set_xlabel('x (km)')
ax1.set_ylabel('h (m)')
ax1.text(-0.075, 1.05, 'a', transform=ax1.transAxes, fontsize=14)

rho = 910
tot_mass = rho*np.sum(H1*dx, axis=1)
ax2.plot(tt/365/86400, tot_mass/tot_mass[-1])
ax2.grid()
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Relative mass (-)')
ax1.text(-0.075, 1.05, 'b', transform=ax2.transAxes, fontsize=14)
ax1.set_title('Gamma = %.3e (%s)' % (Gamma, balance))

plt.tight_layout()

plt.show()
fig1.savefig('flat_bed.png', dpi=600)
