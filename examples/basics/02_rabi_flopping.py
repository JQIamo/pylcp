"""
@author: spe

This example covers Rabi flopping in a magnetic field for a single spin.  This
should be compared most closely to examples/hamiltonians/00_spin_in_magnetic_field.py
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
import pylcp
from pylcp.common import spherical2cart
plt.style.use('paper')

# %%
"""
First, define the problem:
"""
#H_0, mu_q = pylcp.hamiltonians.hyperfine_coupled(0, 1/2, gJ=0, gI=1, Ahfs=1)
gF = 1 # gF > 0 magnetic moment OPPOSITE F.  If the magnetic moment is
       # opposite the spin, then it will rotate counter clockwise when viewed
       # from the tip of the magnetic field vector
H_0, mu_q = pylcp.hamiltonians.singleF(5, gF=gF, muB=1)

# Construct operators for calculation of expectation values of spin and mu:
mu = spherical2cart(mu_q)
S = -mu/gF # Note that muB=1

hamiltonian = pylcp.hamiltonian()
hamiltonian.add_H_0_block('g', H_0)
hamiltonian.add_mu_q_block('g', mu_q)

magField = pylcp.magField(lambda R: np.array([0., 1., 0.]))
laserBeams = {}

# %%
"""
Now, create the OBE:
"""
obe = pylcp.obe(laserBeams, magField, hamiltonian, transform_into_re_im=False)
pop = np.zeros((H_0.shape[0],))
pop[-1] = 1

obe.set_initial_rho_from_populations(pop)
obe.evolve_density([0, np.pi/2], t_eval=np.linspace(0, np.pi/2, 51))

# Compute expectation values:
(t, rho) = obe.reshape_sol()
avS = np.zeros((3,)+ t.shape)
for jj in range(3):
    for ii in range(t.size):
        avS[jj, ii] = np.real(np.sum(S[jj]*rho[:, :, ii]))

fig, ax = plt.subplots(1, 1)
[ax.plot(t, avS[ii]) for ii in range(3)]

# %%
"""
Take the last value and propogate around z:
"""
obe.set_initial_rho(obe.sol.y[:-6, -1])
obe.magField = pylcp.magField(lambda R: np.array([0., 0., 1.]))
obe.evolve_density([0, 2*np.pi], t_eval=np.linspace(0, 2*np.pi, 51))
(t, rho) = obe.reshape_sol()

# Compute expectation values:
(t, rho) = obe.reshape_sol()
avS = np.zeros((3,)+ t.shape)
for jj in range(3):
    for ii in range(t.size):
        avS[jj, ii] = np.real(np.sum(S[jj]*rho[:, :, ii]))

fig, ax = plt.subplots(1, 1)
[ax.plot(t, avS[ii]) for ii in range(3)]
