"""
author: spe

This file simulates the motion of an atom in a quadrupole trap using the
OBEs.  This should be fun, as it can take into account Majoranna spin flips.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
import pylcp
plt.style.use('paper')

# %%
"""
Let's define the simple problem:
"""
H0, muq = pylcp.hamiltonians.singleF(1/2, gF=2, muB=1)

hamiltonian = pylcp.hamiltonian()
hamiltonian.add_H_0_block('g', H0)
hamiltonian.add_mu_q_block('g', muq)

magField = pylcp.magField(lambda R: np.array([0., 0., R[2]]))
laserBeams = {}

obe = pylcp.obe(laserBeams, magField, hamiltonian, include_mag_forces=True)
obe.set_initial_position(np.array([0., 0., 0.]))
theta = 3*np.pi/4
psi = np.array([np.cos(theta/2), np.sin(theta/2)])
rho = np.array([[psi[0]*psi[0], psi[0]*psi[1]], [psi[1]*psi[0], psi[1]*psi[1]]])

obe.set_initial_rho(rho.reshape(4,))
obe.evolve_motion([0., 5.], )

t, rho = obe.reshape_sol()
fig, ax = plt.subplots(1, 2, figsize=(6.25, 2.75))
[ax[0].plot(t, rho[ii, ii]) for ii in range(2)]
ax[0].plot(t, np.real(rho[0, 1]), '--')
ax[0].plot(t, np.imag(rho[0, 1]), '--')
ax[1].plot(t, obe.sol.y[-3:, :].T)

# %%
"""
Now, let's try the quadrupole trap with
"""
magField = pylcp.magField(lambda R: np.array([-0.5*R[0], -0.5*R[1], 1*R[2]]))

obe = pylcp.obe(laserBeams, magField, hamiltonian, include_mag_forces=True)

obe.set_initial_position(5*(2*np.random.rand(3)-1))
obe.set_initial_velocity(5*(2*np.random.rand(3)-1))
obe.set_initial_rho_from_populations(np.array([0., 1.]))

obe.evolve_motion([0, 30])

t, rho = obe.reshape_sol()
fig, ax = plt.subplots(1, 2, figsize=(6.25, 2.75))
[ax[0].plot(t, rho[ii, ii]) for ii in range(2)]
ax[1].plot(t, obe.sol.y[-3:, :].T)
