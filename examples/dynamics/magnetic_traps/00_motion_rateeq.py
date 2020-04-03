"""
author: spe

This file simulates the motion of an atom in a quadrupole trap using the
rate equations.  The fun thing about the rate equations is that they do not
take into account the fact that the spin could not be along the local
magnetic-field axis.  This assumption comes from the fact that we are
neglecting coherences.
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

magField = pylcp.magField(lambda R: np.array([-0.5*R[0], -0.5*R[1], 1*R[2]]))
laserBeams = {}

rateeq = pylcp.rateeq(laserBeams, magField, hamiltonian,
                      include_mag_forces=True)

rateeq.set_initial_pop(np.array([0., 1.]))
rateeq.set_initial_position(np.array([5., 2., 4.]))
rateeq.evolve_motion([0, 1000])

# %%
fig, ax = plt.subplots(1, 2, figsize=(6.25, 2.75))
ax[0].plot(rateeq.sol.t, rateeq.sol.y[:2, :].T)
ax[1].plot(rateeq.sol.t, rateeq.sol.y[-3:, :].T)

# %%
"""
Now, let's try it with the OBE:
"""
callable(magField)
obe = pylcp.obe(laserBeams, magField, hamiltonian, include_mag_forces=True)

obe.set_initial_position(np.array([5., 0., 0.]))
obe.set_initial_rho_from_populations(np.array([0., 1.]))

obe.evolve_motion([0, 30])

# %%
t, rho = obe.reshape_sol()
fig, ax = plt.subplots(1, 2, figsize=(6.25, 2.75))
[ax[0].plot(t, rho[ii, ii]) for ii in range(2)]
ax[1].plot(t, obe.sol.y[-3:, :].T)
