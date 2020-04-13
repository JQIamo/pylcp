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
<<<<<<< Updated upstream
rateeq.set_initial_position(np.array([5., 2., 4.]))
rateeq.evolve_motion([0, 1000])

# %%
=======
rateeq.set_initial_position(np.array([5., 2., 1.]))
rateeq.evolve_motion([0, 2000], t_eval=np.linspace(0, 2000, 101), method='LSODA')

# %%
"""
Plot it up.
"""
>>>>>>> Stashed changes
fig, ax = plt.subplots(1, 2, figsize=(6.25, 2.75))
ax[0].plot(rateeq.sol.t, rateeq.sol.y[:2, :].T)
ax[1].plot(rateeq.sol.t, rateeq.sol.y[-3:, :].T)

# %%
"""
<<<<<<< Updated upstream
Now, let's try it with the OBE:
"""
callable(magField)
=======
One of the dead giveaways that we are in a magnetic quarupole trap is that the
frequency of oscillation should be dependent on the initial position.
"""
fig, ax = plt.subplots(1, 1)
for z0 in np.arange(2., 21., 4.):
    rateeq.set_initial_position(np.array([0., 0., z0]))
    rateeq.evolve_motion([0, 2000], t_eval=np.linspace(0, 2000, 101), method='LSODA')
    ax.plot(rateeq.sol.t, rateeq.sol.y[-1])

ax.set_xlabel('$t\'$')
ax.set_ylabel('$z\'$')
fig.savefig('oscillation_vs_height.pdf')

# And for the same height, the frequency should be twice that in the z direction:
fig, ax = plt.subplots(1, 1)
for jj in range(3):
    r0 = np.zeros((3,))
    r0[jj] = 5.
    rateeq.set_initial_position(r0)
    rateeq.evolve_motion([0, 2000], t_eval=np.linspace(0, 2000, 101), method='LSODA')
    ax.plot(rateeq.sol.t, rateeq.sol.y[-3+jj])

ax.set_xlabel('$t\'$')
ax.set_ylabel('$z\'$')
fig.savefig('oscillation_vs_axis.pdf')
# %%
"""
Now, let's try it with the OBE:
"""
>>>>>>> Stashed changes
obe = pylcp.obe(laserBeams, magField, hamiltonian, include_mag_forces=True)

obe.set_initial_position(np.array([5., 0., 0.]))
obe.set_initial_rho_from_populations(np.array([0., 1.]))

obe.evolve_motion([0, 30])

# %%
t, rho = obe.reshape_sol()
fig, ax = plt.subplots(1, 2, figsize=(6.25, 2.75))
[ax[0].plot(t, rho[ii, ii]) for ii in range(2)]
ax[1].plot(t, obe.sol.y[-3:, :].T)
