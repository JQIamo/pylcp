"""
author: spe

This script shows that the rate equations can in fact integrate an F=0->F=1 atom
moving in a Zeeman slower.
"""
import numpy as np
import matplotlib.pyplot as plt
import pylcp
import cProfile, pstats, io
from scipy.integrate import solve_ivp
plt.style.use('paper')

# %%
"""
Define the problem:
"""
det = -2
beta = 3
laserBeams = [pylcp.laserBeam(kvec=[0., 0., +1], pol=+1, delta=det,  beta=beta)]

magField_zero = lambda R: np.array([0., 0., 1e-10])
magField_slower = lambda R: np.array([0., 0., R[2]**0.5])

Hg, Bgq = pylcp.hamiltonians.singleF(F=0, muB=1)
He, Beq = pylcp.hamiltonians.singleF(F=1, muB=1)
dijq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
hamiltonian = pylcp.hamiltonian(Hg, He, Bgq, Beq, dijq)

rateeq_zero = pylcp.rateeq.rateeq(laserBeams, magField_zero, hamiltonian)
rateeq_slower = pylcp.rateeq.rateeq(laserBeams, magField_zero, hamiltonian)

# %%
"""
Let's start with evolution of the atom in a single laser beam:
"""
mass_ratio=2.1e-2
pr = cProfile.Profile()

rateeq_zero.set_initial_position_and_velocity(np.array([0., 0., 100.]),
                                              np.array([0., 0., -4.]))
rateeq_zero.set_initial_pop_from_equilibrium()
pr.enable()
rateeq_zero.evolve_motion([0., 5e3], mass_ratio=mass_ratio,
                          freeze_axis=[True, True, False])
pr.disable()

def atom_motion_dt(t, y):
    return np.array([mass_ratio*beta/2/(1 + beta + 4*(det - y[0])**2), y[0]])

sol = solve_ivp(atom_motion_dt, [0, 5e3], [-4, 100])

fig, ax = plt.subplots(1, 2, num='motion', figsize=(6.5, 2.75))
ax[0].plot(rateeq_zero.sol.t, rateeq_zero.sol.y[-4, :])
ax[0].plot(sol.t, sol.y[0, :], '--')
ax[1].plot(rateeq_zero.sol.y[-1, :], rateeq_zero.sol.y[-4, :])
ax[1].plot(sol.y[1, :], sol.y[0, :], '--')
ax[0].set_ylabel('$v_z/(\Gamma/k)$')
ax[0].set_xlabel('$\Gamma t$')
ax[1].set_xlabel('$kz$')

fig, ax = plt.subplots(1, 1, num='Populations')
ax.plot(rateeq_zero.sol.t, rateeq_zero.sol.y[0:4, :].T)
ax.set_ylabel('Populations')
ax.set_xlabel('$\Gamma t$')

# %%
"""
Print profile results.
"""
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

# %%
"""
Now, let's do a Zeeman slower.
"""
