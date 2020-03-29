"""
This example attempts to compute the temperature of a standard, six-beam MOT.
"""
import numpy as np
import matplotlib.pyplot as plt
import pylcp
from pylcp.tools import (quadrupoleField3D, standard_six_beam_MOT)
import cProfile, pstats, io
plt.style.use('paper')

# %%
"""
Define the problem:
"""
# Define the laser and field properties:
det = -3.0
beta = 2.0
alpha = 0.0005

# Define the laser beams:
laserBeams = standard_six_beam_MOT(beta, det)

magField = lambda R: quadrupoleField3D(R, alpha)

# Define the atomic Hamiltonian:
Hg, mugq = pylcp.hamiltonians.singleF(F=2, gF=0, muB=1)
He, mueq = pylcp.hamiltonians.singleF(F=3, gF=1/3., muB=1)

dijq = pylcp.hamiltonians.dqij_two_bare_hyperfine(2, 3)

hamiltonian = pylcp.hamiltonian(Hg, He, mugq, mueq, dijq)

rateeq = pylcp.rateeq(laserBeams, magField, hamiltonian)
obe = pylcp.obe(laserBeams, magField, hamiltonian)

# %%
x = np.linspace(-10., 10., 101)/alpha
rateeq.generate_force_profile(
    [x, np.zeros(x.shape), np.zeros(x.shape)],
    np.zeros((3,)+x.shape),
    name='Fx')
rateeq.generate_force_profile(
    [np.zeros(x.shape), x, np.zeros(x.shape)],
    np.zeros((3,)+x.shape),
    name='Fy')
rateeq.generate_force_profile(
    [np.zeros(x.shape), np.zeros(x.shape), x],
    np.zeros((3,)+x.shape),
    name='Fz')

laserBeams.beam_vector[0].return_parameters(np.array([0., 0., 0.]), 0.)
plt.figure()
plt.plot(x*alpha, rateeq.profile['Fx'].F[0])
plt.plot(x*alpha, rateeq.profile['Fy'].F[1])
plt.plot(x*alpha, rateeq.profile['Fz'].F[2])

# %%
"""
Let's now evolve the motion of a single particle using the rate equations.
"""
r = np.zeros((3,))
v = np.zeros((3,))
t_span = [0, 500]

rateeq.set_initial_position_and_velocity(r, v)
rateeq.set_initial_pop_from_equilibrium()
pr = cProfile.Profile()
pr.enable()
rateeq.evolve_motion(t_span, recoil_velocity=0.03,
                     random_recoil=True, max_scatter_probability=0.5)
pr.disable()

print(len(rateeq.sol.t_events[0]), len(rateeq.sol.t))

fig, ax = plt.subplots(1, 2, figsize=(6.25, 2.75))
ax[0].plot(rateeq.sol.t, rateeq.sol.y[-3:].T)
ax[1].plot(rateeq.sol.t, rateeq.sol.y[-6:-3].T)

# %%
"""
Print out the profile results:
"""
s = io.StringIO()
sortby = 'tottime'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

# %%
"""
Let's try the OBE:
"""
obe.set_initial_position_and_velocity(r, v)
obe.set_initial_rho_from_rateeq()
pr = cProfile.Profile()
pr.enable()
obe.evolve_motion(t_span, random_recoil=True, max_scatter_probability=0.5)
pr.disable()

print(len(rateeq.sol.t_events[0]), len(rateeq.sol.t))

t, rho = obe.reshape_sol()

fig, ax = plt.subplots(1, 2, figsize=(6.25, 2.75))
ax[0].plot(obe.sol.t, obe.sol.y[-3:].T)
ax[1].plot(obe.sol.t, obe.sol.y[-6:-3].T)

# %%
"""
Print out the profile results:
"""
s = io.StringIO()
sortby = 'cumtime'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
