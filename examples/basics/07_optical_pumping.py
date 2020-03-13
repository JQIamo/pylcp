"""
author:spe

This little script tests the optical pumping from the optical Bloch equations
and rate equations.  It reproduces Fig. 5 of Ungar, P. J., Weiss, D. S., Riis,
E., & Chu, S. (1989). Optical molasses and multilevel atoms: theory. Journal of
the Optical Society of America B, 6(11), 2058.
http://doi.org/10.1364/JOSAB.6.002058

There seems to be at least a factor of 2 pi missing in the above paper.

Note that agreement will only occur with the rate equations in the case of
a single laser beam.  This is because the rate equations assume that the
lasers are incoherent (their electric fields do not add to give twice the
amplitude) whereas the optical bloch equations do.
"""
import numpy as np
import matplotlib.pyplot as plt
import pylcp
import time
plt.style.use('paper')

# %%
"""
Define the problem to start.  Note that one can put the detuning on the laser
or put the detuning on the Hamiltonian.  The latter is faster.
"""
# First the laser beams:
laserBeams = []
laserBeams.append(pylcp.laserBeam(np.array([1., 0., 0.]),
                                  pol=np.array([0., 0., 1.]),
                                  delta=-2.73, beta=4*0.16))
"""laserBeams.append(pylcp.laserBeam(np.array([-1., 0., 0.]),
                                  pol=np.array([0., 0., 1.]),
                                  delta=-2.73, beta=lambda k, R: 0.16))"""
# Then the magnetic field:
magField = lambda R: np.zeros(R.shape)

# Hamiltonian for F=2->F=3
Hg, mugq = pylcp.hamiltonians.singleF(F=2, gF=0, muB=1)
He, mueq = pylcp.hamiltonians.singleF(F=3, gF=1/3, muB=1)
dijq = pylcp.hamiltonians.dqij_two_bare_hyperfine(2, 3)
hamiltonian = pylcp.hamiltonian(Hg, He-0.*np.eye(He.shape[0]),
                                mugq, mueq, dijq)
hamiltonian.print_structure()
# 0.16/2*1/(1+4*2.73**2)*dijq[1,0,1]**2


# %% Now, let's compute the optical pumping based on the rate equations:
rateeq = pylcp.rateeq(laserBeams, magField, hamiltonian)

N0 = np.zeros((rateeq.hamiltonian.n,))
N0[0] = 1
rateeq.set_initial_pop(N0)
rateeq.evolve_populations([0, 2*np.pi*600])

fig, ax = plt.subplots(1, 1)
for jj in range(5):
    ax.plot(rateeq.sol.t/2/np.pi, rateeq.sol.y[jj,:], '-', color='C{0:d}'.format(jj),
            linewidth=0.5)
ax.set_xlabel('$t/(2\pi\Gamma)$')
ax.set_ylabel('$\\rho_{ii}$')

# %% Now try the optical Bloch equations:
obe1 = pylcp.obe(laserBeams, magField, hamiltonian,
                 transform_into_re_im=False)

rho0 = np.zeros((obe1.hamiltonian.n**2,))
rho0[0] = 1.
obe1.set_initial_rho(np.real(rho0))
tic = time.time()
obe1.evolve_density(t_span=[0, 2*np.pi*600])
toc = time.time()
print('Computation time is  %.2f s.' % (toc-tic))

(t, rho1) = obe1.reshape_sol()
for jj in range(5):
    ax.plot(t/2/np.pi, np.abs(rho1[jj, jj]), '--',
            color='C{0:d}'.format(jj),
            linewidth=0.5)
fig

# %% Now try the optical Bloch equations, using the obe, transformed into Re/Im.
obe2 = pylcp.obe(laserBeams, magField, hamiltonian,
                 transform_into_re_im=True)
obe2.set_initial_rho(np.real(rho0))
tic = time.time()
obe2.evolve_density(t_span=[0, 2*np.pi*600], method='RK45')
toc = time.time()
print('Computation time is  %.2f s.' % (toc-tic))

(t, rho2) = obe2.reshape_sol()
for jj in range(5):
    ax.plot(t/2/np.pi, np.abs(rho2[jj, jj]), '-.',
            color='C{0:d}'.format(jj),
            linewidth=0.5)
fig

# %% Add the end points game:
Neq = rateeq.equilibrium_populations(np.array([0., 0., 0.]),
                                     np.array([0., 0., 0.]), 0.)

for jj in range(5):
    ax.plot(obe1.sol.t[-1]/2/np.pi,
            Neq[jj], '.',
            color='C{0:d}'.format(jj),
            linewidth=0.5)
fig
