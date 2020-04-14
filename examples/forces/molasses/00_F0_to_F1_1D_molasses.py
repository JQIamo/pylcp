"""
author: spe

This example covers calculating the forces in a one-dimensional optical molasses
using the optical bloch equations.  This example does the boring thing and
checks that everything is working on the F=0->F=1 transition, which of course
has no sub-Doppler effect.

It first checks the force along the z-direction.  One should look to see that
things agree with what one expects whether or not one puts the detuning on
the lasers or on the Hamilonian.  One should also look at whether the force
depends on transforming the OBEs into the real/imaginary components.
"""
import numpy as np
import matplotlib.pyplot as plt
import pylcp
import pylcp.tools
import time
#from pstats import SortKey
from scipy.optimize import fsolve
plt.style.use('paper')

# %% Define the multiple laser beam configurations to start:
# Can play with how we divide up the rotating frame a bit.  Answers should
# of course be independent (speed may not be)
laser_det = 0.
ham_det = -2.
beta = 1.25

laserBeams = {}
laserBeams['$\\sigma^+\\sigma^+$'] = pylcp.laserBeams([
    {'kvec':np.array([0., 0., 1.]), 'pol':np.array([0., 0., 1.]),
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0., 0., -1.]), 'pol':np.array([0., 0., 1.]),
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta},
    ])

laserBeams['$\\sigma^+\\sigma^-$'] = pylcp.laserBeams([
    {'kvec':np.array([0., 0., 1.]), 'pol':np.array([0., 0., 1.]),
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0., 0., -1.]), 'pol':np.array([1., 0., 0.]),
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta},
    ])

laserBeams['$\\pi_x\\pi_x$'] = pylcp.laserBeams([
    {'kvec':np.array([0., 0., 1.]), 'pol':np.array([1., 0., 0.]),
     'pol_coord':'cartesian', 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0., 0., -1.]), 'pol':np.array([1., 0., 0.]),
     'pol_coord':'cartesian', 'delta':laser_det, 'beta':beta},
    ])

laserBeams['$\\pi_x\\pi_y$'] = pylcp.laserBeams([
    {'kvec':np.array([0., 0., 1.]), 'pol':np.array([1., 0., 0.]),
     'pol_coord':'cartesian', 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0., 0., -1.]), 'pol':np.array([0., 1., 0.]),
     'pol_coord':'cartesian', 'delta':laser_det, 'beta':beta},
    ])

laserBeams['$\\sigma^+\\sigma^-$'].total_electric_field_gradient(np.array([0., 0., 0.]), 0.)
magField = lambda R: np.zeros(R.shape)

# Hamiltonian for F=0->F=1
Hg, Bgq = pylcp.hamiltonians.singleF(F=0, gF=0, muB=1)
He, Beq = pylcp.hamiltonians.singleF(F=1, gF=1, muB=1)
dijq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
ham_F0_to_F1 = pylcp.hamiltonian(Hg, He - ham_det*np.eye(3), Bgq, Beq, dijq)

obe={}
rateeq={}

# %%
"""
First, check to see that the rate equations and OBE agree for F=1 to F=2,
two-state solution:
"""
# Define a v axis:
v = np.arange(-5.0, 5.1, 0.5)

for jj, key in enumerate(laserBeams.keys()):
    obe[key] = pylcp.obe(laserBeams[key], magField, ham_F0_to_F1,
                             transform_into_re_im=False)

    # Generate a rateeq model of what's going on:
    obe[key].rateeq.generate_force_profile(
        [np.zeros(v.shape), np.zeros(v.shape), np.zeros(v.shape)],
        [np.zeros(v.shape), np.zeros(v.shape), v],
        name='molasses'
    )

    tic = time.time()
    obe[key].generate_force_profile(
        [np.zeros(v.shape), np.zeros(v.shape), np.zeros(v.shape)],
        [np.zeros(v.shape), np.zeros(v.shape), v],
        name='molasses', deltat_tmax=2*np.pi*100, deltat_v=4, itermax=1000,
        progress_bar=True,
    )
    toc=time.time()
    print('Total computation time for %s is %.3f' % (key, toc-tic))

# %%
"""
Plot 'er up:
"""
fig, ax = plt.subplots(1, 2, num='Optical Molasses F=0->F1', figsize=(6.5, 2.75))
for jj, key in enumerate(laserBeams.keys()):
    ax[0].plot(obe[key].profile['molasses'].V[2],
               obe[key].profile['molasses'].F[2],
               label=key, linewidth=0.5, color='C%d'%jj)
    ax[0].plot(obe[key].rateeq.profile['molasses'].V[2],
               obe[key].rateeq.profile['molasses'].F[2], '--',
               linewidth=0.5, color='C%d'%jj)
ax[0].legend(fontsize=6)
ax[0].set_xlabel('$v/(\Gamma/k)$')
ax[0].set_ylabel('$F/(\hbar k \Gamma)$')

key = '$\\pi_x\\pi_y$'
types = ['-', '--', '-.']
for q in range(3):
    ax[1].plot(v, obe[key].profile['molasses'].fq['g->e'][2, :, q, 0], types[q],
            linewidth=0.5, color='C0', label='$+k$, $q=%d$'%(q-1))
    ax[1].plot(v, obe[key].profile['molasses'].fq['g->e'][2, :, q, 1], types[q],
            linewidth=0.5, color='C1', label='$-k$, $q=%d$'%(q-1))
ax[1].plot(v, obe[key].profile['molasses'].F[2], 'k-',
           linewidth=0.75)
ax[1].legend(fontsize=6)
ax[1].set_xlabel('$v/(\Gamma/k)$')
fig.subplots_adjust(wspace=0.15)

# %%
"""
Let's run along x and y just to make sure that everything is Kosher.
"""
laserBeams = {}
laserBeams['x'] = {}
laserBeams['x']['$\\sigma^+\\sigma^+$'] = pylcp.laserBeams([
    {'kvec':np.array([ 1., 0., 0.]), 'pol':+1, 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([-1., 0., 0.]), 'pol':-1, 'delta':laser_det, 'beta':beta},
    ])
laserBeams['x']['$\\sigma^+\\sigma^-$'] = pylcp.laserBeams([
    {'kvec':np.array([ 1., 0., 0.]), 'pol':+1, 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([-1., 0., 0.]), 'pol':+1, 'delta':laser_det, 'beta':beta},
    ])
laserBeams['y'] = {}
laserBeams['y']['$\\sigma^+\\sigma^+$'] = pylcp.laserBeams([
    {'kvec':np.array([0.,  1., 0.]), 'pol':+1, 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0., -1., 0.]), 'pol':-1, 'delta':laser_det, 'beta':beta},
    ])
laserBeams['y']['$\\sigma^+\\sigma^-$'] = pylcp.laserBeams([
    {'kvec':np.array([0.,  1., 0.]), 'pol':+1, 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0., -1., 0.]), 'pol':+1, 'delta':laser_det, 'beta':beta},
    ])

obe = {}
for coord_key in laserBeams:
    obe[coord_key] = {}
    for pol_key in laserBeams[coord_key]:
        obe[coord_key][pol_key] = pylcp.obe(laserBeams[coord_key][pol_key],
                                            magField, ham_F0_to_F1,
                                            transform_into_re_im=False)

        if coord_key is 'x':
            V = [v, np.zeros(v.shape), np.zeros(v.shape)]
        elif coord_key is 'y':
            V = [np.zeros(v.shape), v, np.zeros(v.shape)]
        R = np.zeros((3,)+v.shape)
        # Generate a rateeq model of what's going on:
        obe[coord_key][pol_key].rateeq.generate_force_profile(
            R, V, name='molasses'
        )

        tic = time.time()
        obe[coord_key][pol_key].generate_force_profile(
            R, V, name='molasses', deltat_tmax=2*np.pi*100, deltat_v=4,
            itermax=1000, progress_bar=True
        )
        toc=time.time()
        print('Total computation time for %s along %s is %.3f' % (pol_key, coord_key, toc-tic))

# %%
"""
Plot 'er up:
"""
fig, ax = plt.subplots(1, 2, num='Optical Molasses F=0->F1', figsize=(6.5, 2.75))
for ii, coord_key in enumerate(laserBeams.keys()):
    for jj, pol_key in enumerate(laserBeams[coord_key].keys()):
        ax[ii].plot(obe[coord_key][pol_key].profile['molasses'].V[ii],
                   obe[coord_key][pol_key].profile['molasses'].F[ii],
                   label=pol_key, linewidth=0.5, color='C%d'%jj)
        ax[ii].plot(obe[coord_key][pol_key].rateeq.profile['molasses'].V[ii],
                   obe[coord_key][pol_key].rateeq.profile['molasses'].F[ii], '--',
                   linewidth=0.5, color='C%d'%jj)
ax[1].legend(fontsize=6)
ax[0].set_xlabel('$v_x/(\Gamma/k)$')
ax[1].set_xlabel('$v_y/(\Gamma/k)$')
ax[1].set_ylabel('$F/(\hbar k \Gamma)$')

# %%
"""
Run a simulation at resonance to see what the coherences and such are doing.
"""
v_i=0.2
key = '$\\pi_x\\pi_y$'
obe[key].set_initial_position_and_velocity(
    np.array([0., 0., 0.]), np.array([0., 0., v_i])
    )
rho0 = np.zeros((obe[key].hamiltonian.n**2,), dtype='complex128')
rho0[0] = 1.

if v_i==0 or np.abs(2*np.pi*20/v_i)>500:
    t_max = 500
else:
    t_max = 2*np.pi*20/np.abs(v_i)

obe[key].set_initial_rho_from_rateeq()
obe[key].evolve_density(t_span=[0, t_max], t_eval=np.linspace(0, t_max, 1001),)
(t, rho) = obe[key].reshape_sol()

f, flaser = obe[key].force_from_sol(return_laser=True)

fig, ax = plt.subplots(2, 2, num='OBE F=0->F1', figsize=(6.25, 5.5))
ax[0, 0].plot(t, np.real(rho[0, 0]), label='$\\rho_{00}$')
ax[0, 0].plot(t, np.real(rho[1, 1]), label='$\\rho_{11}$')
ax[0, 0].plot(t, np.real(rho[2, 2]), label='$\\rho_{22}$')
ax[0, 0].plot(t, np.real(rho[3, 3]), label='$\\rho_{33}$')
ax[0, 0].legend(fontsize=6)

ax[0, 1].plot(t, np.abs(rho[0, 1]), label='$\\rho_{01}$')
ax[0, 1].plot(t, np.abs(rho[0, 2]), label='$\\rho_{02}$')
ax[0, 1].plot(t, np.abs(rho[0, 3]), label='$\\rho_{03}$')
ax[0, 1].plot(t, np.abs(rho[1, 3]), label='$\\rho_{13}$')
ax[0, 1].legend(fontsize=6)

ax[1, 0].plot(t, flaser['g->e'][2, 0], '-', linewidth=0.75)
ax[1, 0].plot(t, flaser['g->e'][2, 1], '-', linewidth=0.75)
ax[1, 0].plot(t, f[2], 'k-', linewidth=0.5)

ax[1, 1].plot(t, obe[key].sol.y[-1], '-', label='$z$')
ax[1, 1].plot(t, obe[key].sol.y[-4], '--', label='$v_z$')
ax[1, 1].legend(fontsize=6)
#ax.plot(v, F_rateeq_F0_to_F1, v, F_obe)
