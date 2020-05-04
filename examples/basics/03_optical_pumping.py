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
amplitude) whereas the optical Bloch equations do.
"""
import numpy as np
import matplotlib.pyplot as plt
import pylcp
from pylcp.common import spherical2cart
from scipy.integrate import solve_ivp
import time
plt.style.use('paper')

transform = False # Change the variable to transform OBEs into re/im components.

# %%
"""
Let's start by checking rotations by optically pumping with circular polarized
light along all three axes and seeing if it pumps to the appropriate spin state
along those axes.

We will do this in multiple ways.  The first is to construct the Hamiltonian
directly, the second is
"""
# Which polarization do we want?
pol = +1

# First, create the laser beams:
laserBeams = {}
laserBeams['x']= pylcp.laserBeams([
    {'kvec': np.array([1., 0., 0.]), 'pol':pol, 'delta':0., 'beta':2.0}
    ])
laserBeams['y']= pylcp.laserBeams([
    {'kvec': np.array([0., 1., 0.]), 'pol':pol, 'delta':0., 'beta':2.0}
    ])
laserBeams['z']= pylcp.laserBeams([
    {'kvec': np.array([0., 0., 1.]), 'pol':pol, 'delta':0., 'beta':2.0}
    ])

# For making the Hamiltonian
E={}
for key in laserBeams:
    E[key] = laserBeams[key].cartesian_pol()[0]

# Then the magnetic field:
magField = lambda R: np.zeros(R.shape)
gF=1
# Hamiltonian for F=0->F=1
Hg, mugq, basis_g = pylcp.hamiltonians.singleF(F=0, gF=gF, muB=1, return_basis=True)
He, mueq, basis_e = pylcp.hamiltonians.singleF(F=1, gF=gF, muB=1, return_basis=True)
d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
hamiltonian = pylcp.hamiltonian(Hg, He, mugq, mueq, d_q)
laserBeams['x'].total_electric_field(np.array([0., 0., 0.]), 0.)
obe = {}

basis =np.concatenate((basis_g, basis_e))

# Excited state spin observable.  The spin is not along the
S_ex = -spherical2cart(mueq)/gF
S = np.zeros((3, hamiltonian.n, hamiltonian.n), dtype='complex128')
S[:, 1:, 1:] = S_ex
fig, ax = plt.subplots(3, 2, figsize=(6.5, 2.5*2.75))
for ii, key in enumerate(laserBeams):
    # In this loop, we will make the
    ax[ii, 1].plot(np.linspace(0, 4*np.pi, 51),
                   pol*(1-np.cos(np.linspace(0, 4*np.pi, 51)))/2, 'k-',
                   linewidth=1)
    d = spherical2cart(d_q)
    H_sub = -0.5*np.tensordot(d, E[key], axes=(0, 0))

    H = np.zeros((4, 4)).astype('complex128')
    H[0, 1:] = H_sub
    H[1:, 0] = np.conjugate(H_sub)

    H_sub2 = np.zeros(d_q[0].shape).astype('complex128')
    for kk, q in enumerate(np.arange(-1, 2, 1)):
        H_sub2 -= 0.5*(-1.)**q*d_q[kk]*laserBeams[key].beam_vector[0].pol[2-kk]

    H2 = np.zeros((4, 4)).astype('complex128')
    H2[0, 1:] = H_sub2
    H2[1:, 0] = np.conjugate(H_sub2)

    H3 = hamiltonian.return_full_H({'g->e': laserBeams[key].beam_vector[0].pol},
                                   np.zeros((3,)).astype('complex128'))
    psi0 = np.zeros((4,)).astype('complex128')
    psi0[0] = 1.
    sol = solve_ivp(lambda t, x: -1j*H2 @ x, [0, 4*np.pi], psi0,
                    t_eval=np.linspace(0, 4*np.pi, 51))

    print(np.allclose(H, H2), np.allclose(H, H3))

    S_av = np.zeros(sol.t.shape)
    for jj in range(sol.t.size):
        S_av[jj] = np.conjugate(sol.y[1:, jj])@S_ex[ii]@sol.y[1:, jj]

    for jj in range(4):
        ax[ii, 0].plot(sol.t, np.abs(sol.y[jj, :])**2, '--', color='C%d'%jj)

    ax[ii, 1].plot(sol.t, S_av, '--')

    obe[key] = pylcp.obe(laserBeams[key], magField, hamiltonian,
                         transform_into_re_im=transform)
    rho0 = np.zeros((16,))
    rho0[0] = 1 # Always start in the ground state.

    obe[key].ev_mat['decay'][:, :] = 0. # Forcibly turn off decay, make it like S.E.
    obe[key].set_initial_rho(rho0)
    obe[key].evolve_density(t_span=[0, 2*np.pi*2], t_eval=np.linspace(0, 4*np.pi, 51))
    obe[key].observable(S)
    (t, r, v, rho) = obe[key].reshape_sol()

    for jj in range(4):
        ax[ii, 0].plot(t, np.real(rho[jj, jj]), linewidth=0.75, color='C%d'%jj,
                       label='$|%d,%d\\rangle$'%(basis[jj, 0], basis[jj, 1]))
    ax[ii, 0].set_ylabel('$\\rho_{ii}$')

    ax[ii, 1].plot(t, S_av, linewidth=0.75)
    ax[ii, 1].set_ylabel('$\\langle S_%s\\rangle$'%key)

[ax[-1, jj].set_xlabel('$\Gamma t$') for jj in range(2)]
ax[0, 0].legend()
fig.subplots_adjust(left=0.08, bottom=0.05, wspace=0.22)

# %%
"""
Next, let's apply a magnetic field and see if we can tune the lasers into
resonance.  This will check to make sure that we have the detunings right.

With g_F>0, the shift for +m_F is upwards, requiring a blue-shift on the
lasers (or, equivalently, the Hamiltonian) to compensate.
"""
magField = {}
magField['x'] = lambda R: np.array([-1., 0., 0.])
magField['y'] = lambda R: np.array([0., -1., 0.])
magField['z'] = lambda R: np.array([0., 0., -1.])

pol=+1
laser_det=-0.5
ham_det=-0.5
laserBeams = {}
laserBeams['x']= pylcp.laserBeams([
    {'kvec': np.array([1., 0., 0.]), 'pol':pol, 'delta':laser_det, 'beta':2.0}
    ])
laserBeams['y']= pylcp.laserBeams([
    {'kvec': np.array([0., 1., 0.]), 'pol':pol, 'delta':laser_det, 'beta':2.0}
    ])
laserBeams['z']= pylcp.laserBeams([
    {'kvec': np.array([0., 0., 1.]), 'pol':pol, 'delta':laser_det, 'beta':2.0}
    ])

hamiltonian = pylcp.hamiltonian(Hg, He-ham_det*np.eye(3), mugq, mueq, d_q)

fig, ax = plt.subplots(3, 2, figsize=(6.5, 2.5*2.75))
for ii, key in enumerate(laserBeams):
    obe[key] = pylcp.obe(laserBeams[key], magField[key], hamiltonian,
                         transform_into_re_im=transform)
    rho0 = np.zeros((16,))
    rho0[0] = 1 # Always start in the ground state.

    obe[key].ev_mat['decay'][:, :] = 0. # Forcibly turn off decay, make it like S.E.
    obe[key].set_initial_rho(rho0)
    obe[key].evolve_density(t_span=[0, 2*np.pi*2], t_eval=np.linspace(0, 4*np.pi, 51))

    (t, r, v, rho) = obe[key].reshape_sol()

    for jj in range(4):
        ax[ii, 0].plot(t, np.real(rho[jj, jj]), linewidth=0.75, color='C%d'%jj,
                        label='$|%d,%d\\rangle$'%(basis[jj, 0], basis[jj, 1]))
    ax[ii, 0].set_ylabel('$\\rho_{ii}$')

    S_av = np.zeros(t.shape)
    for jj in range(t.size):
        S_av[jj] = np.real(np.sum(np.sum(S_ex[ii]*rho[1:, 1:, jj])))

    ax[ii, 1].plot(t, S_av, linewidth=0.75)
    ax[ii, 1].set_ylabel('$\\langle S_%s\\rangle$'%key)
    ax[ii, 1].set_ylim((0, 1))

[ax[-1, jj].set_xlabel('$\Gamma t$') for jj in range(2)]
ax[0, 0].legend()
fig.subplots_adjust(left=0.08, bottom=0.05, wspace=0.22)

# %%
"""
Finally, let's switch to some different polarization along z:
"""
# Which polarization do we want?
pol = +1
ham_det = 0.

# First, create the laser beams:
laserBeams['$\\pi_x$']= pylcp.laserBeams([
    {'kvec': np.array([0., 0., 1.]), 'pol':np.array([1., 0., 0.]),
     'pol_coord':'cartesian', 'delta':0., 'beta':2.0}
    ])
laserBeams['$\\pi_y$']= pylcp.laserBeams([
    {'kvec': np.array([0., 0., 1.]), 'pol':np.array([0., 1., 0.]),
     'pol_coord':'cartesian', 'delta':0., 'beta':2.0}
    ])

hamiltonian = pylcp.hamiltonian(Hg, He-ham_det*np.eye(3), mugq, mueq, d_q)
magField = lambda R: np.array([0., 0., 0.])

fig, ax = plt.subplots(2, 2, figsize=(6.5, 1.75*2.75))
for ii, key in enumerate(['$\\pi_x$', '$\\pi_y$']):
    d = spherical2cart(d_q)
    E = spherical2cart(laserBeams[key].beam_vector[0].pol)

    H_sub = -0.5*np.tensordot(d, E, axes=(0, 0))

    H = np.zeros((4, 4)).astype('complex128')
    H[0, 1:] = H_sub
    H[1:, 0] = np.conjugate(H_sub)

    H_sub2 = np.zeros(d_q[0].shape).astype('complex128')
    for kk, q in enumerate(np.arange(-1, 2, 1)):
        H_sub2 -= 0.5*(-1.)**q*d_q[kk]*laserBeams[key].beam_vector[0].pol[2-kk]

    H2 = np.zeros((4, 4)).astype('complex128')
    H2[0, 1:] = H_sub2
    H2[1:, 0] = np.conjugate(H_sub2)

    H3 = hamiltonian.return_full_H({'g->e': laserBeams[key].beam_vector[0].pol},
                                   np.zeros((3,)).astype('complex128'))

    psi0 = np.zeros((4,)).astype('complex128')
    psi0[0] = 1.
    sol = solve_ivp(lambda t, x: -1j*H @ x, [0, 4*np.pi], psi0, t_eval=np.linspace(0, 4*np.pi, 51))

    print(np.allclose(H, H2), np.allclose(H, H3))

    S_av = np.zeros((3,)+ sol.t.shape)
    for jj in range(3):
        for kk in range(sol.t.size):
            S_av[jj, kk] = np.conjugate(sol.y[1:, kk])@S_ex[jj]@sol.y[1:, kk]

    for jj in range(4):
        ax[ii, 0].plot(sol.t, np.abs(sol.y[jj, :])**2, '--', color='C%d'%jj)

    for jj in range(3):
        ax[ii, 1].plot(sol.t, S_av[jj], '--', color='C%d'%jj)

    obe[key] = pylcp.obe(laserBeams[key], magField, hamiltonian, transform_into_re_im=transform)
    rho0 = np.zeros((16,))
    rho0[0] = 1 # Always start in the ground state.

    obe[key].ev_mat['decay'][:, :] = 0. # Forcibly turn off decay, make it like S.E.
    obe[key].set_initial_rho(rho0)
    obe[key].evolve_density(t_span=[0, 2*np.pi*2], t_eval=np.linspace(0, 4*np.pi, 51))
    S_av = obe[key].observable(S)
    (t, r, v, rho) = obe[key].reshape_sol()

    for jj in range(4):
        ax[ii, 0].plot(t, np.real(rho[jj, jj]), linewidth=0.75, color='C%d'%jj,
                       label='$|%d,%d\\rangle$'%(basis[jj, 0], basis[jj, 1]))
    ax[ii, 0].set_ylabel('$\\rho_{ii}$')

    [ax[ii, 1].plot(t, S_av[jj], linewidth=0.75, color='C%d'%jj) for jj in range(3)]
    ax[ii, 1].set_ylabel('$\\langle S\\rangle$')

# %%
"""
Let's now re-define the problem to match the Ungar paper.  Note that one can
put the detuning on the laser or put the detuning on the Hamiltonian.
The latter is faster.
"""
# First the laser beams:
laserBeams = {}
laserBeams['$\\pi_z$']= pylcp.laserBeams([
    {'kvec': np.array([1., 0., 0.]), 'pol':np.array([0., 0., 1.]),
     'pol_coord':'cartesian', 'delta':-2.73, 'beta':4*0.16}
    ])
laserBeams['$\\pi_y$']= pylcp.laserBeams([
    {'kvec': np.array([0., 0., 1.]), 'pol':np.array([0., 1., 0.]),
     'pol_coord':'cartesian', 'delta':-2.73, 'beta':4*0.16}
    ])
laserBeams['$\\pi_x$']= pylcp.laserBeams([
    {'kvec': np.array([0., 0., 1.]), 'pol':np.array([1., 0., 0.]),
     'pol_coord':'cartesian', 'delta':-2.73, 'beta':0.16},
    {'kvec': np.array([0., 0., -1.]), 'pol':np.array([1., 0., 0.]),
     'pol_coord':'cartesian', 'delta':-2.73, 'beta':0.16}
    ])

# Then the magnetic field:
magField = lambda R: np.zeros(R.shape)

# Hamiltonian for F=2->F=3
gamma = 1
H_g, muq_g = pylcp.hamiltonians.singleF(F=2, gF=1, muB=1)
H_e, mue_q = pylcp.hamiltonians.singleF(F=3, gF=1, muB=1)
d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(2, 3)
hamiltonian = pylcp.hamiltonian()
hamiltonian.add_H_0_block('g', H_g)
hamiltonian.add_H_0_block('e', H_e-0.*np.eye(H_e.shape[0]))
hamiltonian.add_d_q_block('g', 'e', d_q, gamma=gamma)

hamiltonian.print_structure()

obe = {}
rateeq = {}
rateeq['$\\pi_z$'] = pylcp.rateeq(laserBeams['$\\pi_z$'], magField,
                                  hamiltonian)
obe['$\\pi_z$'] = pylcp.obe(laserBeams['$\\pi_z$'], magField, hamiltonian,
                            transform_into_re_im=transform)

N0 = np.zeros((rateeq['$\\pi_z$'].hamiltonian.n,))
N0[0] = 1
rateeq['$\\pi_z$'].set_initial_pop(N0)
rateeq['$\\pi_z$'].evolve_populations([0, 2*np.pi*600/gamma],
                                      max_step=1/gamma)

rho0 = np.zeros((obe['$\\pi_z$'].hamiltonian.n**2,))
rho0[0] = 1.
obe['$\\pi_z$'].set_initial_rho(np.real(rho0))
tic = time.time()
obe['$\\pi_z$'].evolve_density(t_span=[0, 2*np.pi*600/gamma],
                               max_step=1/gamma)
toc = time.time()
print('Computation time is  %.2f s.' % (toc-tic))

# Calculate the equilibrium populations:
Neq = rateeq['$\\pi_z$'].equilibrium_populations(np.array([0., 0., 0.]),
                                                 np.array([0., 0., 0.]), 0.)

fig, ax = plt.subplots(1, 1)
(t, r, v, rho1) = obe['$\\pi_z$'].reshape_sol()
for jj in range(5):
    ax.plot(gamma*rateeq['$\\pi_z$'].sol.t/2/np.pi,
            rateeq['$\\pi_z$'].sol.y[jj, :], '--',
            color='C{0:d}'.format(jj),
            linewidth=1.0)
    ax.plot(gamma*t/2/np.pi, np.abs(rho1[jj, jj]), '-',
            color='C{0:d}'.format(jj),
            linewidth=0.5)
    ax.plot(gamma*t[-1]/2/np.pi, Neq[jj], '.', color='C{0:d}'.format(jj),
            linewidth=0.5)

ax.set_xlabel('$\\Gamma t/2\\pi$')

# %%
"""
Next, we want to check that our rotations are working properly, so we will
run the same calculation for the z going beam with pi_y polarization.  But
before we even bother working with the OBE, we need to create the initial
state first.
"""
mug = spherical2cart(mugq)
S = -mug

# What are the eigenstates of 'y'?
E, U = np.linalg.eig(S[1])

inds = np.argsort(E)
E = E[inds]
U = U[:, inds]
Uinv = np.linalg.inv(U)

# In a positive magnetic field with g_F>0, I want the lowest eigenvalue. That
# corresponds to the -m_F state.
psi = U[:, 0]

rho0 = np.zeros((hamiltonian.n, hamiltonian.n), dtype='complex128')
for ii in range(hamiltonian.ns[0]):
    for jj in range(hamiltonian.ns[0]):
        rho0[ii, jj] = psi[ii]*np.conjugate(psi[jj])

print(rho0[:5,:5])
print(Uinv@rho0[:5,:5]@U)

obe['$\\pi_y$'] = pylcp.obe(laserBeams['$\\pi_y$'], magField, hamiltonian,
                            transform_into_re_im=transform)
obe['$\\pi_y$'].set_initial_rho(rho0.reshape(hamiltonian.n**2,))
obe['$\\pi_y$'].evolve_density(t_span=[0, 2*np.pi*600])

(t, rho) = obe['$\\pi_y$'].reshape_sol()

for jj in range(t.size):
    rho[:5, :5, jj] = Uinv@rho[:5, :5, jj]@U

fig, ax = plt.subplots(1, 1)
for jj in range(5):
    ax.plot(t/2/np.pi, np.abs(rho[jj, jj]), '-',
            color='C{0:d}'.format(jj),
            linewidth=0.5)
ax.set_xlabel('$\\Gamma t/2\\pi$')

# %%
"""
Now, let's do the same thing for $\pi_x$, except this time we have two laser
beams, with 1/4 of the intensity:
"""
# What are the eigenstates of 'y'?
E, U = np.linalg.eig(S[0])

inds = np.argsort(E)
E = E[inds]
U = U[:, inds]
Uinv = np.linalg.inv(U)

# In a positive magnetic field with g_F>0, I want the lowest eigenvalue. That
# corresponds to the -m_F state.
psi = U[:, 0]

rho0 = np.zeros((hamiltonian.n, hamiltonian.n), dtype='complex128')
for ii in range(hamiltonian.ns[0]):
    for jj in range(hamiltonian.ns[0]):
        rho0[ii, jj] = psi[ii]*np.conjugate(psi[jj])

obe['$\\pi_x$'] = pylcp.obe(laserBeams['$\\pi_x$'], magField, hamiltonian,
                            transform_into_re_im=transform)
obe['$\\pi_x$'].set_initial_rho(rho0.reshape(hamiltonian.n**2,))
obe['$\\pi_x$'].evolve_density(t_span=[0, 2*np.pi*600])

(t, rho) = obe['$\\pi_x$'].reshape_sol()

for jj in range(t.size):
    rho[:5, :5, jj] = Uinv@rho[:5, :5, jj]@U

fig, ax = plt.subplots(1, 1)
for jj in range(5):
    ax.plot(t/2/np.pi, np.abs(rho[jj, jj]), '-',
            color='C{0:d}'.format(jj),
            linewidth=0.5)
ax.set_xlabel('$\\Gamma t/2\\pi$')
