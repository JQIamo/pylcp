# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:48:48 2020

@author: sg

Single atom evolution in a MOT, with no schochatic force, no gravity, no 
interaction. Using OBE

"""


import numpy as np
import matplotlib.pyplot as plt
import pylcp
import pylcp.tools
import time
from scipy.optimize import fsolve
# plt.style.use('paper')

# %% Define the multiple laser beam configurations to start:
laser_det = 0
det = 2.5
beta = 1.25
transform = True

laserBeams = pylcp.laserBeams([
    {'kvec':np.array([1., 0., 0.]), 'pol':-1,
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([-1, 0., 0.]), 'pol':-1,
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0., 1., 0.]), 'pol':-1,
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0, -1., 0.]), 'pol':-1,
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0., 0., 1.]), 'pol':np.array([0., 0., 1.]),
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta},
    {'kvec':np.array([0, 0., -1]), 'pol':np.array([1, 0., 0.]),
     'pol_coord':'spherical', 'delta':laser_det, 'beta':beta}
    ])

alpha = 1e-3
magField = lambda R: pylcp.tools.quadrupoleField3D(R, alpha)
# magField = lambda R: np.zeros(R.shape)

#new_mag = {'3000': lambda R: pylcp.tools.quadrupoleField3D(R, 1e-2)}

# Hamiltonian for F=0->F=1
H_g, muq_g = pylcp.hamiltonians.singleF(F=0, gF=0, muB=1)
H_e, muq_e = pylcp.hamiltonians.singleF(F=1, gF=1, muB=1)
d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
hamiltonian = pylcp.hamiltonian(H_g, det*np.eye(3)+H_e, muq_g, muq_e, d_q)

obes = pylcp.obes(laserBeams, magField, hamiltonian,
                     transform_into_re_im=transform)



# %% Now try to evolve some initial state!

t = np.arange(0,501,100)

obes.evolve_motion_interaction(t, random_recoil=False, progress_bar=True, dipole_interaction=False, atom_in_out=True)

# %% Plot 'er up:
n = 1

fig, ax = plt.subplots(1, 2, num='Optical Molasses F=0->F1', figsize=(6.5, 2.75))
ax[0].plot(t,obes.rS[:,n,0],
           label='rx', linewidth=0.5)
ax[0].plot(t,obes.rS[:,n,1],
           label='ry', linewidth=0.5)
ax[0].plot(t,obes.rS[:,n,2],
           label='rz', linewidth=0.5)
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$r$')

ax[1].plot(t,obes.vS[:,n,0],
           label='vx', linewidth=0.5)
ax[1].plot(t,obes.vS[:,n,1],
           label='vy', linewidth=0.5)
ax[1].plot(t,obes.vS[:,n,2],
           label='vz', linewidth=0.5)
ax[1].legend(fontsize=6)
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$v$')
fig.subplots_adjust(wspace=0.15)
