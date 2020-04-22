"""
author: spe

This little script tests damped Rabi flopping as calculated with the optical
Bloch equations.
"""
import numpy as np
import matplotlib.pyplot as plt
import pylcp
import pylcp.rateeq
import pylcp.obe
from scipy.integrate import solve_ivp
plt.style.use('paper')

# %% Define the problem to start:
laserBeams = []
laserBeams.append(pylcp.laserBeam(np.array([1., 0., 0.]),
                                  pol=np.array([0., 0., 1.]),
                                  delta=1, beta=1))
"""laserBeams.append(pylcp.laserBeam(np.array([1., 0., 0.]),
                                  pol=np.array([0., 0., 1.]),
                                  delta=0, beta=lambda R, k: 3))"""
magField = lambda R: np.zeros(R.shape)

# Now define the extremely simple Hamiltonian:
Hg = np.array([[0.]])
mugq = np.array([[[0.]], [[0.]], [[0.]]])
He = np.array([[0.]])
mueq = np.array([[[0.]], [[0.]], [[0.]]])
dijq = np.array([[[0.]], [[1.]], [[0.]]])

hamiltonian = pylcp.hamiltonian(Hg, He, mugq, mueq, dijq)
hamiltonian.print_structure()

# %%
"""
Now, evolve it!
"""
# First the OBE:
obe = pylcp.obe.obe(laserBeams, magField, hamiltonian,
                     transform_into_re_im=False)
rho0 = np.zeros((hamiltonian.n**2,), dtype='complex128')
rho0[0] = 1.
obe.set_initial_rho(rho0)
obe.evolve_density([0, 15])

# Next, the rate equation:
rateeq = pylcp.rateeq.rateeq(laserBeams, magField, hamiltonian)
N0 = np.zeros((rateeq.hamiltonian.n,))
N0[0] = 1
rateeq.set_initial_pop(N0)
rateeq.evolve_populations([0, 15])

# Plot it all up:
def final_value(s, det):
    return s/2/(1+s+4*det**2)

(t, rho) = obe.reshape_sol()

fig, ax = plt.subplots(1, 1, num='evolution')
ax.plot(t/2/np.pi, np.abs(rho[0, 0, :]), linewidth=0.5,
         label='$\\rho_{00}$')
ax.plot(t/2/np.pi, np.abs(rho[1, 1, :]), linewidth=0.5,
         label='$\\rho_{11}$')
ax.plot(rateeq.sol.t/2/np.pi, np.abs(rateeq.sol.y[0, :]), linewidth=0.5,
         label='$\\rho_{00}$ (rate eq.)')
ax.plot(rateeq.sol.t/2/np.pi, np.abs(rateeq.sol.y[-1, :]), linewidth=0.5,
         label='$\\rho_{11}$ (rate eq.)')
ax.plot(t[-1]/2/np.pi,
         final_value(len(laserBeams)*laserBeams[0].return_beta(np.array([0., 0., 0.])),
                     laserBeams[0].delta) ,'o')
ax.legend(fontsize=6)
ax.set_xlabel('$t/\Gamma$')
ax.set_ylabel('$\\rho_{ii}$')

#
