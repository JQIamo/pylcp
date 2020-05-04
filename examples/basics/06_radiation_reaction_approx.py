import numpy as np
import matplotlib.pyplot as plt
import pylcp
from pylcp.common import spherical2cart
from scipy.integrate import solve_ivp
import time
plt.style.use('paper')

laserBeams = pylcp.laserBeams([
    {'kvec': np.array([1., 0., 0.]), 'pol':np.array([0., 1., 0.]), 'delta':0., 'beta':2.0}
    ])

# Then the magnetic field:
magField = lambda R: np.zeros(R.shape)

# %%
# Now define the extremely simple Hamiltonian:
Hg = np.array([[0.]])
mugq = np.array([[[0.]], [[0.]], [[0.]]])
He = np.array([[0.]])
mueq = np.array([[[0.]], [[0.]], [[0.]]])
dijq = np.array([[[0.]], [[1.]], [[0.]]])

hamiltonian = pylcp.hamiltonian(Hg, He, mugq, mueq, dijq)
hamiltonian.print_structure()

"""H_g, muq_g = pylcp.hamiltonians.singleF(F=0, gF=1, muB=1)
H_e, mue_q = pylcp.hamiltonians.singleF(F=1, gF=1, muB=1)
d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
hamiltonian = pylcp.hamiltonian()
hamiltonian.add_H_0_block('g', H_g)
hamiltonian.add_H_0_block('e', H_e-0.*np.eye(H_e.shape[0]))
hamiltonian.add_d_q_block('g', 'e', d_q, gamma=gamma)"""

obe = pylcp.obe(laserBeams, magField, hamiltonian)

def build_decay_ev(self):
    """
    This method constructs the decay portion of the OBE using the radiation
    reaction approximation.
    """
    d_q_bare = self.hamiltonian.d_q_bare
    d_q_star = self.hamiltonian.d_q_star

    ev_mat = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                       dtype='complex128')

    for key in d_q_bare:
        d_q = d_q_bare[key] + d_q_star[key]
        # The first index we want to capture
        for ii in range(self.hamiltonian.n):
            # The second index we want to capture
            for jj in range(self.hamiltonian.n):
                # The state that comes from the Hamiltonian
                for kk in range(self.hamiltonian.n):
                    # The state comes from for the expansion of E
                    for ll in range(self.hamiltonian.n):
                        for mm, q in enumerate(np.arange(-1., 2., 1)):
                            # first term in the commutator
                            """ev_mat[self.density_index(ii, jj),
                                   self.density_index(ii, ll)] -= \
                            (-1.)**q*d_q_star[key][mm, jj, kk]*d_q_bare[key][2-mm, kk, ll]"""
                            # second term in commutator
                            ev_mat[self.density_index(ii, jj),
                                   self.density_index(ll, jj)] += \
                            (-1.)**q*d_q_star[key][mm, kk, ii]*d_q_bare[key][2-mm, ll, kk]

                            """# first term in the commutator
                            ev_mat[self.density_index(ii, jj),
                                   self.density_index(ii, ll)] -= \
                            (-1.)**q*d_q_bare[key][mm, kk, jj]*d_q_star[key][2-mm, ll, kk]
                            # second term in commutator
                            ev_mat[self.density_index(ii, jj),
                                   self.density_index(ll, jj)] += \
                            (-1.)**q*d_q_bare[key][mm, ii, kk]*d_q_star[key][2-mm, kk, ll]"""
    return ev_mat

obe.hamiltonian.d_q_bare
ev_mat = build_decay_ev(obe)
obe.build_decay_ev()
obe.ev_mat['decay']
print(ev_mat, obe.ev_mat['decay'])
np.allclose(obe.ev_mat['decay'], ev_mat)
