"""
author: SPE

Thus script calculates the transition rates between basic F sublevels and for
the entire Na D2 manifold as shown in Fig. 4 of Ungar, P. J., Weiss, D. S.,
Riis, E., & Chu, S. (1989). Optical molasses and multilevel atoms: theory.
Journal of the Optical Society of America B, 6(11), 2058.
http://doi.org/10.1364/JOSAB.6.002058
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
from pylcp import hamiltonians
from pylcp.atom import atom
plt.style.use('paper')

# %% Make the simple F->0 to F'->1
dijq_nonnorm = hamiltonians.dqij_two_bare_hyperfine(0, 1, normalize=False)
dijq_norm = hamiltonians.dqij_two_bare_hyperfine(0, 1)

print('dij^{-1} = ', dijq_nonnorm[0], '; ', ' dij^{0} = ', dijq_nonnorm[1],
      ';', 'dij^{1} = ', dijq_nonnorm[2])
print('dij^{-1} = ', dijq_norm[0], '; ', ' dij^{0} = ', dijq_norm[1],
      ';', 'dij^{1} = ', dijq_norm[2])

# %% Make the simple F->1 to F'->1
dijq_nonnorm = hamiltonians.dqij_two_bare_hyperfine(1, 1, normalize=False)
dijq_norm = hamiltonians.dqij_two_bare_hyperfine(1, 1)

print('dij^{-1} = ', dijq_nonnorm[0], '; ', ' dij^{0} = ', dijq_nonnorm[1],
      ';', 'dij^{1} = ', dijq_nonnorm[2])
print('dij^{-1} = ', dijq_norm[0], '; ', ' dij^{0} = ', dijq_norm[1],
      ';', 'dij^{1} = ', dijq_norm[2])

# %% Make the simple F->1 to F'->2
dijq_nonnorm = hamiltonians.dqij_two_bare_hyperfine(1, 2, normalize=False)
dijq_norm = hamiltonians.dqij_two_bare_hyperfine(1, 2)

print('dij^{-1} = ', dijq_nonnorm[0], '; ', ' dij^{0} = ', dijq_nonnorm[1],
      ';', 'dij^{1} = ', dijq_nonnorm[2])
print('dij^{-1} = ', dijq_norm[0], '; ', ' dij^{0} = ', dijq_norm[1],
      ';', 'dij^{1} = ', dijq_norm[2])

# %% Now do the whole manifold:
dijq, basis_g, basis_e = hamiltonians.dqij_two_hyperfine_manifolds(
    1/2, 3/2, 3/2, normalize=True, return_basis=True
    )

for jj in range(dijq.shape[1]):
    for kk in range(dijq.shape[2]):
        for ii in range(dijq.shape[0]):
            if np.abs(dijq[ii, jj, kk])>0:
                print(basis_g[jj,:], basis_e[kk,:],
                      int(np.round(60*dijq[ii, jj, kk]**2)))
