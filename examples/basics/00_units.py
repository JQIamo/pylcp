"""
author: spe

Just a little script that calculates various ratios for various atoms.  Note
that this package works in dimensionless units, where lengths are measured in
k and time is measured in Gamma.  This makes the equations simple in practice,
but there are two places where one needs to be careful.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts

plt.style.use('paper')

# %%
atoms = ['Li', 'Na', 'K', 'Rb', 'Cs',
         'Ca', 'Sr', 'Cd', 'Yb',
         'Cr', 'Ho', 'Dy', 'Er']
Gamma = [6, 10, 6, 6, 5.2,
         35, 30, 91, 28,
         5, 32.5, 32, 27.5]
k = [670, 589, 766, 780, 852,
     423, 461, 229, 399,
     425, 410, 421, 401]
T = [400, 250, 100, 75, 30,
     500, 650, 80, 425,
     1600, 1150, 1250, 1300]
m = [7, 23, 38, 87, 132.9,
     40, 88, 112.4, 171,
     52, 165, 162, 166]

# %%
Gamma = 2*np.pi*np.array(Gamma)*1e6
k = 2*np.pi/(np.array(k)*1e-9)
T = np.array(T)+273
m = np.array(m)*1.661e-27
d = 1e-2

# %%
for (label, gamma_i, k_i, m_i) in zip(atoms, Gamma, k, m):
    print(label, cts.hbar*k_i**2/m_i/gamma_i)

# %%
fig, ax = plt.subplots(1, 1)
ax.semilogy(np.arange(len(atoms)), (d*k*cts.hbar*Gamma/(4*cts.k*T))**2,'v',
            markersize=4)
ax.set_ylabel('$(d\hbar k \Gamma/4 k_B T)^2$')
ax.set_xticks(np.arange(len(atoms)))
ax.set_xticklabels(atoms)
#fig.savefig('ease_of_cooling.pdf')
