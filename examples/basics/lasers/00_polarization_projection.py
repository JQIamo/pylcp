"""
author: spe

This little script helps to sort out what is going on with respect to
polarization projection.  When specifying a laser beam polarization, we allow
the use to specify either sigma^{+/-} light (defined in the case of the
quantitization axis of the system parallel to the k-vector of the light), or,
in terms of the a full three vector of the polarization.  In the latter case,
we also allow the user to specify either cartesian coordinates or spherical
coordinates (relative to +\hat{z}).  In either case, we will convert the
polarization first to spherical coordinates, then use a Wigner rotation matrix
to handle the polarization projection.
"""
import numpy as np
import matplotlib.pyplot as plt
import pylcp
plt.style.use('paper')

# %% These are some basic tests to how to handle weird points in the quant_axis:
"""X, Z = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3))
R = np.array([X, np.zeros(X.shape), Z])
print(R)
Rmag = np.linalg.norm(R, axis=0)
print(Rmag)

Rhat = np.array([np.zeros(Rmag.shape), np.zeros(Rmag.shape), np.ones(Rmag.shape)])
for ii in range(3):
    Rhat[ii][Rmag!=0] = R[ii][Rmag!=0]/Rmag[Rmag!=0]
print(Rhat)

Rhat = np.zeros(R.shape)
for ii in range(3):
    Rhat[ii] = R[ii]/Rmag
print(Rhat)

for ii in range(3):
    if ii<2:
        Rhat[ii][np.isnan(Rhat[-1])] = 0.0
    else:
        Rhat[ii][np.isnan(Rhat[-1])] = 1.0

print(Rhat)"""

# %% Let's first use three different ways to define sigma^+ light.
laser1 = pylcp.laserBeam(np.array([0, 0, 1]), pol=+1)
laser2 = pylcp.laserBeam(np.array([0, 0, 1]), pol=np.array([0, 0, 1]),
                         pol_coord='spherical')
laser3 = pylcp.laserBeam(np.array([0, 0, 1]), pol=np.array([1/np.sqrt(2),
                                                            -1j/np.sqrt(2),
                                                            0]))

ths = np.linspace(0, np.pi, 51)
rs = np.array([np.sin(ths), np.zeros(ths.shape), np.cos(ths)])
rot_pols = laser1.project_pol(rs)
rot_pols2 = laser2.project_pol(rs)
rot_pols3 = laser3.project_pol(rs)

plt.figure()
plt.plot(ths, np.abs(rot_pols.T)**2, label='pol1')
plt.plot(ths, np.abs(rot_pols2.T)**2, '--', label='pol2')
plt.plot(ths, np.abs(rot_pols3.T)**2, '-.', label='pol3')
plt.legend()

# %% Basic check works.  Now, test the polarization strictly along x:
laser4 = pylcp.laserBeam(np.array([0.0, 1.0, 0.0]),
                         pol=np.array([1.0, 0.0, 0.0]))
print(laser4.pol)

rs2 =  np.array([np.zeros(ths.shape), np.sin(ths), np.cos(ths)])
plt.figure()
plt.plot(ths, np.abs(laser4.project_pol(rs)[0, :]**2), label='$\sigma^-$')
plt.plot(ths, np.abs(laser4.project_pol(rs)[1, :]**2), '--', label='$\pi$')
plt.plot(ths, np.abs(laser4.project_pol(rs)[2, :]**2), '-.', label='$\sigma^+$')
plt.plot(ths, np.abs(laser4.project_pol(rs2)[0, :]**2),
        label='$\sigma^-$, $\gamma=\pi/2$')
plt.plot(ths, np.abs(laser4.project_pol(rs2)[1, :]**2), '--',
        label='$\pi$, $\gamma=\pi/2$')
plt.plot(ths, np.abs(laser4.project_pol(rs2)[2, :]**2), '-.',
        label='$\sigma^+$, $\gamma=\pi/2$')
plt.legend()

# %% Now, let's put the polarization along y:
laser5 = pylcp.laserBeam(np.array([1.0, 0.0, 0.0]),
                         pol=np.array([0.0, 1.0, 0.0]))
print(laser4.pol)

plt.figure()
plt.plot(ths, np.abs(laser5.project_pol(rs)[0, :]**2), label='$\sigma^-$')
plt.plot(ths, np.abs(laser5.project_pol(rs)[1, :]**2), '--', label='$\pi$')
plt.plot(ths, np.abs(laser5.project_pol(rs)[2, :]**2), '-.', label='$\sigma^+$')
plt.plot(ths, np.abs(laser5.project_pol(rs2)[0, :]**2),
        label='$\sigma^-$, $\gamma=\pi/2$')
plt.plot(ths, np.abs(laser5.project_pol(rs2)[1, :]**2), '--',
        label='$\pi$, $\gamma=\pi/2$')
plt.plot(ths, np.abs(laser5.project_pol(rs2)[2, :]**2), '-.',
        label='$\sigma^+$, $\gamma=\pi/2$')
plt.legend()

# %% Now, let's try the polarization along z:
laser6 = pylcp.laserBeam(np.array([1.0, 0.0, 0.0]),
                         pol=np.array([0.0, 0.0, 1.0]))
print(laser4.pol)

plt.figure()
plt.plot(ths, np.abs(laser6.project_pol(rs)[0, :]**2), label='$\sigma^-$')
plt.plot(ths, np.abs(laser6.project_pol(rs)[1, :]**2), '--', label='$\pi$')
plt.plot(ths, np.abs(laser6.project_pol(rs)[2, :]**2), '-.', label='$\sigma^+$')
plt.legend()

# %% Finally, what happens if we sweep the magnetic field along a direction
# that is different from y?
rs3 = np.array([np.sin(ths)*np.cos(np.pi/3), np.sin(ths)*np.sin(np.pi/3), np.cos(ths)])

plt.figure()
plt.plot(ths, np.abs(laser4.project_pol(rs3)[0, :]**2), label='$\sigma^-$')
plt.plot(ths, np.abs(laser4.project_pol(rs3)[1, :]**2), '--', label='$\pi$')
plt.plot(ths, np.abs(laser4.project_pol(rs3)[2, :]**2), '-.', label='$\sigma^+$')
plt.legend()

# %%
"""
Based on these tests, it looks like the matrix elements of the little d matrix
are working properly, and given the fact that we sometimes rolled the
quantitization axis through the y axis rather than x and still got the correct
result, the exp(i gamma) phases appear to be working as well.
"""
