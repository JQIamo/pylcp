#%% Run the MOT trap depth along particular directions: z, x, and y=x.  This
# calculation uses the limiting cases in the lasercoolingphysics package.
beta = 10**np.arange(-3.0,4.1,0.2)
#xb = (0.0142/2)/atom_Rb.x0
#xc = (0.0254/2)/atom_Rb.x0
#alpha = 0.71
xb = (0.0142/2)/atom_Li.x0
xc = (0.0254/2)/atom_Li.x0
alpha = 0.071
det = np.arange(-6.0,-0.9,1.0)

ve_x = np.zeros((beta.size, det.size))
ve_z = np.zeros((beta.size, det.size))
ve_xy = np.zeros((beta.size, det.size))
ve_avg = np.zeros((beta.size, det.size))

for jj in range(det.size):
    for ii in range(beta.size):
        ve_avg[ii,jj], ves = lcp.escape_velocity_3D_MOT(alpha, beta[ii],
                                                        det[jj], xc, xb, xb)

        ve_z[ii,jj] = ves[0]
        ve_x[ii,jj] = ves[1]
        ve_xy[ii,jj] = ves[2]

# Plot up the result:
plt.figure(5,figsize=(6.5,2.75))
plt.clf()
plt.subplot('121')
for ii in range (det.size):
    plt.plot(beta,ve_avg[:,ii],'-',linewidth=1,
               color='C{0:d}'.format(ii),label='$\delta = {0:.1f}$'.format(det[ii]))
    plt.plot(beta,ve_z[:,ii],'--',linewidth=0.25,
               color='C{0:d}'.format(ii))
    plt.plot(beta,ve_x[:,ii],'-.',linewidth=0.25,
               color='C{0:d}'.format(ii))
    plt.plot(beta,ve_xy[:,ii],':',linewidth=0.25,
               color='C{0:d}'.format(ii))
plt.xlabel('$I/I_{sat}$')
plt.ylabel('$v_e/v_0$')
plt.xlim((0,5))
plt.subplot('122')
for ii in range (det.size):
    plt.plot(beta,ve_avg[:,ii],'-',linewidth=1,
               color='C{0:d}'.format(ii),label='$\delta = {0:.1f}$'.format(det[ii]))
    plt.loglog(beta,ve_z[:,ii],'--',linewidth=0.25,
               color='C{0:d}'.format(ii))
    plt.loglog(beta,ve_x[:,ii],'-.',linewidth=0.25,
               color='C{0:d}'.format(ii))
    plt.loglog(beta,ve_xy[:,ii],':',linewidth=0.25,
               color='C{0:d}'.format(ii))
plt.xlabel('$I/I_{sat}$')
plt.legend()
plt.subplots_adjust(left=0.125,wspace=0.2,right=0.95)

#%% Do the next best thing, average over the one dimension of the MOT:
def atom_motion(y, t, laserBeams, magField):
    accel = lcp.forceFromBeams(y[0:3].tolist(), y[3:6].tolist(), laserBeams, magField)
    return np.concatenate((y[3:6],accel))

def isRecaptured(v0, th, phi, laserBeams, magField,xc,cutoff = 1e-7):
    t = np.arange(0,100*np.sum(v0**2),0.1)

    init_cond = np.array([1e-9, 1e-9, 1e-9, v0*np.sin(th)*np.cos(phi),
                        v0*np.sin(th)*np.sin(phi), v0*np.cos(th)])

    sol_vs_t = odeint(atom_motion, init_cond, t, args=(laserBeams, magField))

    vf = sol_vs_t[-1,3:6]
    xf = sol_vs_t[-1,0:3]

    if np.abs(xf[0])<xc and np.abs(xf[1])<xc and np.abs(xf[2])<xc and np.sum(vf**2)<cutoff:
        return True
    else:
        return False

ii = 15; jj=1
print beta[ii], det[jj], ve_z[ii,jj], ve_x[ii,jj], ve_xy[ii,jj]
laserBeams = lcp.standard_six_beam_MOT(beta[ii],det[jj],xb, xc)

v0 = 6.52; th0 = 0; phi0 = 0
init_cond = np.array([0.0, 0.0, 1e-9, v0*np.sin(th0)*np.cos(phi0),
                        v0*np.sin(th0)*np.sin(phi0), v0*np.cos(th0)])

t = np.arange(0,1e3,0.1)
sol_vs_t = odeint(atom_motion, init_cond, t,
                  args = (laserBeams, lambda R: lcp.quadrupoleField3D(R,alpha)))

fig = plt.figure(7)
fig.clf()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_vs_t[:,0],sol_vs_t[:,1],sol_vs_t[:,2])
ax.set_xlim((-xc,xc))
ax.set_ylim((-xc,xc))
ax.set_zlim((-xc,xc))

#%% Now run the calculation over all possible angles, using code in the
# laser cooling physics package:
[ve_avg_adv, temp] = lcp.escape_velocity_3D_MOT(alpha, beta[ii], det[jj], xc,
                                                xb, xb,adv_calc = True,
                                                n_th_pts = 40)
TH = temp[0]
PHI = temp[1]
VE = temp[2]
temp = None

#th = np.linspace(0,np.pi/2,40)
#phi = np.linspace(0,np.pi/4,20)
#
#TH,PHI = np.meshgrid(th,phi)
#
#VE = np.zeros(TH.shape)
#for ii in range(len(th)):
#    for jj in range(len(phi)):
#        print ii,jj
#        ve_temp, i = bisectFindChangeValue(isRecaptured,6.0,
#                             args=(th[ii],phi[jj],laserBeams,
#                                   lambda R: lcp.quadrupoleField3D(R,alpha),xc),
#                             tol=1e-4)
#        VE[jj,ii] = ve_temp

plt.figure(8)
plt.clf()
ax = plt.subplot('211')
plt.contourf(TH,PHI,VE,25)
#plt.imshow(VE,extent=(th[0],th[-1],phi[0],phi[-1]),origin='lower',aspect='auto')
plt.ylabel('$\\phi$')
plt.yticks([0,np.pi/8,np.pi/4],('0','$\pi/8$','$\pi/4$'))
plt.xticks([])

ax2 = plt.subplot('212')
#plt.plot(th,np.transpose(VE)**2,'.-')
plt.plot(TH[0,:],np.transpose(VE)**2,'.-')
plt.xticks(np.pi*np.arange(0,1.0/2.0+0.01,1.0/8.0),('0','$\pi/8$','$\pi/4$','$3\pi/8$','$\pi/2$'))
plt.xlabel('$\\theta$')
plt.ylabel('$(v_e/v_0)^2$')

#ve_avg_adv = np.trapz(np.trapz(np.sin(TH)*VE,x=th,axis=-1),x=phi)/(4*np.pi/16.0)

plt.figure(5)
plt.subplot('121')
plt.plot(beta[ii],ve_avg_adv,'o',color='C{0:d}'.format(jj))
plt.subplot('122')
plt.plot(beta[ii],ve_avg_adv,'o',color='C{0:d}'.format(jj))

#%% Add some experimental data comparison:
#
#

#%% Let's try to match Kirk Madison's data from PRA 84 022708 (2011):
xbv = (0.007/2)/atom_Rb.x0 # Vertical 1/e^2 diameter
xbh = (0.0095/2)/atom_Rb.x0 # Horizontal 1/e^2 diameter
xc = (0.0254/2)/atom_Rb.x0
# xc = 0.050/x0Rb

# Select alpha:
# alpha = cts.value('Bohr magneton in Hz/T')/1e4*(27.9*100)*x0/atom.gammaHz # Vertical alpha
alpha = cts.value('Bohr magneton in Hz/T')*(27.9*100*1e-4)*atom_Rb.x0/atom_Rb.gammaHz # Horizontal alpha
print "alpha = {0:.3f}".format(alpha)

# Madison's data from Table II:
expbeta = np.array([2.7, 2.7, 2.7, 6.9, 9.6, 34.5])/atom_Rb.Isat
expdet = np.array([-5.0, -8.0, -10.0, -12.0, -12.0, -12.0])*1e6/atom_Rb.gammaHz

expdepth = np.array([0.55, 0.77, 1.05, 1.64, 1.93, 2.20])
dexpdepth = np.array([0.15, 0.17, 0.22, 0.10, 0.07, 0.05])

thrve_simp = np.zeros(expbeta.shape)
thrve_adv = np.zeros(expbeta.shape)
thrve_z = np.zeros(expbeta.shape)
thrve_x = np.zeros(expbeta.shape)
thrve_xy = np.zeros(expbeta.shape)

# Now, run the trap depth calculation:
for jj in range(expbeta.size):
    thrve_simp[jj], ve = lcp.escape_velocity_3D_MOT(alpha, expbeta[jj], expdet[jj],
         xc, xbh, xbv)

    thrve_z[jj] = ve[0]
    thrve_x[jj] = ve[1]
    thrve_xy[jj] = ve[2]

    thrve_adv[jj], ve = lcp.escape_velocity_3D_MOT(alpha, expbeta[jj], expdet[jj],
         xc, xbh, xbv, adv_calc=True, n_th_pts=10)

thrdepth_simp = 0.5*atom_Rb.mass*(thrve_simp*atom_Rb.v0)**2/cts.k
thrdepth_adv = 0.5*atom_Rb.mass*(thrve_adv*atom_Rb.v0)**2/cts.k

plt.figure(10)
plt.clf()
plt.errorbar(thrdepth_simp,expdepth,yerr=dexpdepth,fmt='.',label="simple")
plt.errorbar(thrdepth_adv,expdepth,yerr=dexpdepth,fmt='.',label="complex")
plt.plot([0,2.5],[0,2.5],'k--',linewidth=1.0)
plt.xlabel('$U_{theory}$ (K)')
plt.ylabel('$U_{exp}$ (K)')
plt.legend(loc='lower right')
