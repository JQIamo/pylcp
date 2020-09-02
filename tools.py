"""
author: spe

Tools that are useful for the pylcp packageself.
"""
import numpy as np
from pylcp import laserBeam, laserBeams
# from pylcp.rateeq import equilibrium_force
from scipy.optimize import fsolve


"""
Define some functions that make basic MOT configurations:
"""
def universaldvdxcurve(alpha,beta,det,nb):
    # First, pick out the inflection point that is largest in y:
    y0 = np.amax([1.0/6.0*(-np.sqrt(3.0*(1+nb*beta)) - 6.0*det),\
                  1.0/6.0*(np.sqrt(3.0*(1+nb*beta)) - 6.0*det)])

    # Next, calculate the expansion coefficients for the force:
    f1 = 3.0*np.sqrt(3.0)*nb*beta/4.0/(1+nb*beta)**1.5
    f0 = -(3.0*nb*beta)/(4.0*(1.0+nb*beta))

    # Next, add in small corrections due to the other beam's force (important at small delta):
    f1 -= (3.0*(nb*beta*np.sqrt(3.0*(1.0 + nb*beta)) - 12.0*det))/\
            (4.0*(1.0 + nb*beta - 2.0*np.sqrt(3.0*(1.0 + nb*beta))*det + 12.0*det**2)**2)
    f0 += (3.0*nb*beta)/(4.0*(1.0 + nb*beta - \
              2.0*np.sqrt(3.0)*np.sqrt(1.0 + nb*beta)*det + 12.0*det**2))

    if f1 + 4*alpha>0 and f1>0:
        # Last, calculate the two possible slopes:
        vp = np.array([0.5*(f1 - np.sqrt(f1)*np.sqrt(f1 + 4*alpha)),
                  0.5*(f1 + np.sqrt(f1)*np.sqrt(f1 + 4*alpha))])

        #print vp
        # The real slope is the one with vp<alpha:
        vp = vp[vp<0]

        # Now calculate the intercept:
        v0 = np.array([(f0*f1 - f1**2*y0 - f0*np.sqrt(f1)*np.sqrt(f1 + 4*alpha) + \
                f1**1.5*y0*np.sqrt(f1 + 4*alpha))/(2*f1*alpha),\
              (f0*f1 - f1**2*y0 + f0*np.sqrt(f1)*np.sqrt(f1 + 4*alpha) - \
                f1**1.5*y0*np.sqrt(f1 + 4*alpha))/(2*f1*alpha)])

        #print v0
        # The real intercept is the one with v0>y0
        v0 = v0[v0>0]

        return vp, v0
    else:
        return 0.0,0.0

# Now, try to solve the problem in the small beam limit with the  little
#  theory I worked out:
def smallBeamVcVeprop(alpha,beta,delta,nb):
    from scipy.optimize import fsolve

    f1 = 16*nb*beta*delta/(1+nb*beta+4*delta**2)**2

    if np.abs(f1)>4*alpha:
        gammap = -f1/2 + np.sqrt(f1**2+4*f1*alpha)/2
        gammam = -f1/2 - np.sqrt(f1**2+4*f1*alpha)/2

        xvst = lambda t: -gammam/(gammap-gammam)*np.exp(-gammap*t) + \
               +gammap/(gammap-gammam)*np.exp(-gammam*t)
        vvst = lambda t: gammam*gammap/(gammap-gammam)*np.exp(-gammap*t) + \
               -gammap*gammam/(gammap-gammam)*np.exp(-gammam*t)

        tzero = fsolve(xvst,-1/gammap)
        tm1 = fsolve(lambda t: xvst(t)+1,-1/gammap)

        veprop = vvst(tzero)
        vcprop = vvst(tm1)

        return veprop, vcprop, xvst, vvst, gammam, gammap, tm1, tzero
    else:
        raise(NotImplementedError, "underdamped calculation not yet implemented.")



"""
A useful set of functions for defining atom motion:
"""
def atom_motion(y, t, laserBeams, magField, extraBeta = None, dim=3):
    num_of_dimensions = len(y)/2

    if num_of_dimensions == 3:
        accel = forceFromBeams(y[0:3], y[3:6], laserBeams, magField,
                               extra_beta = extraBeta)
        if np.sum(np.isnan(accel))>0:
            raise RuntimeWarning("NaN encountered at r = " + str(y[0:3]) + \
                                 ") and v = (" + str(y[3:6]) + ").")

        return np.concatenate((y[3:6],accel))
    elif num_of_dimensions == 2:
        raise NotImplementedError("two dimensions not yet implemented")
    elif num_of_dimensions == 1:
        """if isinstance(y[0],np.ndarray):
            R = [np.zeros(y[0].shape)]*3
            V = [np.zeros(y[0].shape)]*3
        else:
            R = [0.0]*3
            V = [0.0]*3"""

        R = [0.0]*3
        V = [0.0]*3

        R[dim-1] = y[0]
        V[dim-1] = y[1]

        A = forceFromBeams(R, V, laserBeams, magField, extra_beta = extraBeta)

        return [V[dim-1],A[dim-1]]

def atom_motion_dvdx(v, x, laserBeams, magField, extraBeta, path=3, normalize = False):
    if isinstance(path,int):
        R = np.zeros((3,))
        V = np.zeros((3,))

        R[path-1] = x
        V[path-1] = v

        a = forceFromBeams(R, V, laserBeams, magField, extraBeta)[path-1]

        return a/v
    elif callable(path):
        """
        User has supplied a function that defines the path to integrate
        along, i.e., R = f(x), where x is the distance along the path and R
        is the position vector.  Now, we need to determine the velocity along
        the path as well, so we need the dl vector for the path.  We determine
        that by dl = ((R+dR)-R)/dx
        """
        dx = 1e-8
        R = path(x)
        dl = (path(x+dx)-path(x))/dx
        dl = dl/np.linalg.norm(dl) # Normalize

        V = dl*v

        A = forceFromBeams(R, V, laserBeams, magField, extraBeta)

        return np.sum(A*dl)/v

def isCaptured(xi, vi, laserBeams, magField,
                 x_center = np.array([0,0,0]), x_cutoff = 1e-3, v_cutoff = 1e-7):

    t = np.arange(0,100*np.sum(vi**2),0.1)

    init_cond = np.concatenate((xi,vi))

    sol_vs_t = odeint(atom_motion, init_cond, t, args=(laserBeams, magField))

    vf = sol_vs_t[-1,3:6]
    xf = sol_vs_t[-1,0:3]

    if np.sum((xf-x_center)**2)<x_cutoff and np.sum(vf**2)<v_cutoff:
        return True
    else:
        return False

def isRecaptured(v0, th, phi, laserBeams, magField,
                 x_center = np.array([0,0,0]), x_cutoff = 1.0, v_cutoff = 1e-7):

    v0 = np.array([v0*np.sin(th)*np.cos(phi),
                          v0*np.sin(th)*np.sin(phi), v0*np.cos(th)])

    return isCaptured(x_center,v0,laserBeams,magField,x_center,x_cutoff,v_cutoff)


"""
A useful function for analysis: returns the trap depth of a 3D MOT with the
given properties.
"""
def escape_velocity_3D_MOT(alpha, beta, delta, xc, xbh, xbv, adv_calc=False,
                           n_th_pts=10, n_phi_pts=20, dx = 0.05):
    nb = 6.0;

    if not adv_calc:
        def atom_motion_1D_xyz(v, x, delta, beta, alpha, betatot):
            accel = dimForce1Dxyz(x, v, delta, beta, alpha, betatot)
            return accel/v

        def atom_motion_1D_xeqy(v, x, delta, beta, alpha, betatot):
            accel = dimForce1Dyeqx(x, v, delta, beta, alpha, betatot)
            return accel/v

        th = np.linspace(0,np.pi/2,n_th_pts)
        phi = np.linspace(0,np.pi/4,n_phi_pts)

        TH,PHI = np.meshgrid(th,phi)

        xint = np.arange(xc,0.0,-dx)
        sol_dvdx = odeint(atom_motion_1D_xyz, 1e-9, xint,
                          args=(delta,lambda x: nb*beta, alpha,
                                lambda x: beta*(2.0 + (nb-2.0)*np.exp(-2*x**2/xbh**2))))
        ve_z = sol_dvdx[-1]

        sol_dvdx = odeint(atom_motion_1D_xyz, 1e-9, xint,
                          args=(delta,lambda x: nb*beta, 0.5*alpha,
                                lambda x: beta*(2.0 + 2.0*np.exp(-2*x**2/xbh**2))\
                                                 + 2.0*np.exp(-2*x**2/xbv**2)))
        ve_x = sol_dvdx[-1]

        sol_dvdx = odeint(atom_motion_1D_xeqy, 1e-9, xint,
                          args=(delta,lambda x: nb*beta*np.exp(-x**2/xbh**2), 0.5*alpha,
                                lambda x: beta*(4.0*np.exp(-x**2/xbh**2)\
                                                + 2.0*np.exp(-2*x**2/xbv**2))))
        ve_xy = sol_dvdx[-1]

        # Build an interpolation function that goes between everything:
        f = interp2d([0,0,np.pi/2,np.pi/2],[0,np.pi/4,0,np.pi/4],
                                 [ve_z, ve_z, ve_x, ve_xy])
        ve_avg = np.trapz(np.trapz(np.sin(TH)*f(th,phi),x=th,axis=-1),x=phi)/(4*np.pi/16.0)

        return ve_avg, [ve_z,ve_x,ve_xy]
    else:
        if n_th_pts%2 != 0:
            raise ValueError("n_th_pts must be divisible by two for advanced calculation")

        laserBeams = standard_six_beam_MOT(beta,delta, xbh, xc, xbv=xbv)

        # Add in the Gauss-Legendre quadrature points to evalute the function:
        xi, wi = leggauss(2*n_th_pts)
        th = np.arccos(xi)[::-1]
        phi = np.arange(np.pi/2/(2*n_th_pts),2*np.pi,np.pi/(2*n_th_pts))

        # Truncate to the 1/16 of the sphere that matters:
        wi = wi[th<np.pi/2]
        th = th[th<np.pi/2]
        phi = phi[phi<np.pi/4]

        # Mesh it up:
        TH, PHI = np.meshgrid(th,phi)

        # Now, actually run the calculation:
        ll=0

        VE = np.zeros(TH.shape)
        for ii in range(len(th)):
            for jj in range(len(phi)):
                ve_temp, i = bisectFindChangeValue(isRecaptured,6.0,
                                     args=(th[ii],phi[jj],laserBeams,
                                           lambda R: quadrupoleField3D(R,alpha),xc),
                                     tol=1e-4)
                VE[jj,ii] = ve_temp

                ll+=1
                printProgressBar(ll, ii*jj, prefix = 'Progress:',
                             suffix = 'complete', decimals = 1, length = 40)

        # Make the weights matrix:
        W = np.tile(wi,(len(phi),1))

        # Calculate the average:
        ve_avg = 16.0*(np.pi/(2*n_th_pts)*np.sum(np.sum(W*VE)))/4/np.pi

        return ve_avg, [TH,PHI,VE]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #%% Just as a check of the rotations in the Gaussian beam function, let's look at some stuff:
    xb = 5

    laserBeams = standard_six_beam_MOT(0, 1.0, xb, xb)

    rbeam = np.linspace(-2*xb, 2*xb, 201)

    plt.figure(2, figsize=(4, 2.75))
    plt.clf()
    plt.subplot('131')
    for jj in range(len(laserBeams)):
        plt.plot(rbeam,
                 laserBeams[jj].beta(
                     [rbeam] + [np.zeros(rbeam.shape)]*2)\
                 +0.1*jj)
    plt.subplot('132')
    for jj in range(len(laserBeams)):
        plt.plot(rbeam,
                 laserBeams[jj].beta(
                     [np.zeros(rbeam.shape)] + [rbeam] + [np.zeros(rbeam.shape)])\
                 +0.1*jj)
    plt.subplot('133')
    for jj in range(len(laserBeams)):
        plt.plot(rbeam,
                 laserBeams[jj].beta(
                     [np.zeros(rbeam.shape)]*2 + [rbeam])\
                 +0.1*jj,label="beam #{0:d}".format(jj))
    plt.legend()
