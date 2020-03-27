"""
author: spe

Tools that are useful for the pylcp packageself.
"""
import numpy as np
from pylcp import laserBeam, laserBeams
# from pylcp.rateeq import equilibrium_force
from scipy.optimize import fsolve

"""
Define some functions that make some basic magnetic fields for MOTs, like
quarupole fields:
"""
# Define a function for the standard 3D quadrupole magnetic field
def quadrupoleField3D(R, alpha):
    if not len(R) == 3:
        raise ValueError("length of R must be equal to three")

    return np.array([-0.5*alpha*R[0], -0.5*alpha*R[1], alpha*R[2]])

# Define a function for the standard 2D quadrupole magnetic field
def quadrupoleField2D(R, alpha):
    if not len(R) == 2:
        raise ValueError("length of R must be equal to three")

    return np.array([-alpha*R[0], alpha*R[1]])

"""
Define some functions that make some basic shapes for laser beams, like
Gaussian beams:
"""
def gaussianBeam(R, k, wb):
    """
    Returns the intensity profile of a simple, Gaussian beam

    wb: 1/e^2 radius of the Gaussian beam.
    """
    if len(R) != len(k):
        raise ValueError("length of R and knorm must be equal!")
    elif len(R) == 2:
        th = np.arctan2(k[1], k[0])
        return np.exp(-2*((-R[0]*np.sin(th)+R[1]*np.cos(th))**2)/wb**2)
    elif len(R) == 3:
        # Angles of rotation:
        th = np.arccos(k[2])
        phi = np.arctan2(k[1], k[0])

        rvals = [np.cos(th)*np.cos(phi) - np.sin(phi),\
                 np.cos(phi) + np.cos(th)*np.sin(phi),\
                 -np.sin(th)]

        return np.exp(-2*(sum((rvals[jj]*R[jj])**2 for jj in range(len(R))))/wb**2)

# Next, the slightly more advanced "clipped" Gaussian beam:
def clipped_gaussian_beam(R, k, wb, rs):
    """
    Returns the intensity profile of a Gaussian beam that is clipped.

    wb: 1/e^2 radius of the Gaussian beam.

    rs: radius of the beam stop.
    """
    if len(R) != len(k):
        raise ValueError("length of R and knorm must be equal!")
    elif len(R) == 2:
        raise ValueError("clipped_gaussian_beam not implemented for len(R) = 2!")
        #th = np.arctan2(k[1], k[0])
        #return np.exp(-2*((-R[0]*np.sin(th)+R[1]*np.cos(th))**2)/wb**2)
    elif len(R) == 3:
        # Angles of rotation:
        th = np.arccos(k[2])
        phi = np.arctan2(k[1], k[0])
        # transform R into cylindrical coordinates centered on the beam:
        rvals = np.array([np.cos(th)*np.cos(phi) - np.sin(phi),\
                 np.cos(phi) + np.cos(th)*np.sin(phi),\
                 -np.sin(th)])
        # compute rho^2:
        rho_sq = sum((rvals[jj]*R[jj])**2 for jj in range(len(R)))

        return np.exp(-2*(rho_sq)/wb**2)*(np.sqrt(rho_sq)<rs)


"""
Define some functions that make basic MOT configurations:
"""
def standard_six_beam_MOT(beta, delta, pol=+1):
    """
    Method that returns the standard, six beam MOT configuration.
    Hint: define beta=lambda R, k: beta*gaussianBeam(R, k, xb) for a
    Gaussian beam.

    beta: saturation parameter of each laser beam

    delta: laser detuning from atomic resonance

    pol: polarization of the laser laser beams
    """
    return laserBeams([{'kvec':np.array([1., 0., 0.]), 'beta':beta,
                        'delta':delta, 'pol':-pol},
                       {'kvec':np.array([-1., 0., 0.]), 'beta':beta,
                        'delta':delta, 'pol':-pol},
                       {'kvec':np.array([0., 1., 0.]), 'beta':beta,
                        'delta':delta, 'pol':-pol},
                       {'kvec':np.array([0., -1., 0.]), 'beta':beta,
                        'delta':delta, 'pol':-pol},
                       {'kvec':np.array([0., 0., 1.]), 'beta':beta,
                        'delta':delta, 'pol':pol},
                       {'kvec':np.array([0., 0., -1.]), 'beta':beta,
                        'delta':delta, 'pol':pol}
                      ])


def standard_four_beam_MOT(beta,delta,xb,xc):
    """
    Method that returns the standard, four beam MOT configuration:
    """
    laserBeams = [0]*6
    laserBeams[0] = laserBeam([1, 0], beta=beta, delta=delta, pol=-1)
    laserBeams[1] = laserBeam([-1, 0], beta=beta, delta=delta, pol = -1)
    laserBeams[2] = laserBeam([0, 1], beta=beta, delta=delta, pol = +1)
    laserBeams[3] = laserBeam([0, -1], beta=beta, delta = delta, pol = +1)

    return laserBeams(laserBeams)


def grating_MOT_beams(delta, s, nr, thd,
                      pol=np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0]),
                      reflected_pol=np.array([np.pi, 0]),
                      reflected_pol_basis='poincare',
                      eta=None,
                      return_basis_vectors=False,
                      grating_angle=0):
    """
    Creates beams that would be made from a grating.  Parmeters:
        delta: detuning of the laser beams

        s: intensity of the laser beams

        nr: number of reflected beams

        thd: diffraction angle

        pol: input polarization.  Can be +1 or -1

        reflected_pol: two parameters that describe the reflected polarization
            depending on the reflected_pol_basis.

        reflected_pol_basis: the basis in which the reflection polarization is
            defined.  There are three bases currently programmed:

            'poincare': the poincare basis is the Poincare sphere with sigma^+
            at the north pole, sigma^- at the south pole, and s and p along the
            +x and -x axes, respectively.  In this case, reflected_pol[0] is
            the polar angle (0 corresponding to \sigma^- and \pi corresponding
            to sigma^+) and reflected_pol[1] is the azimuthal angle (0
            corresponding to p and pi corresponding to s)

            'jones_vector': relected_pol[0]*svec + reflected_pol[1]*pvec

            'waveplate': relected_pol[0] specified the angle of the slow axis
            relative to the p-vector.  reflection_pol[1] specifies the phase
            delay.

        eta: diffraction efficiency of each of the reflected beams

        grating_angle: overall azimuthal rotation of the grating
    """
    if not eta:
        eta = 1/nr

    if not isinstance(pol, np.ndarray) and not isinstance(pol, list):
        if pol == 1:
            pol = np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0])
        elif pol == -1:
            pol = np.array([-1/np.sqrt(2), -1j/np.sqrt(2), 0])
        else:
            raise ValueError('pol must be a three-vector or +/-1.')

    beams = []
    beams.append(laserBeam(np.array([0., 0., 1.]),
                           beta=s,
                           pol=pol, delta=delta))  # Incident beam

    # Preallocate memory for the polarizations (no need to store kvec or the
    # polarization because those are stored in the laser)
    svec = np.zeros((3, nr))
    pvec = np.zeros((3, nr))

    for ii in range(nr):  # Reflected beams
        kvec = np.array([-np.sin(thd)*np.cos(2*np.pi*ii/nr+grating_angle),
                         -np.sin(thd)*np.sin(2*np.pi*ii/nr+grating_angle),
                         -np.cos(thd)])
        svec[:, ii] = np.array([-np.sin(2*np.pi*ii/nr+grating_angle),
                                np.cos(2*np.pi*ii/nr+grating_angle),
                                0.])
        pvec[:, ii] = np.array([np.cos(thd)*np.cos(2*np.pi*ii/nr+grating_angle),
                                np.cos(thd)*np.sin(2*np.pi*ii/nr+grating_angle),
                                -np.sin(thd)])


        if reflected_pol_basis == 'poincare':
            pol_ref = np.sin(reflected_pol[0]/2)*np.exp(1j*reflected_pol[1]/2)*\
                      (svec[:, ii]/np.sqrt(2) + 1j/np.sqrt(2)*pvec[:, ii]) +\
                      np.cos(reflected_pol[0]/2)*np.exp(-1j*reflected_pol[1]/2)*\
                      (svec[:, ii]/np.sqrt(2) - 1j/np.sqrt(2)*pvec[:, ii])
        elif reflected_pol_basis == 'jones_vector':
            pol_ref = reflected_pol[0]*svec[:, ii] +\
                reflected_pol[1]*pvec[:, ii]
        elif reflected_pol_basis == 'stokes_parameters':
            raise NotImplementedError("Stokes parameters not yet implemented.")
        elif reflected_pol_basis == 'waveplate':
            svec_inp = -svec[:, ii]
            pvec_inp = np.array([np.cos(2*np.pi*ii/nr+grating_angle),
                                 np.sin(2*np.pi*ii/nr+grating_angle),
                                 0.])

            """
            The action of the waveplate is to induce a phase shift phi between
            the fast and slow axes of a beam.  We consider the waveplate as
            doing all the action on the input beam prior to reflection, rather
            than the complicated procedure of phase shifting, then reflecting
            then phase shifting again differently because of the angle.  Note
            that the slow and fast axes are defined relative to the input s and
            p:
            """
            slow_axis = pvec_inp*np.cos(reflected_pol[0]) +\
                        svec_inp*np.sin(reflected_pol[0])

            fast_axis = -pvec_inp*np.sin(reflected_pol[0]) +\
                        svec_inp*np.cos(reflected_pol[0])

            # Compute new polarization:
            pol_after_waveplate = \
            np.dot(pol, slow_axis)*np.exp(-1j*reflected_pol[1]/2)*slow_axis +\
            np.dot(pol, fast_axis)*np.exp(1j*reflected_pol[1]/2)*fast_axis

            # Reproject onto s and p:
            pol_ref = -pvec[:, ii]*np.dot(pol_after_waveplate, pvec_inp) +\
                svec[:, ii]*np.dot(pol_after_waveplate, svec_inp)
        else:
            raise NotImplementedError(
                "{0:s} polarization basis not implemented.".format(pol_basis)
                )

        #print(np.dot(pol,pvec_inp), np.dot(pol,svec_inp))
        #print(kvec, pol_ref, np.dot(kvec, pol_ref))

        beams.append(laserBeam(
            np.array([-np.sin(thd)*np.cos(2*np.pi*ii/nr+grating_angle),
                      -np.sin(thd)*np.sin(2*np.pi*ii/nr+grating_angle),
                      -np.cos(thd)]),
            beta=eta*s/np.cos(thd), pol=pol_ref, delta=delta))

    if return_basis_vectors:
        return (laserBeams(beams), pvec, svec)
    return laserBeams(beams)

# Helper Functions for grating_MOT_beams:
def chip_masked_beam(R, k, nr, s, wb, rs,
                     center_hole=0.0,
                     zgrating=1.0,
                     grating_angle=0):
    """Masks the intensity profile of the input beam after it
    goes through the chip.
    R: x,y,z coordinates at which to calculate intensity.

    k: normalized k vector of the laser beam.
    Assumed to be along the z axis!

    s: peak intensity of the laser beam.

    wb: 1/e^2 radius of the input gaussian beam.

    rs: radius of the input beam stop.

    nr: number of reflected beams.

    center_hole: inscribed radius of center hole.

    zgrating: z position of the diffraction grating chip.

    grating_angle: overall azimuthal rotation of the grating.
    """
    # Check that k is aligned with z axis:
    if np.any(np.arccos(k[2]) != 0):
        raise ValueError('chip_Masked_Beam must be aligned with z axis')
    # Determine the center angle of this section:
    th_center = (2*np.pi*np.arange(0, nr)/nr)+grating_angle
    # Initialize mask:
    if isinstance(R[0], np.ndarray):
        MASK = np.ones(R[0].shape, dtype=bool)
    else:
        MASK = True
    # Add in the center hole:
    for th_center_i in th_center:
        MASK = np.bitwise_and(MASK, (R[0]*np.cos(th_center_i) +
                                     R[1]*np.sin(th_center_i)) <= center_hole)
    # Make sure that the mask only applies after the chip.
    MASK = np.bitwise_or(MASK, R[2] <= zgrating)

    # Next, calculate the BETA function:
    BETA = s*clipped_gaussian_beam(R,k,wb,rs)*MASK.astype(float)

    return BETA

def grating_reflected_beam(R, k, ii, nr, s, eta, thd, k_in, wb, rs,
                      center_hole=0.0,
                      outer_radius=10.0,
                      zgrating=1.0,
                      grating_angle=0):
    """Intensity profile of the reflected beams from the chip.
    R: x,y,z coordinates at which to calculate intensity.

    k: normalized k vector of the laser beam.

    ii: beam number.

    nr: number of reflected beams.

    s: peak intensity of the INPUT laser beam.

    eta: diffraction efficiency of the grating.

    thd: diffraction angle for the grating.

    k_in: normalized k vector of the INPUT laser beam.
    Assumed to be along the z axis!

    wb: 1/e^2 radius of the INPUT gaussian beam.

    rs: radius of the INPUT beam stop.

    center_hole: inscribed radius of center hole.

    outer_radius: outer radius of the diffraction gratings.

    zgrating: z position of the diffraction grating chip.

    grating_angle: overall azimuthal rotation of the grating.
    """

    # Determine the center angle of this section:
    th_center = (2*np.pi*ii/nr)+grating_angle

    # Make a primed coordinate system that translates X,Y,Z positions back down
    # to the grating plane.
    Rp = [None]*3
    Rp[0] = R[0] - (R[2]-zgrating)*k[0]/k[2]
    Rp[1] = R[1] - (R[2]-zgrating)*k[1]/k[2]
    Rp[2] = zgrating # calculate beam profile at the chip

    # Next, calculate R and theta for this primed coordinate system:
    THp = np.arctan2(Rp[1], Rp[0])
    Radp = np.sqrt(Rp[0]**2 + Rp[1]**2)

    # Next, define the mask:
    wrap = lambda ang: (ang + np.pi) % (2 * np.pi ) - np.pi
    th_lower = wrap(th_center-np.pi/nr)
    th_upper = wrap(th_center+np.pi/nr)
    if th_upper < th_lower: # We extend over the pi branch cut:
        MASK = np.bitwise_or(THp < th_upper, THp > th_lower)
    else:
        MASK = np.bitwise_and(THp < th_upper, THp > th_lower)
    MASK = np.bitwise_and(MASK, Radp < outer_radius)
    MASK = np.bitwise_and(MASK, R[2] <= zgrating)

    # Add in the center hole:
    MASK = np.bitwise_and(MASK, (Rp[0]*np.cos(th_center) + \
                                 Rp[1]*np.sin(th_center)) >= center_hole)

    # Next, calculate the BETA function:
    BETA = eta*s*clipped_gaussian_beam(Rp,k_in,wb,rs)*MASK.astype(float)/np.cos(thd)

    return BETA

def grating_MOT_beams_adv(delta, s, nr, thd,
                      pol=np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0]),
                      reflected_pol=np.array([np.pi, 0]),
                      reflected_pol_basis='poincare',
                      eta=None,
                      wb=100.0,
                      rs=100.0,
                      return_basis_vectors=False,
                      center_hole=0.0,
                      outer_radius=10.0,
                      zgrating=1.0,
                      grating_angle=0):
    """
    Creates beams that would be made from a grating.  Parmeters:
        delta: detuning of the laser beams

        s: intensity of the laser beams

        nr: number of reflected beams

        thd: diffraction angle

        pol: input polarization.  Can be +1 or -1

        reflected_pol: two parameters that describe the reflected polarization
            depending on the reflected_pol_basis.

        reflected_pol_basis: the basis in which the reflection polarization is
            defined.  There are three bases currently programmed:

            'poincare': the poincare basis is the Poincare sphere with sigma^+
            at the north pole, sigma^- at the south pole, and s and p along the
            +x and -x axes, respectively.  In this case, reflected_pol[0] is
            the polar angle (0 corresponding to \sigma^- and \pi corresponding
            to sigma^+) and reflected_pol[1] is the azimuthal angle (0
            corresponding to p and pi corresponding to s)

            'jones_vector': relected_pol[0]*svec + reflected_pol[1]*pvec

            'waveplate': relected_pol[0] specified the angle of the slow axis
            relative to the p-vector.  reflection_pol[1] specifies the phase
            delay.

        eta: diffraction efficiency of each of the reflected beams

        wb: 1/e^2 radius of the INPUT gaussian beam.

        rs: radius of the INPUT beam stop.

        center_hole: inscribed radius of center hole.

        outer_radius: outer radius of the diffraction gratings.

        zgrating: z position of the diffraction grating chip.

        grating_angle: overall azimuthal rotation of the grating
    """
    if not eta:
        eta = 1/nr

    if not isinstance(pol, np.ndarray) and not isinstance(pol, list):
        if pol == 1:
            pol = np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0])
        elif pol == -1:
            pol = np.array([-1/np.sqrt(2), -1j/np.sqrt(2), 0])
        else:
            raise ValueError('pol must be a three-vector or +/-1.')

    k_in = np.array([0., 0., 1.])
    beams = []
    beams.append(laserBeam(k_in,
                           beta=lambda R,k: chip_masked_beam(R,k,nr,s,wb,rs,
                                                             center_hole,
                                                             zgrating,
                                                             grating_angle),
                           pol=pol,
                           delta=delta))  # Incident beam

    def factory(ii):
        return lambda R, k: grating_reflected_beam(R, k, ii, nr, s, eta, thd,
                                                   k_in, wb, rs, center_hole,
                                                   outer_radius, zgrating,
                                                   grating_angle)

    # Preallocate memory for the polarizations (no need to store kvec or the
    # polarization because those are stored in the laser)
    svec = np.zeros((3, nr))
    pvec = np.zeros((3, nr))

    for ii in range(nr):  # Reflected beams
        kvec = np.array([-np.sin(thd)*np.cos(2*np.pi*ii/nr+grating_angle),
                         -np.sin(thd)*np.sin(2*np.pi*ii/nr+grating_angle),
                         -np.cos(thd)])
        svec[:, ii] = np.array([-np.sin(2*np.pi*ii/nr+grating_angle),
                                np.cos(2*np.pi*ii/nr+grating_angle),
                                0.])
        pvec[:, ii] = np.array([np.cos(thd)*np.cos(2*np.pi*ii/nr+grating_angle),
                                np.cos(thd)*np.sin(2*np.pi*ii/nr+grating_angle),
                                -np.sin(thd)])


        if reflected_pol_basis == 'poincare':
            pol_ref = np.sin(reflected_pol[0]/2)*np.exp(1j*reflected_pol[1]/2)*\
                      (svec[:, ii]/np.sqrt(2) + 1j/np.sqrt(2)*pvec[:, ii]) +\
                      np.cos(reflected_pol[0]/2)*np.exp(-1j*reflected_pol[1]/2)*\
                      (svec[:, ii]/np.sqrt(2) - 1j/np.sqrt(2)*pvec[:, ii])
        elif reflected_pol_basis == 'jones_vector':
            pol_ref = reflected_pol[0]*svec[:, ii] +\
                reflected_pol[1]*pvec[:, ii]
        elif reflected_pol_basis == 'stokes_parameters':
            raise NotImplementedError("Stokes parameters not yet implemented.")
        elif reflected_pol_basis == 'waveplate':
            svec_inp = -svec[:, ii]
            pvec_inp = np.array([np.cos(2*np.pi*ii/nr+grating_angle),
                                 np.sin(2*np.pi*ii/nr+grating_angle),
                                 0.])

            """
            The action of the waveplate is to induce a phase shift phi between
            the fast and slow axes of a beam.  We consider the waveplate as
            doing all the action on the input beam prior to reflection, rather
            than the complicated procedure of phase shifting, then reflecting
            then phase shifting again differently because of the angle.  Note
            that the slow and fast axes are defined relative to the input s and
            p:
            """
            slow_axis = pvec_inp*np.cos(reflected_pol[0]) +\
                        svec_inp*np.sin(reflected_pol[0])

            fast_axis = -pvec_inp*np.sin(reflected_pol[0]) +\
                        svec_inp*np.cos(reflected_pol[0])

            # Compute new polarization:
            pol_after_waveplate = \
            np.dot(pol, slow_axis)*np.exp(-1j*reflected_pol[1]/2)*slow_axis +\
            np.dot(pol, fast_axis)*np.exp(1j*reflected_pol[1]/2)*fast_axis

            # Reproject onto s and p:
            pol_ref = -pvec[:, ii]*np.dot(pol_after_waveplate, pvec_inp) +\
                svec[:, ii]*np.dot(pol_after_waveplate, svec_inp)
        else:
            raise NotImplementedError(
                "{0:s} polarization basis not implemented.".format(pol_basis)
                )

        #print(np.dot(pol,pvec_inp), np.dot(pol,svec_inp))
        #print(kvec, pol_ref, np.dot(kvec, pol_ref))

        beams.append(laserBeam(
            np.array([-np.sin(thd)*np.cos(2*np.pi*ii/nr+grating_angle),
                      -np.sin(thd)*np.sin(2*np.pi*ii/nr+grating_angle),
                      -np.cos(thd)]),
            beta=factory(ii),
            pol=pol_ref, delta=delta))

    if return_basis_vectors:
        return (laserBeams(beams), pvec, svec)
    return laserBeams(beams)

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
