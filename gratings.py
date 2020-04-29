import numpy as np
from .fields import laserBeam, laserBeams, infinitePlaneWaveBeam, clippedGaussianBeam

class infiniteGratingMOTBeams(laserBeams):
    def __init__(self, delta, s, nr, thd, pol=np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0]),
                 reflected_pol=np.array([np.pi, 0]), reflected_pol_basis='poincare',
                 eta=None, return_basis_vectors=False, grating_angle=0):
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

        # Turn on a bunch of stuff for making this laser beam collection:
        super().__init__()

        self.nr = nr
        self.thd = thd
        self.grating_angle = grating_angle

        if not eta:
            self.eta = 1/nr

        if not isinstance(pol, np.ndarray) and not isinstance(pol, list):
            if pol == 1:
                pol = np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0])
            elif pol == -1:
                pol = np.array([-1/np.sqrt(2), -1j/np.sqrt(2), 0])
            else:
                raise ValueError('pol must be a three-vector or +/-1.')

        beams = []
        self.add_beam(infinitePlaneWaveBeam(np.array([0., 0., 1.]), pol,
                                            s, delta))

            #print(np.dot(pol,pvec_inp), np.dot(pol,svec_inp))
            #print(kvec, pol_ref, np.dot(kvec, pol_ref))

        # Calculate the reflected polarizations and k-vectors:
        kvec_refs, pol_refs, svec, pvec = self.__calculate_reflected_kvecs_and_pol()

        for kvec_ref, pol_ref in kvec_refs, pol_refs:
            self.add_beam(infinitePlaneWaveBeam(kvec_ref, pol_ref,
                                                eta*s/np.cos(thd), delta))

        self.pvec = pvec
        self.svec = svec


    def __calculate_reflected_kvecs_and_pol(self, reflected_pol, reflected_pol_basis):
        # Preallocate memory for the polarizations (no need to store kvec or the
        # polarization because those are stored in the laser)
        svec = np.zeros((3, self.nr))
        pvec = np.zeros((3, self.nr))

        for ii in range(nr):  # Reflected beams
            kvec = np.array([-np.sin(self.thd)*np.cos(2*np.pi*ii/self.nr
                                                      +self.grating_angle),
                             -np.sin(self.thd)*np.sin(2*np.pi*ii/self.nr+
                                                      self.grating_angle),
                             -np.cos(self.thd)])
            svec[:, ii] = np.array([-np.sin(2*np.pi*ii/self.nr+
                                            self.grating_angle),
                                     np.cos(2*np.pi*ii/self.nr+
                                            self.grating_angle),
                                     0.])
            pvec[:, ii] = np.array([np.cos(self.thd)*np.cos(2*np.pi*ii/nr+
                                                            self.grating_angle),
                                    np.cos(self.thd)*np.sin(2*np.pi*ii/nr+
                                                            self.grating_angle),
                                    -np.sin(self.thd)])


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
                pvec_inp = np.array([np.cos(2*np.pi*ii/nr+self.grating_angle),
                                     np.sin(2*np.pi*ii/nr+self.grating_angle),
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

        return kvec, pol_ref, svec, pvec


class inputGaussianBeam(clippedGaussianBeam):
    def __init__(self, center_hole=0.0, zgrating=1.0, grating_angle=0):

        self.center_hole = kwargs.pop('center_hole', 0.0)
        self.nb = kwargs.pop('nb', 3)

        self.k, s, wb, rs):

        __
    # Helper Functions for grating_MOT_beams:
    def beta(self, R, t):
        """
        Masks the intensity profile of the input beam after it
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
            raise ValueError('chip_masked_beam must be aligned with z axis')
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


class reflectedGaussianBeam(clippedGaussianBeam):
    def __init__(self):
        pass

    def __grating_reflected_beam(R, k, ii, nr, s, eta, thd, k_in, wb, rs,
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

class maskedGaussianGratingMOTBeams(gratingMOTBeams):
    def __init__(self, delta, s, nr, thd,
                 pol=np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0]),
                 reflected_pol=np.array([np.pi, 0]),
                 reflected_pol_basis='poincare',
                 eta=None,
                 eta0=None,
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


        # Calculate the reflected polarizations and k-vectors:
        kvec_refs, pol_refs, svec, pvec = self.__calculate_reflected_kvecs_and_pol()

        k_in = np.array([0., 0., 1.])
        beams = []
        beams.append(gaussianBeam(k_in,
                                  beta=lambda R,k: chip_masked_beam(R,k,nr,s,wb,rs,
                                                                 center_hole,
                                                                 zgrating,
                                                                 grating_angle),
                               pol=pol,
                               delta=delta))  # Incident beam


        if eta0 is not None:
            # Should be free to choose s and p directions for normal incidence.
            # Make choice to match convention for 1st order beams.
            svec0=np.array([0.,1.,0.])
            pvec0=np.array([1.,0.,0.])
            if reflected_pol_basis == 'poincare':
                pol_0 = np.sin(reflected_pol[0]/2)*np.exp(1j*reflected_pol[1]/2)*\
                          (svec0/np.sqrt(2) + 1j/np.sqrt(2)*pvec0) +\
                          np.cos(reflected_pol[0]/2)*np.exp(-1j*reflected_pol[1]/2)*\
                          (svec0/np.sqrt(2) - 1j/np.sqrt(2)*pvec0)
            else:
                raise NotImplementedError("Only Poincare basis is implemented for the zeroth order beam.")
            beams.append(laserBeam(-k_in,
                                   beta=lambda R,k: chip_masked_reflected_beam(R,k,
                                                                   nr,s,wb,rs,eta0,
                                                                   center_hole,
                                                                   zgrating,
                                                                   grating_angle),
                                   pol=pol_0,
                                   delta=delta))  # Zeroth order reflected beam

        def factory(ii):
            return lambda R, k: grating_reflected_beam(R, k, ii, nr, s, eta, thd,
                                                       k_in, wb, rs, center_hole,
                                                       outer_radius, zgrating,
                                                       grating_angle)
