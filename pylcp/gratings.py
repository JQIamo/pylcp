import numpy as np
from pylcp.fields import laserBeams, infinitePlaneWaveBeam, clippedGaussianBeam
import numba

@numba.njit
def reflected_int_single_point(R, zgrating, back_project_x, back_project_y, cos_th_center,
                                sin_th_center, center_hole, th_lower, th_upper, r_stop, s_max, wb):
    # Calcualte the transformed location back at the grating:
    Rp = np.array([R[0] - (R[2]-zgrating)*back_project_x,
                   R[1] - (R[2]-zgrating)*back_project_y,
                   zgrating])

    THp = np.arctan2(Rp[1], Rp[0])
    Radp = np.sqrt(Rp[0]**2 + Rp[1]**2)

    # Are we behind the grating?
    if R[2] > zgrating:
        return 0.
    # Are we in the center hole?
    elif (Rp[0]*cos_th_center + Rp[1]*sin_th_center) < center_hole:
        return 0.
    # Are we in the limit?
    elif np.sqrt(Rp[0]**2+Rp[1]**2) > r_stop:
        return 0.
    # Are we in the right angular range?
    elif (th_upper < th_lower and (THp > th_upper and THp <= th_lower)):
        # We extend over the pi branch cut:
        return 0.
    elif (th_upper >= th_lower and (THp > th_upper or THp <= th_lower)):
        return 0.

    rho_sq = Rp[0]**2+Rp[1]**2

    return (s_max*np.exp(-2*rho_sq/wb**2))


@numba.njit
def input_int_single_point(R, zgrating, center_hole, cos_th_centers,
                            sin_th_centers, r_stop, s_max, wb):
    # Are we behind the grating?
    if R[2] > zgrating:
        # Is there a central hole?
        if center_hole>0:
            # Then check to see if we are within the hole.
            for (cos_th_center, sin_th_center) in zip(cos_th_centers, sin_th_centers):
                if (R[0]*cos_th_center + R[1]*sin_th_center) > center_hole:
                    return 0.
        else:
            return 0.

    # Still good? Check to make sure we are within the stop:
    rho_sq = R[0]**2+R[1]**2

    if np.sqrt(rho_sq) > r_stop:
        return 0.
    else:
        return s_max*np.exp(-2*rho_sq/wb**2)



class infiniteGratingMOTBeams(laserBeams):
    def __init__(self, delta=-1., s=1., nr=3, thd=np.pi/4,
                 pol=np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0]),
                 reflected_pol=np.array([np.pi, 0]),
                 reflected_pol_basis='poincare',
                 eta=None, grating_angle=0, input_phase=0.0, diff_phases=None, **kwargs):
        """
        Creates beams that would be made from a grating.
        Parameters:
            delta: detuning of the laser beams
            s: intensity of the laser beams
            nr: number of reflected beams
            thd: diffraction angle
            pol: input polarization.  Can be +1 or -1
            reflected_pol: two parameters that describe the reflected
            polarization depending on the reflected_pol_basis.
            reflected_pol_basis: the basis in which the reflection polarization is defined.
            There are three bases currently programmed:
                'poincare': the poincare basis is the Poincare sphere with
                sigma^+ at the north pole, sigma^- at the south pole, and s and
                p along the +x and -x axes, respectively. In this case,
                reflected_pol[0] is the polar angle (0 corresponding to sigma^-
                and pi corresponding to sigma^+) and reflected_pol[1] is the
                azimuthal angle (0 corresponding to p and pi corresponding to s)
                'jones_vector': relected_pol[0]*svec + reflected_pol[1]*pvec
                'waveplate': relected_pol[0] specified the angle of the slow
                axis relative to the p-vector.  reflection_pol[1] specifies the
                phase delay.
            eta: diffraction efficiency of each of the reflected beams
            grating_angle: overall azimuthal rotation of the grating
            input_phase: Phase of input beam.
            diff_phases: List or numpy array containing phases of diffracted
            beams.
        """

        # Turn on a bunch of stuff for making this laser beam collection:
        super().__init__()

        self.nr = nr
        self.thd = thd
        self.grating_angle = grating_angle

        if eta is None:
            self.eta = 1/nr
        else:
            self.eta = eta

        self.add_laser(infinitePlaneWaveBeam(kvec=np.array([0., 0., 1.]),
                                             pol=pol, s=s, delta=delta,
                                             pol_coord='cartesian', phase=input_phase, **kwargs))

        # Store the input polarization as a Carterian coordiante:
        self.input_pol = self.beam_vector[0].cartesian_pol()

        # Calculate the reflected polarizations and k-vectors:
        kvec_refs, pol_refs = self._calculate_reflected_kvecs_and_pol(
            reflected_pol, reflected_pol_basis
            )

        for ii in range(self.nr):
            self.add_laser(infinitePlaneWaveBeam(kvec=kvec_refs[:, ii],
                                                 pol=pol_refs[:, ii],
                                                 s=self.eta*s/np.cos(self.thd),
                                                 delta=delta,
                                                 pol_coord='cartesian',
                                                 phase=diff_phases[ii],
                                                 **kwargs))

    def _calculate_reflected_kvecs_and_pol(self, reflected_pol,
                                           reflected_pol_basis):
        # Preallocate memory for the polarizations (no need to store kvec or the
        # polarization because those are stored in the laser)
        kvec = np.zeros((3, self.nr))
        svec = np.zeros((3, self.nr))
        pvec = np.zeros((3, self.nr))
        pol_ref = np.zeros((3, self.nr), dtype=np.complex128)

        for ii in range(self.nr):  # Reflected beams
            kvec[:, ii] = np.array([-np.sin(self.thd)
                                    *np.cos(2*np.pi*ii/self.nr
                                            +self.grating_angle),
                                    -np.sin(self.thd)
                                    *np.sin(2*np.pi*ii/self.nr
                                            +self.grating_angle),
                                    -np.cos(self.thd)])
            svec[:, ii] = np.array([-np.sin(2*np.pi*ii/self.nr+
                                            self.grating_angle),
                                    np.cos(2*np.pi*ii/self.nr+
                                           self.grating_angle),
                                    0.])
            pvec[:, ii] = np.array([np.cos(self.thd)
                                    *np.cos(2*np.pi*ii/self.nr
                                            +self.grating_angle),
                                    np.cos(self.thd)
                                    *np.sin(2*np.pi*ii/self.nr
                                            +self.grating_angle),
                                    -np.sin(self.thd)])

            if reflected_pol_basis == 'poincare':
                pol_ref[:, ii] = np.sin(reflected_pol[0]/2)*np.exp(1j*reflected_pol[1]/2)*\
                    (svec[:, ii]/np.sqrt(2) + 1j/np.sqrt(2)*pvec[:, ii]) +\
                    np.cos(reflected_pol[0]/2)*np.exp(-1j*reflected_pol[1]/2)*\
                    (svec[:, ii]/np.sqrt(2) - 1j/np.sqrt(2)*pvec[:, ii])
            elif reflected_pol_basis == 'jones_vector':
                pol_ref[:, ii] = (reflected_pol[0]*svec[:, ii] +
                                  reflected_pol[1]*pvec[:, ii])
            elif reflected_pol_basis == 'stokes_parameters':
                raise NotImplementedError("Stokes parameters not yet implemented.")
            elif reflected_pol_basis == 'waveplate':
                svec_inp = -svec[:, ii]
                pvec_inp = np.array([np.cos(2*np.pi*ii/self.nr
                                            +self.grating_angle),
                                     np.sin(2*np.pi*ii/self.nr
                                            +self.grating_angle),
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
                    np.dot(self.input_pol, slow_axis) \
                    *np.exp(-1j*reflected_pol[1]/2)*slow_axis \
                    +np.dot(self.input_pol, fast_axis) \
                    *np.exp(1j*reflected_pol[1]/2)*fast_axis

                # Reproject onto s and p:
                pol_ref[:, ii] = -pvec[:, ii]*np.dot(pol_after_waveplate, pvec_inp) \
                                 +svec[:, ii]*np.dot(pol_after_waveplate, svec_inp)
            else:
                raise NotImplementedError(
                    "{0:s} polarization basis not implemented.".format(reflected_pol_basis)
                )

            self.svec = svec
            self.pvec = pvec

        return kvec, pol_ref

    def jones_vector(self, R=np.array([0., 0., 0.]), t=0):
        output = []
        for ii, beam in enumerate(self.beam_vector):
            if (ii%(self.nr+1))>0:
                output.append(beam.jones_vector(
                    self.svec[:, ii%(self.nr+1)-1],
                    self.pvec[:, ii%(self.nr+1)-1],
                    R=R, t=t))
            else:
                output.append(beam.jones_vector(
                    np.array([1., 0, 0]),
                    np.array([0, 1., 0]),
                    R=R, t=t))

        return output

    def stokes_parameters(self, R=np.array([0., 0., 0.]), t=0):
        output = []
        for ii, beam in enumerate(self.beam_vector):
            if (ii%(self.nr+1))>0:
                output.append(beam.stokes_parameters(
                    self.svec[:, ii%(self.nr+1)-1],
                    self.pvec[:, ii%(self.nr+1)-1],
                    R=R, t=t))
            else:
                output.append(beam.stokes_parameters(
                    np.array([1., 0, 0]),
                    np.array([0, 1., 0]),
                    R=R, t=t))

        return output

    def polarization_ellipse(self, R=np.array([0., 0., 0.]), t=0):
        output = []
        for ii, beam in enumerate(self.beam_vector):
            if (ii%(self.nr+1))>0:
                output.append(beam.polarization_ellipse(
                    self.svec[:, ii%(self.nr+1)-1],
                    self.pvec[:, ii%(self.nr+1)-1],
                    R=R, t=t))
            else:
                output.append(beam.polarization_ellipse(
                    np.array([1., 0, 0]),
                    np.array([0, 1., 0]),
                    R=R, t=t))

        return output

class inputGaussianBeam(clippedGaussianBeam):
    def __init__(self, kvec=np.array([0, 0, 1]), s=1., delta=-1.,
                 pol=np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0]),
                 pol_coord='cartesian', wb=1., rs=2., nr=3, center_hole=0.0,
                 zgrating=1.0, grating_angle=0, **kwargs):

        if not np.allclose(kvec[:2], 0.):
            raise ValueError('inputGaussianBeam must be aligned with z axis')

        self.center_hole = center_hole # inscribed radius of center hole.
        self.zgrating = zgrating # z position of the diffraction grating chip.
        self.grating_angle = grating_angle # azimuthal rotation of the grating.
        self.nb = nr # number of reflected beams.

        # Determine the center angle of each grating section:
        self.th_center = (2*np.pi*np.arange(0, nr)/nr)+grating_angle
        self.cos_th_center = np.cos(self.th_center)
        self.sin_th_center = np.sin(self.th_center)

        super().__init__(kvec=kvec, pol=pol, pol_coord=pol_coord, s=s,
                         delta=delta, wb=wb, rs=rs, **kwargs)

    def intensity(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Masks the intensity profile of the input beam after it
        goes through the chip.
        """
        if R.shape == (3,):
            # Single point:
            return input_int_single_point(
                R, self.zgrating, self.center_hole,
                self.cos_th_center, self.sin_th_center,
                self.rs, self.s_max, self.wb
            )
        else:
            # Initialize mask:
            if isinstance(R[0], np.ndarray):
                MASK = np.ones(R[0].shape, dtype=bool)
            else:
                MASK = True
            # Add in the center hole:
            for (cos_th_center_i, sin_th_center_i) in zip(self.cos_th_center, self.sin_th_center):
                MASK = np.bitwise_and(MASK, (R[0]*cos_th_center_i +
                                             R[1]*sin_th_center_i) <= self.center_hole)
            # Make sure that the mask only applies after the chip.
            MASK = np.bitwise_or(MASK, R[2] <= self.zgrating)

            # Compute radial distance:
            rho_sq = R[0]**2+R[1]**2

            # Next, calculate the S function:
            S = (self.s_max
                    *np.exp(-2*rho_sq/self.wb**2)
                    *(np.sqrt(rho_sq) < self.rs)
                    *MASK.astype(float))

            return S


class reflectedGaussianBeam(clippedGaussianBeam):
    def __init__(self, kvec=np.array([-1/np.sqrt(2), 0, -1/np.sqrt(2)]),
                 s=1., delta=-1.,
                 pol=np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0]),
                 pol_coord='cartesian', wb=1., rs=2., nr=3, ii=0, thd=np.pi/4,
                 kvec_in=np.array([0, 0, 1.]), eta=1/3, center_hole=0.0,
                 outer_radius=10., zgrating=1.0, grating_angle=0, **kwargs):

        if not np.allclose(kvec_in[:2], 0.):
            raise ValueError('inputGaussianBeam must be aligned with z axis')

        self.center_hole = center_hole # inscribed radius of center hole.
        self.outer_radius = outer_radius # outer radius of the grating.
        self.zgrating = zgrating # z position of the diffraction grating chip.
        self.grating_angle = grating_angle # azimuthal rotation of the grating.
        self.nb = nr # number of reflected beams.
        #self.ii = ii # beam reflects from the iith grating.
        self.thd = thd # 1st order diffraction angle.
        self.k_in = kvec_in # kvec of the input beam.
        self.eta = eta # 1st order diffraction efficiency.

        self.r_stop = np.amin([rs, outer_radius])

        # Determine the center angle, grating section, and hole parameters:
        self.th_center = (2*np.pi*ii/nr)+grating_angle
        self.th_lower = self.wrap(self.th_center-np.pi/self.nb)
        self.th_upper = self.wrap(self.th_center+np.pi/self.nb)
        self.cos_th_center = np.cos(self.th_center)
        self.sin_th_center = np.sin(self.th_center)

        super().__init__(kvec=kvec, pol=pol, pol_coord=pol_coord, s=eta*s/np.cos(thd),
                         delta=delta, wb=wb, rs=rs, **kwargs)

        self.back_project_x = self.con_kvec[0]/self.con_kvec[2]
        self.back_project_y = self.con_kvec[1]/self.con_kvec[2]


    def wrap(self, ang):
        return (ang + np.pi) % (2 * np.pi) - np.pi

    def back_project(self, R):
        if R.shape == (3,):
            Rp = np.array([R[0] - (R[2]-self.zgrating)*self.back_project_x,
                           R[1] - (R[2]-self.zgrating)*self.back_project_y,
                           self.zgrating])
        else:
            Rp = np.array([R[0] - (R[2]-self.zgrating)*self.back_project_x,
                           R[1] - (R[2]-self.zgrating)*self.back_project_y,
                           np.ones(R[2].shape)*self.zgrating])

        return Rp

    def mask(self, Rp, R):
        THp = np.arctan2(Rp[1], Rp[0])
        Radp = np.sqrt(Rp[0]**2 + Rp[1]**2)

        if R.shape == (3,):
            # Are we behind the grating?
            if R[2] > self.zgrating:
                return 0.
            # Are we in the center hole?
            elif (Rp[0]*self.cos_th_center + Rp[1]*self.sin_th_center) < self.center_hole:
                return 0.
            elif (np.sqrt(Rp[0]**2+Rp[1]**2) > self.r_stop):
                return 0.
            # Are we in the right angular range?
            elif (self.th_upper < self.th_lower and
                  (THp > self.th_upper and THp < self.th_lower)):
                # We extend over the pi branch cut.
                return 0.
            elif (self.th_upper >= self.th_lower and
                  (THp > self.th_upper or THp < self.th_lower)):
                return 0.
            else:
                return 1.
        else:
            if self.th_upper < self.th_lower: # We extend over the pi branch cut:
                MASK = np.bitwise_or(THp < self.th_upper, THp > self.th_lower)
            else:
                MASK = np.bitwise_and(THp < self.th_upper, THp > self.th_lower)
            MASK = np.bitwise_and(MASK, Radp < self.r_stop)
            MASK = np.bitwise_and(MASK, R[2] <= self.zgrating)
            # Add in the center hole:
            MASK = np.bitwise_and(MASK, ((Rp[0]*np.cos(self.th_center)
                                         + Rp[1]*np.sin(self.th_center))
                                         >= self.center_hole))

            return MASK.astype('float')

    def intensity(self, R=np.array([0., 0., 0.]), t=0.):
        if R.shape==(3,):
            return reflected_int_single_point(
                R, self.zgrating, self.back_project_x,
                self.back_project_y, self.cos_th_center, self.sin_th_center,
                self.center_hole, self.th_lower, self.th_upper,
                self.r_stop, self.s_max, self.wb)
        else:
            # Make a primed coordinate system that translates X,Y,Z positions back
            # down to the grating plane.
            Rp = self.back_project(R)

            # Define the mask:
            MASK = self.mask(Rp, R)

            # Compute radial distance:
            rho_sq = Rp[0]**2 + Rp[1]**2

            # Next, calculate the S function:
            S = (self.s_max*np.exp(-2*rho_sq/self.wb**2)*MASK)

            return S

class maskedGaussianGratingMOTBeams(infiniteGratingMOTBeams):
    def __init__(self, delta=-1., s=1., nr=3, thd=np.pi/4,
                 pol=np.array([-1/np.sqrt(2), 1j/np.sqrt(2), 0]),
                 reflected_pol=np.array([np.pi, 0]),
                 reflected_pol_basis='poincare',
                 eta=None,
                 eta0=None,
                 wb=10.0,
                 rs=10.0,
                 center_hole=0.0,
                 outer_radius=10.0,
                 zgrating=1.0,
                 grating_angle=0,
                 input_phase=0.0,
                 diff_phases=None,
                 **kwargs):
        """
        Creates beams that would be made from a grating.
        Parameters:
            delta: detuning of the laser beams
            s: intensity of the laser beams
            nr: number of reflected beams
            thd: diffraction angle
            pol: input polarization.  Can be +1 or -1
            reflected_pol: two parameters that describe the reflected
            polarization depending on the reflected_pol_basis.
            reflected_pol_basis: the basis in which the reflection polarization
            is defined.

                There are three bases currently programmed:
                'poincare': the poincare basis is the Poincare sphere with
                sigma^+ at the north pole, sigma^- at the south pole, and s and
                p along the +x and -x axes, respectively. In this case,
                reflected_pol[0] is the polar angle (0 corresponding to
                sigma^- and pi corresponding to sigma^+) and reflected_pol[1]
                is the azimuthal angle (0 corresponding to p and pi
                corresponding to s)
                'jones_vector': relected_pol[0]*svec + reflected_pol[1]*pvec
                'waveplate': relected_pol[0] specified the angle of the slow
                axis relative to the p-vector. Reflection_pol[1] specifies the
                phase delay.

            eta: diffraction efficiency of each of the reflected beams
            wb: 1/e^2 radius of the INPUT gaussian beam.
            rs: radius of the INPUT beam stop.
            center_hole: inscribed radius of center hole.
            outer_radius: outer radius of the diffraction gratings.
            zgrating: z position of the diffraction grating chip.
            grating_angle: overall azimuthal rotation of the grating.
            input_phase: Phase of input beam.
            diff_phase: List or numpy array containing phases of diffracted
            beams.
        """
        # I would like use to do super().super().__init__(), but that does not work.
        # Nonetheless, these are the only two things that need to be done
        # from laserBeams.__init__():
        self.beam_vector = []
        self.num_of_beams = 0

        self.nr = nr
        self.thd = thd
        self.grating_angle = grating_angle

        if eta is None:
            self.eta = 1/nr
        else:
            self.eta = eta

        if diff_phases is None:
            diff_phases = np.zeros(nr)

        self.add_laser(inputGaussianBeam(kvec=np.array([0., 0., 1.]),
                                         pol=pol, s=s, delta=delta,
                                         pol_coord='cartesian', wb=wb, rs=rs,
                                         nr=self.nr, center_hole=center_hole,
                                         zgrating=zgrating,
                                         grating_angle=self.grating_angle,
                                         phase=input_phase,
                                         **kwargs))

        # Store the input polarization as a Carterian coordiante:
        self.input_pol = self.beam_vector[0].cartesian_pol()

        # Calculate the reflected polarizations and k-vectors:
        kvec_refs, pol_refs = self._calculate_reflected_kvecs_and_pol(
            reflected_pol, reflected_pol_basis
            )

        for ii in range(self.nr):
            self.add_laser(reflectedGaussianBeam(kvec=kvec_refs[:, ii],
                                                 pol=pol_refs[:, ii],
                                                 s=s, delta=delta,
                                                 pol_coord='cartesian', wb=wb,
                                                 rs=rs, nr=self.nr, ii=ii,
                                                 thd=self.thd,
                                                 kvec_in=np.array([0., 0., 1.]),
                                                 eta=self.eta,
                                                 center_hole=center_hole,
                                                 outer_radius=outer_radius,
                                                 zgrating=zgrating,
                                                 grating_angle=self.grating_angle,
                                                 phase=diff_phases[ii],
                                                 **kwargs))

        if eta0 is not None:
            raise NotImplementedError("Zeroth-order reflected beam has not been implemented yet.")
            # Should be free to choose s and p directions for normal incidence.
            # Make choice to match convention for 1st order beams.
            svec0 = np.array([0., 1., 0.])
            pvec0 = np.array([1., 0., 0.])
            if reflected_pol_basis == 'poincare':
                pol_0 = (np.sin(reflected_pol[0]/2)
                         *np.exp(1j*reflected_pol[1]/2)
                         *(svec0/np.sqrt(2) + 1j/np.sqrt(2)*pvec0)
                         +np.cos(reflected_pol[0]/2)
                         *np.exp(-1j*reflected_pol[1]/2)
                         *(svec0/np.sqrt(2) - 1j/np.sqrt(2)*pvec0))
            else:
                raise NotImplementedError("Only Poincare basis is implemented for the zeroth order beam.")
            # add zeroth order later.
