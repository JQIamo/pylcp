import numpy as np
from inspect import signature
from .common import cart2spherical, spherical2cart

import numba

@numba.jit(nopython=True)
def dot2D(a, b):
    c = np.zeros((a.shape[1],), dtype=a.dtype)
    for ii in range(a.shape[1]):
        c[ii] = np.sum(a[:, ii]*b[:, ii])
    return c

@numba.jit(nopython=True)
def speedy_E_field_evaluator(r, t, kvecs, betas, pols, deltas, phases,
                             mean_detuning):
    return np.dot(pols.T,
                  np.sqrt(betas/2)*np.exp(- 1j*np.dot(kvecs, r)
                                          + 1j*(deltas - mean_detuning)*t
                                          - 1j*phases),
                 )

#@numba.jit(nopython=True)
def speedy_delE_field_evaluator(r, t, kvecs, betas, pols, deltas, phases,
                                mean_detuning):
    delE = np.zeros((3, 3), dtype='complex128')
    for (kvec, beta, pol, delta, phase) in zip(kvecs, betas, pols, deltas, phases):
        delE += np.multiply(kvec.reshape(3, 1), pol.reshape(1, 3))*\
                1j*np.sqrt(beta/2)*np.exp(-1j*np.dot(kvec, r) + 1j*delta*t +
                                          -1j*phase)

    return delE


# First, define the laser beam class:
class laserBeam():
    def __init__(self, kvec=np.array([1]), beta = 1,
                 pol = 1, delta = -2, pol_coord='cartesian',
                 phase = 0.):

        def _test_callable_function(func, str1):
            """
            A small, nest functioned  to see what the function gives at a random point:
            """
            vec_test = func(np.random.random((3,)))
            if not np.isclose(np.linalg.norm(vec_test), 1):
                raise ValueError(str1 + 'function does not return unit vector.')

        # Handle the kvec:
        if not callable(kvec):
            self.kvec = np.array(kvec)
            self.kvec_sig = None
        else:
            _test_callable_function(kvec, 'kvec')
            self.kvec = kvec
            self.kvec_sig = str(signature(self.kvec))

        if not callable(beta):
            self.beta = beta
            self.beta_sig = None
        else:
            self.beta = beta
            self.beta_sig = str(signature(self.beta))

        if not callable(delta):
            self.delta = delta
            self.delta_sig = None
        else:
            self.delta = delta
            self.delta_sig = str(signature(self.delta))

        if not callable(pol):
            if isinstance(pol, float) or isinstance(pol, int):
                """
                If the polarization is defined by just a single number (+/-1),
                we assume that the polarization is defined as sigma^+ or sigma^-
                using the k-vector of the light as the axis defining z.  In this
                case, we want to project onto the actual z axis, which is
                relatively simple as there is only one angle.
                """
                self.pol_sig = None # Polarization is constant in this config.

                # Set the polarization in this direction:
                if np.sign(pol)<0:
                    self.pol = np.array([1., 0., 0.], dtype='complex')
                else:
                    self.pol = np.array([0., 0., 1.], dtype='complex')

                # Project onto the actual k-vector:
                self.pol = self.project_pol(self.kvec, invert=True).astype('complex128')
            elif isinstance(pol, np.ndarray):
                if pol.shape != (3,):
                    raise ValueError("pol, when a vector, must be a (3,) array")

                # The user has specified a single polarization vector in
                # cartesian coordinates:
                if pol_coord=='cartesian':
                    # Check for transverseness:
                    if np.abs(np.dot(self.kvec, pol)) > 1e-9:
                        raise ValueError("I'm sorry; light is a transverse wave")

                    # Always store in spherical basis.  Matches Brown and Carrington.
                    self.pol = cart2spherical(pol).astype('complex128')

                # The user has specified a single polarization vector in
                # spherical basis:
                elif pol_coord=='spherical':
                    pol_cart = spherical2cart(pol)

                    # Check for transverseness:
                    if np.abs(np.dot(self.kvec, pol_cart)) > 1e-9:
                        raise ValueError("I'm sorry; light is a transverse wave")

                    # Save the variable:
                    self.pol = pol.astype('complex128')

                # Finally, normalize
                self.pol = self.pol/np.linalg.norm(self.pol)
            else:
                raise ValueError("pol must be +1, -1, or a numpy array")

            self.pol_sig = None
        else:
            # 'Tis a function, so let's save the function:
            self.pol = pol
            self.pol_sig = str(signature(self.pol))

        self.phase = phase


    def check_return_inputs(self, R, t):
        if isinstance(t, list) and not isinstance(t, np.ndarray):
            t = np.array(t)

        if isinstance(R, list) and not isinstance(R, np.ndarray):
            R = np.array(R)

        if isinstance(t, np.ndarray) and t.size>1:
            if R.shape[1::] != t.shape:
                raise TypeError(
                    'Cannot broadcast R with shape %s' % (R.shape,) +
                    'and t with shape %s together.' % (t.shape,)
                )
            else:
                return t.shape
        elif R.shape == (3,):
            return (1,)
        else:
            raise TypeError(
                    'Cannot broadcast R with shape %s' % (R.shape,) +
                    'and t with type %s together.' % type(t)
                )

    def return_kvec(self, R=np.array([0., 0., 0.]), t=0.):
        return_shape = self.check_return_inputs(R, t)
        if self.kvec_sig is None:
            if return_shape != (1,):
                return np.multiply.outer(self.kvec, np.ones(return_shape))
            else:
                return self.kvec
        elif ('(R)' in self.kvec_sig or '(r)' in self.kvec_sig or
              '(x)' in self.kvec_sig):
            return self.kvec(R)
        elif ('(R, t)' in self.kvec_sig or '(r, t)' in self.kvec_sig or
              '(x, t)' in self.kvec_sig):
            return self.kvec(R, t)
        elif self.kvec_sig is '(t)':
            return self.kvec(t)
        else:
            raise ValueError('kvec function must have the form (R), (R, t) ' +
                             'or (t) for arguments.')


    def return_beta(self, R=np.array([0., 0., 0.]), t=0.):
        return_shape = self.check_return_inputs(R, t)
        if self.beta_sig is None:
            if return_shape != (1,):
                return self.beta*np.ones(return_shape)
            else:
                return self.beta
        elif ('(R)' in self.beta_sig or '(r)' in self.beta_sig
              or '(x)' in self.beta_sig):
            return self.beta(R)
        elif ('(R, k)' in self.beta_sig or '(r, k)' in self.beta_sig
              or '(x, k)' in self.beta_sig):
            k = self.return_kvec(R, t)
            return self.beta(R, k)
        elif ('(R, k, t)' in self.beta_sig or '(r, k, t)' in self.beta_sig
              or '(x, k, t)' in self.beta_sig):
            k = self.return_kvec(R, t)
            return self.beta(R, k, t)
        elif ('(R, t)' in self.beta_sig or '(r, t)' in self.beta_sig
              or '(x, t)' in self.beta_sig):
            return self.beta(R, t)
        elif self.beta_sig is '(t)':
            return self.beta(t)
        else:
            raise ValueError('beta function must have (R), (R, k), '+
                             '(R, k, t), (R, t), or (t) for arguments.')

    def return_detuning(self, t=0.):
        if self.delta_sig is None:
            if isinstance(t, np.ndarray):
                return self.delta*np.ones(t.shape)
            else:
                return self.delta
        elif self.delta_sig is '(t)':
            return self.delta(t)
        else:
            raise ValueError('delta function must have the form (t) for ' +
                             'arguments.')

    def return_pol(self, R=np.array([0., 0., 0.]), t=0.):
        return_shape = self.check_return_inputs(R, t)
        if self.pol_sig is None:
            if return_shape != (1,):
                return np.outer(self.pol, np.ones(return_shape))
            else:
                return self.pol
        elif ('(R, t)' in self.pol_sig or '(r, t)' in self.pol_sig or
              '(x, t)' in self.pol_sig):
            return self.pol(R, t)
        elif ('(R)' in self.pol_sig or '(r)' in self.pol_sig or
              '(x)' in self.pol_sig):
            return self.pol(R)
        elif '(t)' in self.pol_sig:
            return self.pol(t)
        else:
            raise ValueError('pol function must have the form (R), (R, t) ' +
                             'or (t) for arguments.')


    def project_pol(self, quant_axis, R=np.array([0., 0., 0.]), t=0,
                    treat_nans=False, calculate_norm=False, invert=False):
        """
        This method projects the polarization of the laser onto the quantization
        axis.  The polarization is stored in the Wigner spherical basis.  Takes
        two arguments:
            quant_axis: A normalized 3-vector of the quantization axis direction.
            R: if polarization is a function (dependent on space), R is the
            3-vectors at which the polarization shall be calculated.
            calculate_norm: renormalizes the quant_axis.
            treat_nans: every place that nan is encoutnered, replace with the
            hat{z} axis as the quantization axis.
        """

        """
        First, return the polarization at the desired R and t.
        """
        pol = self.return_pol(R, t)

        """
        Second, check the quanitization axis if specified by the user.  The fun
        thing here is that we may only need to do this once, since it should
        overwrite the original variable with the improper quant_axis with the
        proper one.
        """
        if calculate_norm:
            quant_axis2 = np.zeros(quant_axis.shape)
            quant_axis[2] = 1.0  # Make the third entry all ones.
            quant_axis_norm = np.linalg.norm(quant_axis, axis=0)
            for ii in range(3):
                quant_axis2[ii][quant_axis_norm!=0] = \
                    quant_axis2[ii][quant_axis_norm!=0]/\
                    quant_axis_norm[quant_axis_norm!=0]
            quant_axis=quant_axis2
        elif treat_nans:
            for ii in range(quant_axis.shape[0]):
                if ii<quant_axis.shape[0]-1:
                    quant_axis[ii][np.isnan(quant_axis[-1])] = 0.0
                else:
                    quant_axis[ii][np.isnan(quant_axis[-1])] = 1.0

        """
        To project the full three-vector, we want to determine the Euler
        angles alpha, beta, and gamma that rotate the z-axis into the
        quantization axis.  The final Euler angle, gamma, only sets the
        phase of the -1 to +1 component, so it does not have any physical
        meaning for the rate equations (where we expect this method to
        mostly be used.)  Thus, we just set that angle equal to zero here.
        """
        cosbeta = quant_axis[2]
        sinbeta = np.sqrt(1-cosbeta**2)
        if isinstance(cosbeta, float):
            if np.abs(cosbeta)<1:
                gamma = np.arctan2(quant_axis[1], quant_axis[0])
            else:
                gamma = 0
            alpha = 0
        else:
            gamma = np.zeros(cosbeta.shape)
            inds = np.abs(quant_axis[2])<1
            gamma[inds] = np.arctan2(quant_axis[1][inds],
                                         quant_axis[0][inds])
            alpha = np.zeros(cosbeta.shape)

        quant_axis = quant_axis.astype('float64')
        pol = pol.astype('complex128')

        D = np.array([
            [(1+cosbeta)/2*np.exp(-1j*alpha + 1j*gamma),
             -sinbeta/np.sqrt(2)*np.exp(-1j*alpha),
             (1-cosbeta)/2*np.exp(-1j*alpha - 1j*gamma)],
            [sinbeta/np.sqrt(2)*np.exp(1j*gamma),
             cosbeta,
             -sinbeta/np.sqrt(2)*np.exp(-1j*gamma)],
            [(1-cosbeta)/2*np.exp(1j*alpha+1j*gamma),
             sinbeta/np.sqrt(2),
             (1+cosbeta)/2*np.exp(1j*alpha-1j*gamma)]
             ])

        if invert:
            D = np.linalg.inv(D)

        # Tensordot is a cool function, it allow you to do
        # multiplication-sums across various axes.  Because D is a
        # 3\times3\times\cdots array and likewise pol is a 3\times\cdots
        # array, we matrix multiply against the 1st dimension of D and the
        # 0th dimension of pol.

        # TODO: This probably won't work in the case pol.shape=3\times\cdots
        # case
        rotated_pol = np.tensordot(D, pol, ([1],[0]))

        return rotated_pol


    def cartesian_pol(self, R=np.array([0., 0., 0.]), t=0):
        """
        Returns the polarization in Cartesian coordinates.
        """
        pol = self.return_pol(R, t)
        return spherical2cart(pol)

    def jones_vector(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        """
        Returns the 2-element Jones vector, using the orthogonal three-vectors
        xp and yp to define the two orthogoanl polarization axes.
        """
        # First, run some basic checks.
        if np.abs(np.dot(xp, yp)) > 1e-10:
            raise ValueError('xp and yp must be orthogonal.')
        if np.abs(np.dot(xp, self.kvec)) > 1e-10:
            raise ValueError('xp and k must be orthogonal.')
        if np.abs(np.dot(yp, self.kvec)) > 1e-10:
            raise ValueError('yp and k must be orthogonal.')
        if np.sum(np.abs(np.cross(xp, yp) - self.kvec)) > 1e-10:
            raise ValueError('xp, yp, and k must form a right-handed' +
                             'coordinate system.')

        pol_cart = self.cartesian_pol(R, t)

        if np.abs(np.dot(pol_cart, self.kvec)) > 1e-9:
            raise ValueError('Something is terribly, terribly wrong.')

        return np.array([np.dot(pol_cart, xp), np.dot(pol_cart, yp)])


    def stokes_parameters(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        """
        Returns the Stokes parameters.
        """
        jones_vector = self.jones_vector(xp, yp, R, t)

        Q = np.abs(jones_vector[0])**2 - np.abs(jones_vector[1])**2
        U = 2*np.real(jones_vector[0]*np.conj(jones_vector[1]))
        V = -2*np.imag(jones_vector[0]*np.conj(jones_vector[1]))

        return (Q, U, V)


    def polarization_ellipse(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        """
        Returns the polarization in terms of the polarization ellipse.
        """
        Q, U, V = self.stokes_parameters(xp, yp, R, t)

        psi = np.arctan2(U, Q)
        while psi<0:
            psi+=2*np.pi
        psi = psi%(2*np.pi)/2
        if np.sqrt(Q**2+U**2)>1e-10:
            chi = 0.5*np.arctan(V/np.sqrt(Q**2+U**2))
        else:
            chi = np.pi/4*np.sign(V)

        return (psi, chi)


    def return_parameters(self, R, t):
        if isinstance(t, float) or (isinstance(t, np.ndarray) and t.size==1):
            if callable(self.kvec):
                kvec = self.return_kvec(R, t)
            else:
                kvec = self.kvec

            if callable(self.beta):
                beta = self.return_beta(R, t)
            else:
                beta = self.beta

            if callable(self.pol):
                pol = self.return_pol(R, t)
            else:
                pol = self.pol

            if callable(self.delta):
                delta = self.return_detuning(t)
            else:
                delta = self.delta
        else:
            return_shape = self.check_return_inputs(R, t)
            if callable(self.kvec):
                kvec = self.return_kvec(R, t)
            else:
                kvec = np.outer(self.kvec,np.ones(return_shape))

            if callable(self.beta):
                beta = self.return_beta(R, t)
            else:
                beta = self.beta*np.ones(return_shape)

            if callable(self.pol):
                pol = self.return_pol(R, t)
            else:
                pol = np.outer(self.pol,np.ones(return_shape))

            if callable(self.delta):
                delta = self.return_detuning(t)
            else:
                delta = self.delta*np.ones(return_shape)

        return (kvec, beta, pol, delta)


    def electric_field(self, R, t, mean_detuning=0):
        """
        Returns the electric field of the laser beam at position R and time t.
        """
        (kvec, beta, pol, delta) = self.return_parameters(R, t)
        amp = np.sqrt(beta/2)

        if isinstance(t, float) or (isinstance(t, np.ndarray) and t.size==1):
            Eq = pol*amp*np.exp(1j*np.dot(kvec, R) - 1j*delta*t +
                                1j*self.phase)
        else:
            Eq = np.multiply(
                pol.reshape(3, t.size),
                amp*np.exp(1j*dot2D(kvec, R) - 1j*delta*t +
                           1j*self.phase)
            )

        return Eq


    def electric_field_gradient(self, R, t, mean_detuning=0):
        """
        Returns the gradient of the electric field of the laser beam at
        position R and time t.  TODO: include effects like gradient of
        amplitude,  kvec, or individual pol components.
        """
        (kvec, beta, pol, delta) = self.return_parameters(R, t)
        amp = np.sqrt(beta/2)

        if isinstance(t, float) or (isinstance(t, np.ndarray) and t.size==1):
            delE = np.multiply(kvec.reshape(3, 1), pol.reshape(1, 3))*\
                 1j*amp*np.exp(1j*np.dot(kvec, R) - 1j*delta*t +
                                          -1j*self.phase)
        else:
            delE = np.multiply(
                np.multiply(kvec.reshape(3, 1, t.size),
                            pol.reshape(1, 3, t.size)),
                1j*amp*np.exp(1j*dot2D(kvec, R) - 1j*delta*t +
                              1j*self.phase)
            )

        return delE



class laserBeams(laserBeam):
    """
    Class laserBeams is a collection of laserBeams.  It extends the
    functionality and makes
    """
    def __init__(self, laserbeamparams=None):
        if laserbeamparams is not None:
            if not isinstance(laserbeamparams, list):
                raise ValueError('laserbeamparams must be a list.')
            self.beam_vector = []
            for laserbeamparam in laserbeamparams:
                if isinstance(laserbeamparam, dict):
                    self.beam_vector.append(laserBeam(**laserbeamparam))
                elif isinstance(laserbeamparam, laserBeam):
                    self.beam_vector.append(laserbeamparam)
                else:
                    raise TypeError('Each element of laserbeamparams must either ' +
                                    'be a list of dictionaries or list of ' +
                                    'laserBeams')

            self.num_of_beams = len(self.beam_vector)
        else:
            self.beam_vector = []
            self.num_of_beams = 0

    def __iadd__(self, other):
        self.beam_vector += other.beam_vector
        self.num_of_beams = len(self.beam_vector)

        return self

    def __add__(self, other):
        return laserBeams(self.beam_vector + other.beam_vector)

    def add_laser(self, new_laser):
        if isinstance(new_laser, laserBeam):
            self.beam_vector.append(new_laser)
            self.num_of_beams = len(self.beam_vector)
        elif isinstance(new_laser, dict):
            self.beam_vector.append(laserBeam(**new_laser))
        else:
            raise TypeError('new_laser should by type laserBeam or a dictionary' +
                            'of arguments to initialize the laserBeam class.')

    def return_pol(self, r, t):
        return np.array([beam.return_pol(r, t) for beam in self.beam_vector])

    def return_beta(self, r, t):
        return np.array([beam.return_beta(r, t) for beam in self.beam_vector])

    def return_kvec(self, r, t):
        return np.array([beam.return_kvec(r, t) for beam in self.beam_vector])

    def return_detuning(self, t=0):
        return [beam.return_detuning(t) for beam in self.beam_vector]

    def return_parameters(self, r, t):
        kvecs = np.array([beam.return_kvec(r, t) for beam in self.beam_vector])
        betas = np.array([beam.return_beta(r, t) for beam in self.beam_vector])
        deltas = np.array([beam.return_detuning(t) for beam in self.beam_vector])
        pols = np.array([beam.return_pol(r, t) for beam in self.beam_vector])

        return (kvecs, betas, pols, deltas)

    def electric_field(self, r, t, mean_detuning=0):
        return np.array([beam.electric_field(r, t) for beam in self.beam_vector])

    def electric_field_gradient(self, r, t, mean_detuning=0):
        return np.array([beam.electric_field_gradient(r, t)
                         for beam in self.beam_vector])

    def total_electric_field(self, r, t, mean_detuning=0):
        return np.sum(self.electric_field(r, t, mean_detuning=mean_detuning),
                      axis=0)

    def total_electric_field_gradient(self, r, t, mean_detuning=0):
        return np.sum(self.electric_field_gradient(r, t,
                                                   mean_detuning=mean_detuning), axis=0)

    def randomize_laser_phases(self):
        for beam in self.beam_vector:
            beam.phase = 2*np.pi*np.random.random((1,))

    def project_pol(self, quant_axis, R=np.array([0., 0., 0.]), t=0, **kwargs):
        """
        To project the full three-vector, we want to determine the Euler
        angles alpha, beta, and gamma that rotate the z-axis into the
        quantization axis.  The final Euler angle, gamma, only sets the
        phase of the -1 to +1 component, so it does not have any physical
        meaning for the rate equations (where we expect this method to
        mostly be used.)  Thus, we just set that angle equal to zero here.
        """
        cosbeta = quant_axis[2]
        sinbeta = np.sqrt(1-cosbeta**2)
        if isinstance(cosbeta, float):
            if np.abs(cosbeta)<1:
                gamma = np.arctan2(quant_axis[1], quant_axis[0])
            else:
                gamma = 0
            alpha = 0
        else:
            gamma = np.zeros(cosbeta.shape)
            inds = np.abs(quant_axis[2])<1
            gamma[inds] = np.arctan2(quant_axis[1][inds],
                                         quant_axis[0][inds])
            alpha = np.zeros(cosbeta.shape)

        quant_axis = quant_axis.astype('float64')

        D = np.array([
            [(1+cosbeta)/2*np.exp(-1j*alpha + 1j*gamma),
             -sinbeta/np.sqrt(2)*np.exp(-1j*alpha),
             (1-cosbeta)/2*np.exp(-1j*alpha - 1j*gamma)],
            [sinbeta/np.sqrt(2)*np.exp(1j*gamma),
             cosbeta,
             -sinbeta/np.sqrt(2)*np.exp(-1j*gamma)],
            [(1-cosbeta)/2*np.exp(1j*alpha+1j*gamma),
             sinbeta/np.sqrt(2),
             (1+cosbeta)/2*np.exp(1j*alpha-1j*gamma)]
             ])

        return [np.tensordot(D, beam.return_pol(R, t), ([1],[0]))
                for beam in self.beam_vector]

    def cartesian_pol(self, R=np.array([0., 0., 0.]), t=0):
        return [beam.cartesian_pol(R, t) for beam in self.beam_vector]

    def jones_vector(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        return [beam.jones_vector(xp, yp, R, t) for beam in self.beam_vector]

    def stokes_parameters(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        return [beam.stokes_parameters(xp, yp, R, t) for beam in self.beam_vector]

    def polarization_ellipse(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        return [beam.polarization_ellipse(xp, yp, R, t) for beam in self.beam_vector]
