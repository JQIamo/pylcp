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

def return_constant_val(R, t, val):
    if R.shape==(3,):
        return val
    elif R.shape[0] == 3:
        return np.outer(val, np.ones(R[0].shape))
    else:
        raise ValueError('The first dimension of R should have length 3, ' +
                         'not %d.'%R.shape[0])

def return_constant_val_t(t, val):
    if isinstance(t, np.ndarray):
        return val*np.ones(t.shape)
    else:
        return val

def promote_to_lambda(val, var_name=None, type='Rt'):
    if type is 'Rt':
        if not callable(val):
            if isinstance(val, list):
                val = np.array(val)
            func = lambda R=np.array([0., 0., 0.]), t=0.: return_constant_val(R, t, val)
            sig = '()'
        else:
            sig = str(signature(val))
            if ('(R)' in sig or '(r)' in sig or '(x)' in sig):
                func = lambda R=np.array([0., 0., 0.]), t=0.: val(R)
                sig = '(R)'
            elif ('(R, t)' in sig or '(r, t)' in sig or '(x, t)' in sig):
                func = lambda R=np.array([0., 0., 0.]), t=0.: val(R, t)
                sig = '(R, t)'
            else:
                raise StandardError('Signature [%s] of function %s not'+
                                    'understood.'% (sig, var_name))

        return func, sig
    elif type is 't':
        if not callable(val):
            func = lambda t=0.: return_constant_val_t(t, val)
            sig = '()'
        else:
            sig = str(signature(val))
            if '(t)' in sig:
                func = lambda t=0.: val(t)
            else:
                raise StandardError('Signature [%s] of function %s not '+
                                    'understood.'% (sig, var_name))

        return func, sig

def return_dx_dy_dz(R, eps):
    if R.shape == (3,):
        dx = np.array([eps, 0., 0.])
        dy = np.array([0., eps, 0.])
        dz = np.array([0., 0., eps])
    else:
        dx = np.zeros(R.shape)
        dy = np.zeros(R.shape)
        dz = np.zeros(R.shape)

        dx[0] = eps
        dy[1] = eps
        dz[2] = eps

    return dx, dy, dz


class magField(object):
    """
    An object the calculates relevant properties of a static, magnetic field.
    """
    def __init__(self, func, FieldMag=None, gradFieldMag=None,
                 gradField=None, eps=1e-5):
        self.eps = eps

        R = np.random.rand(3) # Pick a random point for testing

        # Promote it to a lambda func:
        self.Field, self.FieldSig = promote_to_lambda(func, var_name='for field')

        # Try it out:
        response = self.Field(R, 0.)
        if (isinstance(response, float) or isinstance(response, int) or
            len(response) != 3):
            raise StandardError('Magnetic field function must return a vector.')

        # Next, we deal with the field magnitude:
        # Has it been specified by the user?
        if FieldMag is None:
            # No. Specify it using internal functions.
            self.FieldMagSig = self.FieldSig
            # Is it constant?
            if self.FieldMagSig is '()':
                # Calculate it once and promote to lambda function:
                self.FieldMag = promote_to_lambda(self.Field(R, 0.), var_name='for field magnitude')
            else:
                self.FieldMag = self.internal_FieldMag
        else:
            # Yes. Promote it to a lambda func:
            self.FieldMag, self.FieldMagSig = promote_to_lambda(func, var_name='for field magnitude')

        # Now, test it out and make sure it returns a float:
        response = self.FieldMag(R, 0.)
        if not (isinstance(response, float) or (isinstance(response, list) and len(response) != 1)):
            raise StandardError('Magnetic field magnitude function must ' +
                                'return a scalar.')

        # Next, we deal with the gradient of the field magnitude:
        # Has it been specified by the user?
        if gradFieldMag is None:
            # No. Specify it using internal functions.
            self.gradFieldMagSig = self.FieldMagSig
            # Is it constant?
            if self.gradFieldMagSig is '()':
                # Calculate it once and promote to lambda function:
                self.gradFieldMag = promote_to_lambda(np.array([0., 0., 0.]), var_name='for field magnitude')
            else:
                self.gradFieldMag = self.internal_gradFieldMag
        else:
            # Yes. Promote it to a lambda func:
            self.gradFieldMag, self.gradFieldMagSig = promote_to_lambda(FieldMag, var_name='for field magnitude')

        # Test it out:
        response = self.gradFieldMag(R, 0.)
        if (isinstance(response, float) or isinstance(response, int) or
            len(response) != 3):
            raise StandardError('gradFieldMag field magnitude function must ' +
                                'return a vector.')

        # Next, we deal with the gradient of the field magnitude:
        # Has it been specified by the user?
        if gradFieldMag is None:
            # No. Specify it using internal functions.
            self.gradFieldMagSig = self.FieldMagSig
            # Is it constant?
            if self.gradFieldMagSig is '()':
                # Calculate it once and promote to lambda function:
                self.gradFieldMag = promote_to_lambda(np.array([0., 0., 0.]), var_name='for field magnitude')
            else:
                self.gradFieldMag = self.internal_gradFieldMag
        else:
            # Yes. Promote it to a lambda func:
            self.gradFieldMag, self.gradFieldMagSig = promote_to_lambda(gradFieldMag, var_name='for field magnitude')

        response = self.gradFieldMag(R, 0.)
        if response.shape != (3, ):
            raise StandardError('Magnetic field gradient function must ' +
                                'return a matrix.')
            self.gradField= gradField

        # Next, we deal with the gradient of the field magnitude:
        # Has it been specified by the user?
        if gradField is None:
            # No. Specify it using internal functions.
            self.gradFieldSig = self.FieldSig
            # Is it constant?
            if self.gradFieldSig is '()':
                # Calculate it once and promote to lambda function:
                self.gradField = promote_to_lambda(np.zeros((3, 3)), var_name='for field magnitude')
            else:
                self.gradField = self.internal_gradField
        else:
            # Yes. Promote it to a lambda func:
            self.gradFieldMag, self.gradFieldMagSig = promote_to_lambda(gradField, var_name='for field magnitude')

        response = self.gradField(R, 0.)
        if response.shape != (3, 3):
            raise StandardError('Magnetic field gradient function must ' +
                                'return a matrix.')
            self.gradField= gradField


    def return_dx_dy_dz(self, R):
        if R.shape == (3,):
            dx = np.array([self.eps, 0., 0.])
            dy = np.array([0., self.eps, 0.])
            dz = np.array([0., 0., self.eps])
        else:
            dx = np.zeros(R.shape)
            dy = np.zeros(R.shape)
            dz = np.zeros(R.shape)

            dx[0] = self.eps
            dy[1] = self.eps
            dz[2] = self.eps

        return dx, dy, dz

    def internal_FieldMag(self, R=np.array([0., 0., 0.]), t=0):
        return np.linalg.norm(self.Field(R, t))

    def internal_gradFieldMag(self, R=np.array([0., 0., 0.]), t=0):
        dx, dy, dz = return_dx_dy_dz(R, self.eps)

        return np.array([
            (self.FieldMag(R+dx, t)-self.FieldMag(R-dx, t))/2/self.eps,
            (self.FieldMag(R+dy, t)-self.FieldMag(R-dy, t))/2/self.eps,
            (self.FieldMag(R+dz, t)-self.FieldMag(R-dz, t))/2/self.eps
            ])

    def internal_gradField(self, R=np.array([0., 0., 0.]), t=0):
        dx, dy, dz = return_dx_dy_dz(R, self.eps)

        return np.array([
            (self.Field(R+dx, t) - self.Field(R-dx, t))/2/self.eps,
            (self.Field(R+dy, t) - self.Field(R-dy, t))/2/self.eps,
            (self.Field(R+dz, t) - self.Field(R-dz, t))/2/self.eps
            ])


# First, define the laser beam class:
class laserBeam():
    def __init__(self, kvec=np.array([1., 0., 0.]), beta = 1.,
                 pol = 1., delta = 0., pol_coord='cartesian',
                 phase = 0., eps=1e-5):

        def _test_callable_function(func, str1):
            """
            A small, nest functioned  to see what the function gives at a random point:
            """
            vec_test = func(np.random.random((3,)))
            if not np.isclose(np.linalg.norm(vec_test), 1):
                raise ValueError(str1 + 'function does not return unit vector.')

        # Promote it to a lambda func:
        self.kvec, self.kvec_sig = promote_to_lambda(kvec, var_name='kvector')

        # Promote it to a lambda func:
        self.beta, self.beta_sig = promote_to_lambda(beta, var_name='beta')

        # Promote it to a lambda func:
        self.detuning, self.detuning_sig = promote_to_lambda(delta, var_name='detuning', type='t')

        if not callable(pol):
            if isinstance(pol, float) or isinstance(pol, int):
                """
                If the polarization is defined by just a single number (+/-1),
                we assume that the polarization is defined as sigma^+ or sigma^-
                using the k-vector of the light as the axis defining z.  In this
                case, we want to project onto the actual z axis, which is
                relatively simple as there is only one angle.
                """
                # Set the polarization in this direction:
                if np.sign(pol)<0:
                    self.pol = np.array([1., 0., 0.], dtype='complex')
                else:
                    self.pol = np.array([0., 0., 1.], dtype='complex')

                # Promote to lambda:
                self.pol, self.pol_sig = promote_to_lambda(self.pol, var_name='polarization')

                # Project onto the actual k-vector:
                self.pol = self.project_pol(self.kvec(), invert=True).astype('complex128')

            elif isinstance(pol, np.ndarray):
                if pol.shape != (3,):
                    raise ValueError("pol, when a vector, must be a (3,) array")

                # The user has specified a single polarization vector in
                # cartesian coordinates:
                if pol_coord=='cartesian':
                    # Check for transverseness:
                    if np.abs(np.dot(self.kvec(), pol)) > 1e-9:
                        raise ValueError("I'm sorry; light is a transverse wave")

                    # Always store in spherical basis.  Matches Brown and Carrington.
                    self.pol = cart2spherical(pol).astype('complex128')

                # The user has specified a single polarization vector in
                # spherical basis:
                elif pol_coord=='spherical':
                    pol_cart = spherical2cart(pol)

                    # Check for transverseness:
                    if np.abs(np.dot(self.kvec(), pol_cart)) > 1e-9:
                        raise ValueError("I'm sorry; light is a transverse wave")

                    # Save the variable:
                    self.pol = pol.astype('complex128')

                # Finally, normalize
                self.pol = self.pol/np.linalg.norm(self.pol)
            else:
                raise ValueError("pol must be +1, -1, or a numpy array")

            self.pol, self.pol_sig = promote_to_lambda(self.pol, var_name='polarization')
        else:
            # 'Tis a function, so let's save the function:
            self.pol, self.pol_sig = promote_to_lambda(pol, var_name='polarization')

        self.phase = phase
        self.eps = eps

        self.infinite_beam = not ('R' in self.kvec_sig or
                                  'R' in self.beta_sig or
                                  'R' in self.pol_sig)

        # TODO: add testing of kvec/pol orthogonality.

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
        pol = self.pol(R, t)

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
        if isinstance(cosbeta, (float, int)):
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
        if np.abs(np.dot(xp, self.kvec(R, t))) > 1e-10:
            raise ValueError('xp and k must be orthogonal.')
        if np.abs(np.dot(yp, self.kvec(R, t))) > 1e-10:
            raise ValueError('yp and k must be orthogonal.')
        if np.sum(np.abs(np.cross(xp, yp) - self.kvec(R, t))) > 1e-10:
            raise ValueError('xp, yp, and k must form a right-handed' +
                             'coordinate system.')

        pol_cart = self.cartesian_pol(R, t)

        if np.abs(np.dot(pol_cart, self.kvec(R, t))) > 1e-9:
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


    def electric_field(self, R, t, mean_detuning=0):
        """
        Returns the electric field of the laser beam at position R and time t.
        More specifically, this function returns E^\dagger, or so it would
        appear.
        """
        kvec = self.kvec(R, t)
        beta = self.beta(R, t)
        pol = self.pol(R, t)
        delta = self.detuning(t)

        amp = np.sqrt(beta/2)

        if isinstance(t, float) or (isinstance(t, np.ndarray) and t.size==1):
            Eq = pol*amp*np.exp(-1j*np.dot(kvec, R) + 1j*delta*t +
                                -1j*self.phase)
        else:
            Eq = np.multiply(
                pol.reshape(3, t.size),
                amp*np.exp(-1j*dot2D(kvec, R) + 1j*delta*t +
                           -1j*self.phase)
            )

        return Eq


    def electric_field_gradient(self, R, t, mean_detuning=0):
        """
        Returns the gradient of the electric field of the laser beam at
        position R and time t.  TODO: include effects like gradient of
        amplitude,  kvec, or individual pol components.
        """
        kvec = self.kvec(R, t)
        beta = self.beta(R, t)
        pol = self.pol(R, t)
        delta = self.detuning(t)

        amp = np.sqrt(beta/2)

        if self.infinite_beam:
            if isinstance(t, float) or (isinstance(t, np.ndarray) and t.size==1):
                delE = np.multiply(kvec.reshape(3, 1), pol.reshape(1, 3))*\
                     1j*amp*np.exp(-1j*np.dot(kvec, R) + 1j*delta*t +
                                              -1j*self.phase)
            else:
                delE = np.multiply(
                    np.multiply(kvec.reshape(3, 1, t.size),
                                pol.reshape(1, 3, t.size)),
                    1j*amp*np.exp(-1j*dot2D(kvec, R) + 1j*delta*t +
                                  -1j*self.phase)
                )
        else:
            (dx, dy, dz) = return_dx_dy_dz(R, self.eps)
            return np.array([
                 (self.electric_field(R+dx, t) -
                  self.electric_field(R-dx, t))/2/self.eps,
                 (self.electric_field(R+dy, t) -
                  self.electric_field(R-dy, t))/2/self.eps,
                 (self.electric_field(R+dz, t) -
                  self.electric_field(R-dz, t))/2/self.eps
                ])

        return delEq



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

    def pol(self, R=np.array([0., 0., 0.]), t=0.):
        return np.array([beam.pol(R, t) for beam in self.beam_vector])

    def beta(self, R=np.array([0., 0., 0.]), t=0.):
        return np.array([beam.beta(R, t) for beam in self.beam_vector])

    def kvec(self, R=np.array([0., 0., 0.]), t=0.):
        return np.array([beam.kvec(R, t) for beam in self.beam_vector])

    def detuning(self, t=0):
        return [beam.detuning(t) for beam in self.beam_vector]

    def electric_field(self, R=np.array([0., 0., 0.]), t=0., mean_detuning=0):
        return np.array([beam.electric_field(R, t) for beam in self.beam_vector])

    def electric_field_gradient(self, R=np.array([0., 0., 0.]), t=0., mean_detuning=0):
        return np.array([beam.electric_field_gradient(R, t)
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

        return [np.tensordot(D, beam.pol(R, t), ([1],[0]))
                for beam in self.beam_vector]

    def cartesian_pol(self, R=np.array([0., 0., 0.]), t=0):
        return [beam.cartesian_pol(R, t) for beam in self.beam_vector]

    def jones_vector(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        return [beam.jones_vector(xp, yp, R, t) for beam in self.beam_vector]

    def stokes_parameters(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        return [beam.stokes_parameters(xp, yp, R, t) for beam in self.beam_vector]

    def polarization_ellipse(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        return [beam.polarization_ellipse(xp, yp, R, t) for beam in self.beam_vector]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_field = magField(lambda R: np.array([-0.5*R[0], -0.5*R[1], 1*R[2]]))

    print(test_field.Field())
    print(test_field.gradField(np.array([5., 2., 1.])))

    example_beams = laserBeams([
        {'kvec':np.array([0., 0., 1.]), 'pol':np.array([0., 0., 1.]),
         'pol_coord':'spherical', 'delta':-2, 'beta': 1.},
        {'kvec':np.array([0., 0., -1.]), 'pol':np.array([0., 0., 1.]),
         'pol_coord':'spherical', 'delta':-2, 'beta': 1.},
        ])

    print(example_beams.beam_vector[0].jones_vector(np.array([1., 0., 0.]), np.array([0., 1., 0.])))

    print(example_beams.kvec())
    print(example_beams.pol())
    print(example_beams.beta())
    print(example_beams.electric_field_gradient(np.array([0., 0., 0.]), 0.5))

    example_beams_2 = laserBeams([
        {'kvec':np.array([0., 0., 1.]), 'pol':np.array([0., 0., 1.]),
         'pol_coord':'spherical', 'delta':-2, 'beta': lambda R: 1.},
        {'kvec':np.array([0., 0., -1.]), 'pol':np.array([0., 0., 1.]),
         'pol_coord':'spherical', 'delta':-2, 'beta': lambda R: 1.},
        ])

    print(example_beams_2.electric_field_gradient(np.array([0., 0., 0.]), 0.5))
