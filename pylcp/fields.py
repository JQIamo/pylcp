import numpy as np
from inspect import signature
from pylcp.common import cart2spherical, spherical2cart
from .integration_tools import parallelIntegrator
from scipy.spatial.transform import Rotation

import numba

@numba.njit
def dot2D(a, b):
    c = np.zeros((a.shape[1],), dtype=a.dtype)
    for ii in range(a.shape[1]):
        c[ii] = np.sum(a[:, ii]*b[:, ii])
    return c

@numba.njit
def electric_field(r, t, amp, pol, k, phase):
    return pol*amp*np.exp(-1j*(k[0]*r[0]+k[1]*r[1]+k[2]*r[2]) + 1j*phase)


def return_constant_val(R, t, val):
    if R.shape==(3,):
        return val
    elif R.shape[0] == 3:
        return val*np.ones(R[0].shape)
    else:
        raise ValueError('The first dimension of R should have length 3, ' +
                         'not %d.'%R.shape[0])

def return_constant_vector(R, t, vector):
    if R.shape==(3,):
        return vector
    elif R.shape[0] == 3:
        return np.outer(vector, np.ones(R[0].shape))
    else:
        raise ValueError('The first dimension of R should have length 3, ' +
                         'not %d.'% R.shape[0])

def return_constant_val_t(t, val):
    if isinstance(t, np.ndarray):
        return val*np.ones(t.shape)
    else:
        return val

def promote_to_lambda(val, var_name=None, type='Rt'):
    if type is 'Rt':
        if not callable(val):
            if isinstance(val, list) or isinstance(val, np.ndarray):
                func = lambda R=np.array([0., 0., 0.]), t=0.: return_constant_vector(R, t, val)
            else:
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
                raise TypeError('Signature [%s] of function %s not'+
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
                raise TypeError('Signature [%s] of function %s not '+
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
    Base magnetic field class

    Stores a magnetic defined magnetic field and calculates useful derivatives
    for `pylcp`.

    Parameters
    ----------
    field : array_like with shape (3,) or callable
        If constant, the magnetic field vector, specified as either as an array_like
        with shape (3,).  If a callable, it must have a signature like (R, t), (R),
        or (t) where R is an array_like with shape (3,) and t is a float and it
        must return an array_like with three elements.
    eps : float, optional
        Small distance to use in calculation of numerical derivatives.  By default
        `eps=1e-5`.

    Attributes
    ----------
    eps : float
        small epsilon used for computing derivatives
    """
    def __init__(self, field, eps=1e-5):
        self.eps = eps

        R = np.random.rand(3) # Pick a random point for testing

        # Promote it to a lambda func:
        self.Field, self.FieldSig = promote_to_lambda(field, var_name='for field')

        # Try it out:
        response = self.Field(R, 0.)
        if (isinstance(response, float) or isinstance(response, int) or
            len(response) != 3):
            raise ValueError('Magnetic field function must return a vector.')

    def FieldMag(self, R=np.array([0., 0., 0.]), t=0):
        """
        Magnetic field magnitude at R and t:

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        B : float
            the magnetic field mangitude at position R and time t.
        """
        return np.linalg.norm(self.Field(R, t))

    def gradFieldMag(self, R=np.array([0., 0., 0.]), t=0):
        """
        Gradient of the magnetic field magnitude at R and t:

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dB : array_like, shape (3,)
            :math:`\\nabla|B|`, the gradient of the magnetic field magnitude
            at position :math:`R` and time :math:`t`.
        """
        dx, dy, dz = return_dx_dy_dz(R, self.eps)

        return np.array([
            (self.FieldMag(R+dx, t)-self.FieldMag(R-dx, t))/2/self.eps,
            (self.FieldMag(R+dy, t)-self.FieldMag(R-dy, t))/2/self.eps,
            (self.FieldMag(R+dz, t)-self.FieldMag(R-dz, t))/2/self.eps
            ])

    def gradField(self, R=np.array([0., 0., 0.]), t=0):
        """
        Full spaitial derivative of the magnetic field at R and t:

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dB : array_like, shape (3, 3)
            the full gradient of the magnetic field, with elements

            .. math::
              \\begin{pmatrix}
                \\frac{dB_x}{dx} & \\frac{dB_y}{dx} & \\frac{dB_z}{dx} \\\\
                \\frac{dB_x}{dy} & \\frac{dB_y}{dy} & \\frac{dB_z}{dy} \\\\
                \\frac{dB_x}{dz} & \\frac{dB_y}{dz} & \\frac{dB_z}{dz} \\\\
              \\end{pmatrix}

        Notes
        -----
        This method calculates the derivative stupidly, just using first order
        numerical differentiation using the `eps` parameter.
        """
        dx, dy, dz = return_dx_dy_dz(R, self.eps)

        return np.array([
            (self.Field(R+dx, t) - self.Field(R-dx, t))/2/self.eps,
            (self.Field(R+dy, t) - self.Field(R-dy, t))/2/self.eps,
            (self.Field(R+dz, t) - self.Field(R-dz, t))/2/self.eps
            ])

class iPMagneticField(magField):
    """
    Ioffe-Pritchard trap magnetic field

    Generates a magnetic field of the form

    .. math::
      \mathbf{B} = B_1 x \\hat{x} - B_1 y \\hat{y} + \\left(B_0 + \\frac{B_2}{2}z^2\\right)\\hat{z}

    Parameters
    ----------
    B0 : float
        Constant offset field
    B1 : float
        Magnetic field gradient in x-y plane
    B2 : float
        Magnetic quadratic component along z direction.

    Notes
    -----
    It is currently missing extra terms that are required for it to fulfill
    Maxwell's equations at second order.
    """
    def __init__(self, B0, B1, B2, eps = 1e-5):
        super().__init__(lambda R, t: np.array([B1*R[0]-B2*R[0]*R[2]/2, -R[1]*B1-B2*R[1]*R[2]/2, B0+B2/2*(R[2]**2 - (R[0]**2+R[1]**2)/2)]))
        self.B0 = B0
        self.B1 = B1
        self.B2 = B2

    #Analytical form, not numerical for this and gradField
    def gradFieldMag(self, R=np.array([0., 0., 0.]), t=0):
        a = self.B0
        b = self.B1
        c = self.B2
        x = R[0]
        y = R[1]
        z = R[2]
        mag = self.FieldMag(R, t)
        xcom = 0.5*(2*b**2*x-a*c*x+(c**2)*(x**3)/4+(c**2)*(y**2)*x/4-2*b*c*z*x)/mag
        ycom = 0.5*(2*b**2*(y)-a*c*y+(c**2)*(x**2)*y/4 + (c**2)*(y**3)/4+2*b*c*z*y)/mag
        zcom = 0.5*(0-b*c*(x**2)+b*c*(y**2)+2*a*c*z+(c**2)*(z**3))/mag
        return np.array([xcom, ycom, zcom])

    def gradField(self, R=np.array([0., 0., 0.]), t=0):
        B0 = self.B0
        B1 = self.B1
        B2 = self.B2
        x = R[0]
        y = R[1]
        z = R[2]
        xcom = np.array([B1-B2*z/2, 0, -B2*x/2])
        ycom = np.array([0, -B1-B2*z/2, B2*y/2])
        zcom = np.array([-B2*x/2, -B2*y/2, B2*z])

        return np.array([
            np.array([B1-B2*z/2, 0, -B2*x/2]),
            np.array([0, -B1-B2*z/2, B2*y/2]),
            np.array([-B2*x/2, -B2*y/2, B2*z])
            ])


class constantMagneticField(magField):
    """
    Spatially constant magnetic field

    Represents a magnetic field of the form

    .. math::
      \\mathbf{B} = \mathbf{B}_0

    Parameters
    ----------
    val : array_like with shape (3,)
        The three-vector defintion of the constant magnetic field.
    """
    def __init__(self, B0):
        super().__init__(lambda R, t: B0)

        self.constant_grad_field_mag = np.zeros((3,))
        self.constant_grad_field = np.zeros((3,3))

    def gradFieldMag(self, R=np.array([0., 0., 0.]), t=0):
        """
        Gradient of the magnetic field magnitude at R and t:

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dB : np.zeros((3,))
            The gradient of a constant magnetic field magnitude is always zero.
        """
        return self.constant_grad_field_mag

    def gradField(self, R=np.array([0., 0., 0.]), t=0):
        """
        Gradient of the magnetic field magnitude at R and t:

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dB : np.zeros((3,))
            :math:`\\nabla|B|=0`, the gradient of the magnitude of a constant
            magnetic field is always zero.
        """
        return self.constant_grad_field


class quadrupoleMagneticField(magField):
    """
    Spherical quadrupole  magnetic field

    Represents a magnetic field of the form

    .. math::
      \\mathbf{B} = \\alpha\\left(- \\frac{x\\hat{x}}{2} - \\frac{y\\hat{y}}{2} + z\\hat{z}\\right)

    Parameters
    ----------
    alpha : float
        strength of the magnetic field gradient.
    """
    def __init__(self, alpha, eps=1e-5):
        super().__init__(lambda R, t: alpha*np.array([-0.5*R[0], -0.5*R[1], R[2]]))
        self.alpha = alpha

        self.constant_grad_field = alpha*\
            np.array([[-0.5, 0., 0.], [0., -0.5, 0.], [0., 0., 1.]])

    def gradField(self, R=np.array([0., 0., 0.]), t=0):
        """
        Full spaitial derivative of the magnetic field at R and t:

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dB : array_like, shape (3, 3)
            the full gradient of the magnetic field, with elements

            .. math::
              \\begin{pmatrix}
                -\\alpha/2 & 0 & 0 \\\\
                0 & -\\alpha/2 & 0 \\\\
                0 & 0 & \\alpha \\\\
              \\end{pmatrix}
        """
        return self.constant_grad_field



# First, define the laser beam class:
class laserBeam(object):
    """
    The base class for a single laser beam

    Attempts to represent a laser beam as

    .. math::
        \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}}(r, t) E_0(r, t)
        e^{i\\mathbf{k}(r,t)\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}

    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the deutning,
    :math:`t` is the time, and :math:`\\phi` is the phase.


    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array or as callable function.  If a callable, it
        must have a signature like (R, t), (R), or (t) where R is an array_like with
        shape (3,) and t is a float and it must return an array_like with three
        elements.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,), or as callable function.  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`. If a callable, it must
        have a signature like (R, t), (R), or (t) where R is an array_like with
        shape (3,) and t is a float and it must return an array_like with three
        elements.
    s : float or callable
        The intensity of the laser beam, normalized to the saturation intensity,
        specified as either a float or as callable function.  If a callable,
        it must have a signature like (R, t), (R), or (t) where R is an
        array_like with shape (3,) and t is a float and it must return a float.
    delta: float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    phase : float, optional
        Phase of laser beam.  By default, zero.
    pol_coord : string, optional
        Polarization basis of the input polarization vector: 'cartesian'
        or 'spherical' (default).
    eps : float, optional
        Small distance to use in calculation of numerical derivatives.  By default
        `eps=1e-5`.

    Attributes
    ----------
    eps : float
        Small epsilon used for computing derivatives
    phase : float
        Overall phase of the laser beam.
    """
    def __init__(self, kvec=None, s=None, pol=None, delta=None,
                 phase=0., pol_coord='spherical', eps=1e-5):
        # Promote it to a lambda func:
        if not kvec is None:
            self.kvec, self.kvec_sig = promote_to_lambda(kvec, var_name='kvector')

        # Promote it to a lambda func:
        if not s is None:
            self.intensity, self.intensity_sig = promote_to_lambda(s, var_name='s')

        if not pol is None:
            if not callable(pol):
                pol = self.__parse_constant_polarization(pol, pol_coord)

            # Now, promote!
            self.pol, self.pol_sig = promote_to_lambda(pol, var_name='polarization')

        # Promote it to a lambda func:
        if not delta is None:
            self.delta, self.delta_sig = promote_to_lambda(delta, var_name='delta', type='t')

        if self.delta_sig == '(t)':
            self.delta_phase = parallelIntegrator(self.delta)
        elif self.delta_sig == '()':
            self.delta_phase = lambda t: delta*t

        # Promote it to a lambda func:
        if not phase is None:
            self.phase, self.phase_sig = promote_to_lambda(phase, var_name='phase', type='t')

        self.eps = eps

    def __parse_constant_polarization(self, pol, pol_coord):
        if isinstance(pol, float) or isinstance(pol, int):
            # If the polarization is defined by just a single number (+/-1),
            # we assume that the polarization is defined as sigma^+ or sigma^-
            # using the k-vector of the light as the axis defining z.  In this
            # case, we want to project onto the actual z axis, which is
            # relatively simple as there is only one angle.

            # Set the polarization in this direction:
            if np.sign(pol)<0:
                self.pol = np.array([1., 0., 0.], dtype='complex')
            else:
                self.pol = np.array([0., 0., 1.], dtype='complex')

            # Promote to lambda:
            self.pol, self.pol_sig = promote_to_lambda(self.pol, var_name='polarization')

            # Project onto the actual k-vector:
            self.pol = self.project_pol(self.kvec()/np.linalg.norm(self.kvec()),
                                        invert=True).astype('complex128')

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

        return self.pol


    def kvec(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the k-vector of the laser beam

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        kvec : array_like, size(3,)
            the k vector at position R and time t.
        """
        pass

    def intensity(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the intensity of the laser beam at position R and t

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        s : float or array_like
            Saturation parameter of the laser beam at R and t.
        """
        pass

    def pol(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the polarization of the laser beam at position R and t

        The polarization is returned in the spherical basis.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : array_like, size (3,)
            polarization of the laser beam at R and t in spherical basis.
        """
        pass

    def delta(self, t=0.):
        """
        Returns the detuning of the laser beam at time t

        Parameters
        ----------
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        delta : float or array like
            detuning of the laser beam at time t
        """
        pass

    # TODO: add testing of kvec/pol orthogonality.
    def project_pol(self, quant_axis, R=np.array([0., 0., 0.]), t=0,
                    treat_nans=False, calculate_norm=False, invert=False):
        """
        Project the polarization onto a quantization axis.

        Parameters
        ----------
        quant_axis : array_like, shape (3,)
            A normalized 3-vector of the quantization axis direction.
        R : array_like, shape (3,), optional
            If polarization is a function of R is the
            3-vectors at which the polarization shall be calculated.
        calculate_norm : bool, optional
            If true, renormalizes the quant_axis.  By default, False.
        treat_nans : bool, optional
            If true, every place that nan is encoutnered, replace with the
            $hat{z}$ axis as the quantization axis.  By default, False.
        invert : bool, optional
            If true, invert the process to project the quantization axis
            onto the specified polarization.

        Returns
        -------
        projected_pol : array_like, shape (3,)
            The polarization projected onto the quantization axis.
        """

        # First, return the polarization at the desired R and t.
        pol = self.pol(R, t)

        # Second, check the quanitization axis if specified by the user.  The fun
        # thing here is that we may only need to do this once, since it should
        # overwrite the original variable with the improper quant_axis with the
        # proper one.
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

        # To project the full three-vector, we want to determine the Euler
        # angles alpha, beta, and gamma that rotate the z-axis into the
        # quantization axis.  The final Euler angle, gamma, only sets the
        # phase of the -1 to +1 component, so it does not have any physical
        # meaning for the rate equations (where we expect this method to
        # mostly be used.)  Thus, we just set that angle equal to zero here.
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
        if pol.shape == (3,) and quant_axis.shape == (3,):
            return D @ pol
        else:
            return np.tensordot(D, pol, ([1],[0]))


    def cartesian_pol(self, R=np.array([0., 0., 0.]), t=0):
        """
        Returns the polarization in Cartesian coordinates.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : array_like, size (3,)
            polarization of the laser beam at R and t in Cartesian basis.
        """

        pol = self.pol(R, t)
        return spherical2cart(pol)

    def jones_vector(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        """
        Returns the Jones vector at position

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : array_like, size (2,)
            Jones vector of the laser beam at R and t in Cartesian basis.
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
        The Stokes Parameters of the laser beam at R and t

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : array_like, shape (3,)
            Stokes parameters for the laser beam, [Q, U, V]
        """
        jones_vector = self.jones_vector(xp, yp, R, t)

        Q = np.abs(jones_vector[0])**2 - np.abs(jones_vector[1])**2
        U = 2*np.real(jones_vector[0]*np.conj(jones_vector[1]))
        V = -2*np.imag(jones_vector[0]*np.conj(jones_vector[1]))

        return (Q, U, V)


    def polarization_ellipse(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        """
        The polarization ellipse parameters of the laser beam at R and t

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        psi : float
            :math:`\\psi` parameter of the polarization ellipse
        chi : float
            :math:`\\chi` parameter of the polarization ellipse
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


    def electric_field(self, R, t):
        """
        The electric field at position R and t

        Parameters
        ----------
        R : array_like, size (3,)
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        Eq : array_like, shape (3,)
            electric field in the spherical basis.
        """
        kvec = self.kvec(R, t)
        s = self.intensity(R, t)
        pol = self.pol(R, t)
        delta_phase = self.delta_phase(t)
        phase = self.phase(t)

        amp = np.sqrt(2*s)

        if isinstance(t, float):
            Eq = electric_field(R, t, amp, pol, kvec, delta_phase - phase)
        else:
            Eq = pol.reshape(3, t.size)*\
            (amp*np.exp(-1j*dot2D(kvec, R) + 1j*delta_phase - 1j*phase)).reshape(1, t.size)

        return Eq


    def electric_field_gradient(self, R, t):
        """
        The full derivative of electric field at position R and t

        Parameters
        ----------
        R : array_like, size (3,)
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dEq : array_like, shape (3, 3)
            The full gradient of the electric field, in spherical coordinates.

            .. math::
              \\begin{pmatrix}
                \\frac{dE_{-1}}{dx} & \\frac{dE_0}{dx} & \\frac{dE_{+1}}{dx} \\\\
                \\frac{dE_{-1}}{dy} & \\frac{dE_0}{dy} & \\frac{dE_{+1}}{dy} \\\\
                \\frac{dE_{-1}}{dz} & \\frac{dE_0}{dz} & \\frac{dE_{+1}}{dz} \\\\
              \\end{pmatrix}
        """
        (dx, dy, dz) = return_dx_dy_dz(R, self.eps)
        delEq = np.array([
            (self.electric_field(R+dx, t) -
             self.electric_field(R-dx, t))/2/self.eps,
            (self.electric_field(R+dy, t) -
             self.electric_field(R-dy, t))/2/self.eps,
            (self.electric_field(R+dz, t) -
             self.electric_field(R-dz, t))/2/self.eps
            ])

        return delEq


class infinitePlaneWaveBeam(laserBeam):
    """
    Infinte plane wave beam

    A beam which has spatially constant intensity, k-vector, and polarization.

    .. math::
        \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}} E_0e^{i\\mathbf{k}\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}

    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the deutning,
    :math:`t` is the time, and :math:`\\phi` is the phase.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The intensity of the laser beam, specified as either a float or as
        callable function.
    delta: float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    **kwargs :
        Additional keyword arguments to pass to laserBeam superclass.

    Notes
    -----
    This implementation is much faster, when it can be used, compared to the
    base laserBeam class.
    """
    def __init__(self, kvec, pol, s, delta, **kwargs):
        if callable(kvec):
            raise TypeError('kvec cannot be a function for an infinite plane wave.')

        if callable(s):
            raise TypeError('s cannot be a function for an infinite plane wave.')

        if callable(pol):
            raise TypeError('Polarization cannot be a function for an infinite plane wave.')

        # Use the super class to define the functions kvec, s, pol, and delta.
        super().__init__(kvec=kvec, s=s, pol=pol, delta=delta,
                         **kwargs)

        # Save the constant values (might be useful):
        self.con_kvec = kvec
        self.con_s = s
        self.con_pol = self.pol(np.array([0., 0., 0.]), 0.)
        # Define attributes to speed up gradient calculation:
        self.amp = np.sqrt(2*self.con_s)
        self.dEq_prefactor = (-1j*self.amp*self.con_kvec.reshape(3, 1)*
                              self.con_pol.reshape(1, 3))

    def electric_field_gradient(self, R, t):
        # With a plane wave, this is simple:
        delta_phase = self.delta_phase(t)
        phase = self.phase(t)

        if isinstance(t, float) or (isinstance(t, np.ndarray) and t.size==1):
            delEq = self.dEq_prefactor*\
            np.exp(-1j*np.dot(self.con_kvec, R) + 1j*delta_phase - 1j*phase)
        else:
            delEq = self.dEq_prefactor.reshape(3, 3, 1)*\
            np.exp(-1j*np.dot(self.con_kvec, R) + 1j*delta_phase -1j*phase).reshape(1, 1, t.size)

        return delEq


class gaussianBeam(laserBeam):
    """
    Collimated Gaussian beam

    A beam which has spatially constant k-vector and polarization, with a
    Gaussian intensity modulation.  Specifically,

    .. math::
      \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}} E_0 e^{-\\mathbf{r}^2/w_b^2} e^{i\\mathbf{k}\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}

    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the deutning,
    :math:`t` is the time, and :math:`\\phi` is the phase.  Note that because
    :math:`I\\propto E^2`, :math:`w_b` is the :math:`1/e^2` radius.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The maximum intensity of the laser beam at the center, specified as
        either a float or as callable function.
    delta : float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    wb : float
        The :math:`1/e^2` radius of the beam.
    **kwargs:
        Additional keyword arguments to pass to the laserBeam superclass.
    """
    def __init__(self, kvec, pol, s, delta, wb, **kwargs):
        if callable(kvec):
            raise TypeError('kvec cannot be a function for a Gaussian beam.')

        if callable(pol):
            raise TypeError('Polarization cannot be a function for a Gaussian beam.')

        # Use super class to define kvec(R, t), pol(R, t), and delta(t)
        super().__init__(kvec=kvec, pol=pol, delta=delta, **kwargs)

        # Save the constant values (might be useful):
        self.con_kvec = kvec
        self.con_khat = kvec/np.linalg.norm(kvec)
        self.con_pol = self.pol(np.array([0., 0., 0.]), 0.)

        # Save the parameters specific to the Gaussian beam:
        self.s_max = s # central saturation parameter
        self.wb = wb # 1/e^2 radius
        self.define_rotation_matrix()

    def define_rotation_matrix(self):
        # Angles of rotation:
        th = np.arccos(self.con_khat[2])
        phi = np.arctan2(self.con_khat[1], self.con_khat[0])

        # Use scipy to define the rotation matrix
        self.rmat = Rotation.from_euler('ZY', [phi, th]).inv().as_matrix()

    def intensity(self, R=np.array([0., 0., 0.]), t=0.):
        # Rotate up to the z-axis where we can apply formulas:
        Rp = np.einsum('ij,j...->i...', self.rmat, R)
        rho_sq=np.sum(Rp[:2]**2, axis=0)
        # Return the intensity:
        return self.s_max*np.exp(-2*rho_sq/self.wb**2)


class clippedGaussianBeam(gaussianBeam):
    """
    Clipped, collimated Gaussian beam

    A beam which has spatially constant k-vector and polarization, with a
    Gaussian intensity modulation.  Specifically,

    .. math::
      \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}} E_0 e^{-\\mathbf{r}^2/w_b^2} (|\\mathbf{r}|<r_s) e^{i\\mathbf{k}\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}

    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`r_s` is the radius of the stop,
    :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the deutning,
    :math:`t` is the time, and :math:`\\phi` is the phase. Note that because
    :math:`I\\propto E^2`, :math:`w_b` is the :math:`1/e^2` radius.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The maximum intensity of the laser beam at the center, specified as
        either a float or as callable function.
    delta : float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    wb : float
        The :math:`1/e^2` radius of the beam.
    rs : float
        The radius of the stop.
    **kwargs:
        Additional keyword arguments to pass to the laserBeam superclass.
    """
    def __init__(self, kvec, pol, s, delta, wb, rs, **kwargs):
        super().__init__(kvec=kvec, pol=pol, s=s, delta=delta, wb=wb, **kwargs)

        self.rs = rs # Save the radius of the stop.

    def intensity(self, R=np.array([0., 0., 0.]), t=0.):
        Rp = np.einsum('ij,j...->i...', self.rmat, R)
        rho_sq = np.sum(Rp[:2]**2, axis=0)
        return self.s_max*np.exp(-2*rho_sq/self.wb**2)*(np.sqrt(rho_sq)<self.rs)


class laserBeams(object):
    """
    The base class for a collection of laser beams

    Parameters
    ----------
    laserbeamparams : array_like of laserBeam or array_like of dictionaries
        If array_like contains laserBeams, the laserBeams in the array will be joined
        together to form a collection.  If array_like is a list of dictionaries, the
        dictionaries will be passed as keyword arguments to beam_type
    beam_type : laserBeam or laserBeam subclass, optional
        Type of beam to use in the collection of laserBeams.  By default
        `beam_type=laserBeam`.
    """
    def __init__(self, laserbeamparams=None, beam_type=laserBeam):
        if laserbeamparams is not None:
            if not isinstance(laserbeamparams, list):
                raise ValueError('laserbeamparams must be a list.')
            self.beam_vector = []
            for laserbeamparam in laserbeamparams:
                if isinstance(laserbeamparam, dict):
                    self.beam_vector.append(beam_type(**laserbeamparam))
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
        """
        Add a laser to the collection

        Parameters
        ----------
        new_laser : laserBeam or laserBeam subclass
        """
        if isinstance(new_laser, laserBeam):
            self.beam_vector.append(new_laser)
            self.num_of_beams = len(self.beam_vector)
        elif isinstance(new_laser, dict):
            self.beam_vector.append(laserBeam(**new_laser))
        else:
            raise TypeError('new_laser should by type laserBeam or a dictionary' +
                            'of arguments to initialize the laserBeam class.')

    def pol(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the polarization of the laser beam at position R and t

        The polarization is returned in the spherical basis.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : list of array_like, size (3,)
            polarization of each laser beam at R and t in spherical basis.
        """
        return np.array([beam.pol(R, t) for beam in self.beam_vector])

    def intensity(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the intensity of the laser beam at position R and t

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        s : list of float or array_like
            Saturation parameters of all laser beams at R and t.
        """
        return np.array([beam.intensity(R, t) for beam in self.beam_vector])

    def kvec(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the k-vector of the laser beam

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        kvec : list of array_like, size(3,)
            the k vector at position R and time t for each laser beam.
        """
        return np.array([beam.kvec(R, t) for beam in self.beam_vector])

    def delta(self, t=0):
        """
        Returns the detuning of the laser beam at time t

        Parameters
        ----------
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        delta : float or array like
            detuning of the laser beam at time t for all laser beams
        """
        return np.array([beam.delta(t) for beam in self.beam_vector])

    def electric_field(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the electric field of the laser beams

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        E : list of array_like, size(3,)
            the electric field vectors at position R and time t for each laser beam.
        """
        return np.array([beam.electric_field(R, t) for beam in self.beam_vector])

    def electric_field_gradient(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the gradient of the electric field of the laser beams

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dE : list of array_like, size(3,)
            the electric field gradient matrices at position R and time t for each laser beam.
        """
        return np.array([beam.electric_field_gradient(R, t)
                         for beam in self.beam_vector])

    def total_electric_field(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the total electric field of the laser beams

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        E : array_like, size(3,)
            the total electric field vector at position R and time t of all
            the laser beams
        """
        return np.sum(self.electric_field(R, t), axis=0)

    def total_electric_field_gradient(self, R=np.array([0., 0., 0.]), t=0.):
        """
        Returns the total gradient of the electric field of the laser beams

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dE : array_like, size(3,)
            the total electric field gradient matrices at position R and time t
            of all laser beams.
        """
        return np.sum(self.electric_field_gradient(R, t), axis=0)


    def project_pol(self, quant_axis, R=np.array([0., 0., 0.]), t=0, **kwargs):
        """
        Project the polarization onto a quantization axis.

        Parameters
        ----------
        quant_axis : array_like, shape (3,)
            A normalized 3-vector of the quantization axis direction.
        R : array_like, shape (3,), optional
            If polarization is a function of R is the
            3-vectors at which the polarization shall be calculated.
        calculate_norm : bool, optional
            If true, renormalizes the quant_axis.  By default, False.
        treat_nans : bool, optional
            If true, every place that nan is encoutnered, replace with the
            $hat{z}$ axis as the quantization axis.  By default, False.
        invert : bool, optional
            If true, invert the process to project the quantization axis
            onto the specified polarization.

        Returns
        -------
        projected_pol : list of array_like, shape (3,)
            The polarization projected onto the quantization axis for all
            laser beams
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

        if quant_axis.shape == (3,) and R.shape == (3,):
            return [D @ beam.pol(R, t) for beam in self.beam_vector]
        else:
            return [np.tensordot(D, beam.pol(R, t), ([1],[0]))
                    for beam in self.beam_vector]

    def cartesian_pol(self, R=np.array([0., 0., 0.]), t=0):
        """
        Returns the polarization of all laser beams in Cartesian coordinates.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the polarization.  By default,
            the origin.
        t : float, optional
            time at which to return the polarization.  By default, t=0.

        Returns
        -------
        pol : array_like, shape (num_of_beams, 3)
            polarization of the laser beam at R and t in Cartesian basis.
        """
        return [beam.cartesian_pol(R, t) for beam in self.beam_vector]

    def jones_vector(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        """
        Jones vector at position R and time t

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to evaluate the Jones vector.  By default,
            the origin.
        t : float, optional
            time at which to evaluate the Jones vector.  By default, t=0.

        Returns
        -------
        pol : array_like, size (num_of_beams, 2)
            Jones vector of the laser beams at R and t in Cartesian basis.
        """

        return [beam.jones_vector(xp, yp, R, t) for beam in self.beam_vector]

    def stokes_parameters(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        """
        The Stokes Parameters of the laser beam at R and t

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to calculate the Stokes parameters.
            By default, the origin.
        t : float, optional
            time at which to calculate the Stokes parameters.  By default, t=0.

        Returns
        -------
        pol : array_like, shape (num_of_beams, 3)
            Stokes parameters for the laser beams, [Q, U, V]
        """
        return [beam.stokes_parameters(xp, yp, R, t) for beam in self.beam_vector]

    def polarization_ellipse(self, xp, yp, R=np.array([0., 0., 0.]), t=0):
        """
        The polarization ellipse parameters of the laser beam at R and t

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        list of (psi, chi) : list of tuples
            list of (:math:`\\psi`, :math:`\\chi`) parameters of the
            polarization ellipses for each laser beam
        """
        return [beam.polarization_ellipse(xp, yp, R, t) for beam in self.beam_vector]


class conventional3DMOTBeams(laserBeams):
    """
    A collection of laser beams for 6-beam MOT

    The standard geometry is to generate counter-progagating beams along all
    orthogonal axes :math:`(\\hat{x}, \\hat{y}, \\hat{z})`.

    Parameters
    ----------
    k : float, optional
        Magnitude of the k-vector for the six laser beams.  Default: 1
    pol : int or float, optional
        Sign of the circular polarization for the beams moving along
        :math:`\\hat{z}`.  Default: +1.  Orthogonal beams have opposite
        polarization by default.
    rotation_angles : array_like
        List of angles to define a rotated MOT.  Default: [0., 0., 0.]
    rotation_spec : str
        String to define the convention of the Euler rotations.  Default: 'ZYZ'
    beam_type : pylcp.laserBeam or subclass
        Type of beam to generate.
    **kwargs :
        other keyword arguments to pass to beam_type
    """
    def __init__(self, k=1, pol=+1, rotation_angles=[0., 0., 0.],
                 rotation_spec='ZYZ', beam_type=laserBeam, **kwargs):
        super().__init__()

        rot_mat = Rotation.from_euler(rotation_spec, rotation_angles).as_matrix()

        kvecs = [np.array([ 1.,  0.,  0.]), np.array([-1.,  0.,  0.]),
                 np.array([ 0.,  1.,  0.]), np.array([ 0., -1.,  0.]),
                 np.array([ 0.,  0.,  1.]), np.array([ 0.,  0., -1.])]
        pols = [-pol, -pol, -pol, -pol, +pol, +pol]

        for kvec, pol in zip(kvecs, pols):
            self.add_laser(beam_type(kvec=rot_mat @ (k*kvec), pol=pol, **kwargs))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_field = magField(lambda R: np.array([-0.5*R[0], -0.5*R[1], 1*R[2]]))

    print(test_field.Field())
    print(test_field.gradField(np.array([5., 2., 1.])))

    example_beams = laserBeams([
        {'kvec':np.array([0., 0., 1.]), 'pol':np.array([0., 0., 1.]),
         'pol_coord':'spherical', 'delta':-2, 's': 1.},
        {'kvec':np.array([0., 0., -1.]), 'pol':np.array([0., 0., 1.]),
         'pol_coord':'spherical', 'delta':-2, 's': 1.},
        ])

    print(example_beams.beam_vector[0].jones_vector(np.array([1., 0., 0.]), np.array([0., 1., 0.])))

    print(example_beams.kvec())
    print(example_beams.pol())
    print(example_beams.intensity())
    print(example_beams.electric_field_gradient(np.array([0., 0., 0.]), 0.5))

    example_beams_2 = laserBeams([
        {'kvec':np.array([0., 0., 1.]), 'pol':np.array([0., 0., 1.]),
         'pol_coord':'spherical', 'delta':-2, 's': lambda R: 1.},
        {'kvec':np.array([0., 0., -1.]), 'pol':np.array([0., 0., 1.]),
         'pol_coord':'spherical', 'delta':-2, 's': lambda R: 1.},
        ])

    print(example_beams_2.electric_field_gradient(np.array([0., 0., 0.]), 0.5))

    example_beam = gaussianBeam(np.array([1., 0., 0.]), +1, 5, -2, 1000)
    print(example_beam.s(np.array([0., 1000/np.sqrt(2), 1000/np.sqrt(2)])))

    example_beam = infinitePlaneWaveBeam(np.array([1., 0., 0.]), +1, 5, -2)
    print(example_beam.electric_field_gradient(np.array([0., 0., 0.]), 0.))

    R = np.random.rand(3, 101)
    t = np.linspace(0, 10, 101)
    print(example_beam.electric_field_gradient(R, t).shape)

    MOT_beams = conventional3DMOTBeams(-2, 1, beam_type=gaussianBeam, wb=1000)
    MOT_beams.beam_vector[1].kvec()
