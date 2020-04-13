import numpy as np
<<<<<<< Updated upstream
=======
from inspect import signature

def promote_to_lambda(val, var_name=None):
    if not callable(val):
        val = lambda R, t: val
        sig = '()'
    else:
        sig = str(signature(val))
        if ('(R)' in sig or '(r)' in sig or '(x)' in sig):
            val = lambda R, t: val(R)
            sig = '(R)'
        elif ('(R, t)' in self.kvec_sig or '(r, t)' in self.kvec_sig or
              '(x, t)' in self.kvec_sig):
            val = lambda R, t: val(R)
            sig = '(R, t)'
        else:
            raise StandardError('Signature [%s] of function %s not understood.'% (sig, var_name))

    return val, sig

>>>>>>> Stashed changes

class magField(object):
    """
    An object the calculates relevant properties of a static, magnetic field.
    """
    def __init__(self, func, FieldMag=None, gradFieldMag=None,
                 gradField=None, eps=1e-5):
        self.eps = eps

        R = np.random.rand(3) # Pick a random point for testing

        if callable(func):
            # Try it out:
            response = func(R)
            if (isinstance(response, float) or isinstance(response, int) or
                len(response) != 3):
                raise StandardError('Magnetic field function must return a vector.')
            self.Field = func
        else:
            raise TypeError('Magnetic field must be a callable function')

        if FieldMag is None:
            self.FieldMag = self.internal_FieldMag
        elif callable(mag_fun):
            # Try it out.  It should take a three vector and respond with a
            # scalar
            response = FieldMag(R)
            if not (isinstance(reponse, float) or (isinstance(response, list) and len(response) != 1)):
                raise StandardError('Magnetic field magnitude function must ' +
                                    'return a scalar.')
            self.FieldMag = FieldMag
        else:
            raise TypeError('FieldMag argument must be a callable function')

        if gradFieldMag is None:
            self.gradFieldMag = self.internal_gradFieldMag
        elif callable(mag_fun):
            # Try it out.  It should take a three vector and respond with a
            # scalar
            response = gradFieldMag(R)
            if len(response) != 3:
                raise StandardError('gradFieldMag field magnitude function must ' +
                                    'return a vector.')
            self.gradFieldMag = gradFieldMag
        else:
            raise TypeError('FieldMag argument must be a callable function')

        if gradField is None:
            self.gradField = self.internal_gradField
        elif callable(mag_fun):
            # Try it out.  It should take a three vector and respond with a
            # scalar
            response = gradField(R)
            if response.shape != (3, 3):
                raise StandardError('Magnetic field magnitude function must ' +
                                    'return a matrix.')
            self.gradField= gradField
        else:
            raise TypeError('FieldMag argument must be a callable function')

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

    def internal_FieldMag(self, R):
        return np.linalg.norm(self.Field(R))

    def internal_gradFieldMag(self, R):
        dx, dy, dz = self.return_dx_dy_dz(R)

        return np.array([
            (self.FieldMag(R+dx)-self.FieldMag(R-dx))/2/self.eps,
            (self.FieldMag(R+dy)-self.FieldMag(R-dy))/2/self.eps,
            (self.FieldMag(R+dz)-self.FieldMag(R-dz))/2/self.eps
            ])

    def internal_gradField(self, R):
        dx, dy, dz = self.return_dx_dy_dz(R)

        return np.array([
            (self.Field(R+dx) - self.Field(R-dx))/2/self.eps,
            (self.Field(R+dy) - self.Field(R-dy))/2/self.eps,
            (self.Field(R+dz) - self.Field(R-dz))/2/self.eps
            ])

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_field = magField(lambda R: np.array([-0.5*R[0], -0.5*R[1], 1*R[2]]))

    print(test_field.gradField(np.array([5., 2., 1.])))
