import numpy as np

class magField(object):
    def __init__(self, func, FieldMag=None, gradFieldMag=None,
                 gradFieldFunc=None, eps=1e-5):

        R = np.random.rand(3) # Pick a random point for testing

        if iscallable(func):
            # Try it out:
            response = func(R)
            if len(response) != 3:
                raise StandardError('Magnetic field function must return a vector.')
            self.Field = func
        else:
            raise TypeError('Magnetic field must be a callable function')

        if FieldMag is None:
            self.FieldMag = self.internal_FieldMag
        elif iscallable(mag_fun):
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
        elif iscallable(mag_fun):
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
        elif iscallable(mag_fun):
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
        if R.shape == (3,)
            dx = np.array([eps, 0., 0.])
            dy = np.array([0., eps, 0.])
            dz = np.array([0., 0., eps])
        else:
            dx = np.zeros(R.shape)
            dy = np.zeros(R.shape)
            dz = np.zeros(R.shape)

            dx[0] = self.eps
            dy[1] = self.eps
            dz[2] = self.eps

        return dx, dy, dz

    def internal_FieldMag(self, R):
        return np.linalg.norm(self.magField(R))

    def internal_gradMagFieldMag(self, R):
        dx, dy, dz = self.return_dx_dy_dz(R)

        return np.array([
            (self.FieldMag(self, R+dx)-self.FieldMag(self, R-dx))/2/self.eps,
            (self.FieldMag(self, R+dy)-self.FieldMag(self, R-dy))/2/self.eps,
            (self.FieldMag(self, R+dz)-self.FieldMag(self, R-dz))/2/self.eps
            ])

    def internal_gradField(self, R):
        dx, dy, dz = self.return_dx_dy_dz(R)

        return np.array([
            (self.Field(R+dx) - self.Field(R+dx))/2/eps,
            (self.Field(R+dy) - self.Field(R+dy))/2/eps,
            (self.Field(R+dz) - self.Field(R+dz))/2/eps
            ])
