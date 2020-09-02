import time
import copy
import numpy as np
from scipy.optimize import fsolve

class progressBar(object):
    def __init__(self, decimals=1, fill='█', prefix='Progress:',
                 suffix='', time_remaining_prefix=' time left', length=30,
                 update_rate=0.5):
        self.tic = time.time()
        self.decimals = decimals
        self.fill = fill
        self.length = length
        self.prefix = prefix
        self.suffix = suffix
        self.time_remaining_prefix = time_remaining_prefix
        self.finished = False
        self.max_written_length = 0
        self.last_update = 0.
        self.update_rate = update_rate

    def format_time(self, tic_toc):
        # Print a final time of completion
        if tic_toc>3600:
            time_str = "%d:%02d:%02d" % ((tic_toc)/3600.0,
                                        ((tic_toc)/60.0)%60.0,
                                        (tic_toc)%60.0)
        elif tic_toc>60:
            time_str = "%d:%02d" % ((tic_toc)/60.0,
                                    (tic_toc)%60.0)
        else:
            time_str = "%.2f s" % (tic_toc)

        return time_str

    def print_string(self, string1):
        # Update the maximum length of string written:
        self.max_written_length = max(self.max_written_length, len(string1))
        pad = ''.join([' ']*(self.max_written_length - len(string1)))
        print(string1 + pad, end='\r')

    def update(self, percentage):
        toc = time.time()
        if percentage>0 and percentage<1 and (toc-self.last_update)>self.update_rate:
            percent = ("{0:." + str(self.decimals) + "f}").format(100*percentage)
            filledLength = int(self.length*percentage)
            bar = self.fill*filledLength + '-'*(self.length - filledLength)

            remaining_time = (1-percentage)*((toc-self.tic)/percentage)
            if remaining_time>0:
                time_str = self.format_time(remaining_time)
            else:
                time_str = "0.00 s"
            self.print_string('%s |%s| %s%%%s;%s: %s' %
                              (self.prefix, bar, percent, self.suffix,
                               self.time_remaining_prefix, time_str))
            self.last_update = toc
        elif percentage>=1:
            if not self.finished:
                self.finished = True
                time_str = self.format_time(toc-self.tic)
                self.print_string('Completed in %s.' % time_str)
                print()


def cart2spherical(A):
    return np.array([(A[0]-1j*A[1])/np.sqrt(2), A[2], -(A[0]+1j*A[1])/np.sqrt(2)])

def spherical2cart(A):
    return np.array([1/np.sqrt(2)*(-A[2]+A[0]), 1j/np.sqrt(2)*(A[2]+A[0]), A[1]])

def spherical_dot(A, B):
    return np.tensordot(A, np.array([-1., 1., -1.])*B[::-1], axes=(0, 0))
    #return np.tensordot(A, np.conjugate(B), axes=(0,0))

class base_force_profile():
    def __init__(self, R, V, laserBeams, hamiltonian):
        if not isinstance(R, np.ndarray):
            R = np.array(R)
        if not isinstance(V, np.ndarray):
            V = np.array(V)

        if R.shape[0] != 3 or V.shape[0] != 3:
            raise TypeError('R and V must have first dimension of 3.')

        self.R = copy.copy(R)
        self.V = copy.copy(V)

        self.iterations = np.zeros(R[0].shape, dtype='int64')

        if hamiltonian is None:
            self.Neq = None
        else:
            self.Neq = np.zeros(R[0].shape + (hamiltonian.n,))

        self.f = {}
        for key in laserBeams:
            self.f[key] = np.zeros(R.shape + (len(laserBeams[key].beam_vector),))

        self.f_mag = np.zeros(R.shape)

        self.F = np.zeros(R.shape)

    def store_data(self, ind, Neq, F, F_laser, F_mag):
        if not Neq is None:
            self.Neq[ind] = Neq

        for jj in range(3):
            #self.f[(jj,) + ind] = f[jj]
            self.F[(jj,) + ind] = F[jj]
            for key in F_laser:
                self.f[key][(jj,) + ind] = F_laser[key][jj]

            self.f_mag[(jj,) + ind] = F_mag[jj]


def random_vector():
    a, b = np.random.rand(2)
    th = np.arccos(2*b-1)
    phi = 2*np.pi*a

def random_vector(free_axes=[True, True, True]):
    """
    This function returns a random vector with magnitude 1, in either 1D, 2D
    or 3D, depending on which axes are free to move (defined by the argument
    free_axes)
    """
    if np.sum(free_axes)==1:
        return (np.sign(np.random.rand(1)-0.5)*free_axes).astype('float64')
    elif np.sum(free_axes)==2:
        phi = 2*np.pi*np.random.rand(1)[0]
        x = np.array([np.cos(phi), np.sin(phi)])
        y =np.zeros((3,))
        y[free_axes] = x
        return y
    elif np.sum(free_axes)==3:
        a, b = np.random.rand(2)
        th = np.arccos(2*b-1)
        phi = 2*np.pi*a

        return np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi),
                         np.cos(th)])
    else:
        raise StandardError('free_axes must be a boolean array of length 3.')


def bisectFindChangeValue(fun, initial_guess, args=(), kwargs={},
                          maxiter=1000, tol=1e-9, invert=False, debug=False):

    guess = np.array([initial_guess]).astype('float64')
    boolarray = np.array([0]).astype('bool')
    ind = 0
    i = 0

    while i<maxiter:
        i+=1 #Increment the counter.

        # Run the user-supplied function
        if not invert:
            boolarray[ind] = fun(guess[ind], *args, **kwargs)
        else:
            boolarray[ind] = not fun(guess[ind], *args, **kwargs)

        # Print debug information:
        if (debug):
            print(guess)
            print(boolarray)

        # Check to see if there is a place where the function turns from true to false:
        if guess.size>2:
            # Find the first non captured one:
            ind_noncap = np.argmax(np.invert(boolarray))
            if ind_noncap.size>0:
                if (np.diff(guess[ind_noncap-1:ind_noncap+1])/ \
                    np.mean(guess[ind_noncap-1:ind_noncap+1]))<tol:
                    x = np.mean(guess[ind_noncap-1:ind_noncap+1])
                    break

        # Now, we track our bisection by continually building our array:
        # First condition, we are at the end of the array and we have been captured
        if (boolarray[ind] and boolarray.size==ind+1):
            boolarray = np.append(boolarray,np.array([0]).astype('bool'))
            guess = np.append(guess,2*guess[ind])
            ind = ind+1
        # Second condition, we are at the beginning of the array and we have not been captured
        elif (not(boolarray[ind]) and ind==0):
            boolarray = np.insert(boolarray,0,np.array([0]).astype('bool'))
            guess = np.insert(guess,0,guess[0]/2)
            ind = 0
        # Third condition, we are in the middle of the array and have been captured
        elif boolarray[ind]:
            guess = np.insert(guess,ind+1,(guess[ind]+guess[ind+1])/2)
            boolarray = np.insert(boolarray,ind+1,np.array([0]).astype('bool'))
            ind = ind+1
        elif not(boolarray[ind]):
            guess = np.insert(guess,ind,(guess[ind-1]+guess[ind])/2)
            boolarray = np.insert(boolarray,ind,np.array([0]).astype('bool'))
            ind = ind
        else:
            raise ValueError('bisectFindChangeValue:Unknown condition during bisection')


    if i==maxiter:
        x = np.nan

    return (x, i)


class governingeq(object):
    """
    Base class for a governing equation for atomic motion in a trap.

    Parameters
    ----------
    laserBeams: array_like of laserBeam, dictionary of lists of laser_beams,
    magField:


    """
    def __init__(self, *args, **kwargs):
        r0 = kwargs.pop('r', np.array([0., 0., 0.]))
        v0 = kwargs.pop('v', np.array([0., 0., 0.]))
        self.set_initial_position_and_velocity(r0, v0)

        # Set up a dictionary to store any resulting force profiles.
        self.profile = {}

        # Set the initial sol to zero:
        self.sol = None

        # Set an attribute for the equillibrium position:
        self.r_eq = None

    def set_initial_position_and_velocity(self, r0, v0):
        self.set_initial_position(r0)
        self.set_initial_velocity(v0)

    def set_initial_position(self, r0):
        self.r0 = r0
        self.sol = None

    def set_initial_velocity(self, v0):
        self.v0 = v0
        self.sol = None

    def evolve_motion(self):
        pass

    def force(self):
        pass

    def generate_force_profile(self):
        pass

    def find_equilibrium_position(self, axes=[2], upper_lim=5., lower_lim=-5.,
                                  Npts=51, initial_search=True):
        if self.r_eq is None:
            self.r_eq = np.zeros((3,))

        # Next, find the equilibrium point in z, and evaluate derivatives there:
        r_eqi = np.zeros((3,))
        z = np.linspace(lower_lim, upper_lim, Npts)

        if initial_search:
            for axis in axes:
                v = np.array([np.zeros(z.shape), np.zeros(z.shape), np.zeros(z.shape)])
                r = np.array([np.zeros(z.shape), np.zeros(z.shape), np.zeros(z.shape)])
                r[axis] = z

                default_axis=np.zeros((3,))
                default_axis[axis] = 1.
                self.generate_force_profile(r, v, name='root_search',
                                            default_axis=default_axis)

                z_possible = z[np.where(np.diff(np.sign(
                    self.profile['root_search'].F[axis]))<0)[0]]

                if z_possible.size>0:
                    if z_possible.size>1:
                        ind = np.argmin(z_possible**2)
                    else:
                        ind = 0
                    r_eqi[axis] = z_possible[ind]
                else:
                    r_eqi[axis] = np.nan

                del self.profile['root_search']

        #print('Initial guess: %s' % r_eqi[axes])
        if len(axes)>1:
            def simple_wrapper(r_changing):
                r_wrap = np.zeros((3,))
                r_wrap[axes] = r_changing

                self.set_initial_position_and_velocity(r_wrap, np.array([0.0, 0.0, 0.0]))
                F = self.find_equilibrium_force()

                return np.sum(F**2)

            if np.sum(np.isnan(r_eqi)) == 0:
                # Find the center of the trap:
                result = minimize(simple_wrapper, r_eqi[axes], method='SLSQP')
                if result.success:
                    self.r_eq[axes] = result.x
                else:
                    self.r_eq[axes] = np.nan
            else:
                self.r_eq = np.nan
        else:
            def simple_wrapper(r_changing):
                r_wrap = np.zeros((3,))
                r_wrap[axes] = r_changing

                self.set_initial_position_and_velocity(r_wrap, np.array([0.0, 0.0, 0.0]))
                F = self.find_equilibrium_force()

                return F[axes]

            if np.sum(np.isnan(r_eqi)) == 0:
                self.r_eq[axes] = fsolve(simple_wrapper, r_eqi[axes])[0]
            else:
                self.r_eq[axes] = np.nan

        return self.r_eq

    def trapping_frequencies(self, axes=[0, 2], r=None, eps=0.01):
        self.omega = np.zeros(3,)

        if isinstance(eps, float):
            eps = np.array([eps]*3)

        if r is None and self.r_eq is None:
            r = np.array([0., 0., 0.])
        elif r is None:
            r = self.r_eq

        if hasattr(self, 'mass'):
            mass = self.mass
        else:
            mass = self.hamiltonian.mass

        for axis in axes:
            if not np.isnan(r[axis]):
                rpmdri = np.tile(r, (2,1)).T
                rpmdri[axis, 1] += eps[axis]
                rpmdri[axis, 0] -= eps[axis]

                F = np.zeros((2,))
                for jj in range(2):
                    self.set_initial_position_and_velocity(rpmdri[:, jj],
                                                           np.zeros((3,)))
                    f = self.find_equilibrium_force()

                    F[jj] = f[axis]

                if np.diff(F)<0:
                    self.omega[axis] = np.sqrt(-np.diff(F)/(2*eps[axis]*mass))
                else:
                    self.omega[axis] = 0
            else:
                self.omega[axis] = 0

        return self.omega[axes]

    def damping_coeff(self, axes=[0, 2], r=None, eps=0.01):
        self.beta = np.zeros(3,)

        if isinstance(eps, float):
            eps = np.array([eps]*3)

        if r is None and self.r_eq is None:
            r = np.array([0., 0., 0.])
        elif r is None:
            r = self.r_eq

        for axis in axes:
            if not np.isnan(r[axis]):
                vpmdvi = np.zeros((3,2))
                vpmdvi[axis, 1] += eps[axis]
                vpmdvi[axis, 0] -= eps[axis]

                F = np.zeros((2,))
                for jj in range(2):
                    self.set_initial_position_and_velocity(r, vpmdvi[:, jj])
                    f = self.find_equilibrium_force()

                    F[jj] = f[axis]

                if np.diff(F)<0:
                    self.beta[axis] = -np.diff(F)/(2*eps[axis])
                else:
                    self.beta[axis] = 0
            else:
                self.beta[axis] = 0

        return self.beta[axes]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vectors = []
    for n in range(500):
        vectors.append(random_vector([True, True, True]))

    vectors = np.array(vectors)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vectors[:,0], vectors[:,1], vectors[:,2])
    ax.view_init(elev=-90., azim=0.)
