import numpy as np


# Next, define a Hamiltonian class to work out the internal states:
class hamiltonian():
    """
    This class assembles the full Hamiltonian from the various pieces and does
    necessary manipulations on that Hamiltonian.
    """
    class block():
        def __init__(self, label, M):
            self.label = label
            self.matrix = M

            self.n = M.shape[0]
            self.m = M.shape[1]

        def return_block_in_place(self, i, j, N):
            super_M = np.zeros((N, N), dtype='complex128')
            super_M[i:i+self.n, j:j+self.m] = self.matrix

            return super_M

        def __repr__(self):
            return '(%s %dx%d)' % (self.label, self.n, self.m)

        def __str__(self):
            return '(%s %dx%d)' % (self.label, self.n, self.m)


    class vector_block(block):
        def __init__(self, label, M):
            super().__init__(label, M)

            self.n = M.shape[1]
            self.m = M.shape[2]

        def return_block_in_place(self, i, j, N):
            super_M = np.zeros((3, N, N), dtype='complex128')
            super_M[:, i:i+self.n, j:j+self.m] = self.matrix

            return super_M


    def __init__(self, *args):
        """
        Initializes the class and saves certain elements of the Hamiltonian.
        """
        self.blocks = []
        self.state_labels = []
        self.ns = []
        self.laser_keys = {}

        if len(args) == 5:
            self.add_H_0_block('g', args[0])
            self.add_H_0_block('e', args[1])
            self.add_mu_q_block('g', args[2])
            self.add_mu_q_block('e', args[3])
            self.add_d_q_block('g', 'e', args[4])
        elif len(args)>2:
            raise ValueError('Unknown nummber of arguments.')
        elif len(args)>0:
            raise NotImplementedError('Not yet programmed for %d arguments.' %
                                      len(args))

    def print_structure(self):
        print(self.blocks)

    def recompute_number_of_states(self):
        self.n = 0
        for block in np.diag(self.blocks):
            if isinstance(block, tuple):
                self.n += block[0].n
            else:
                self.n += block.n

    def search_elem_label(self, label):
        ind = ()
        for ii, row in enumerate(self.blocks):
            for jj, element in enumerate(row):
                if isinstance(element, self.block):
                    if element.label == label:
                        ind = (ii, jj)
                        break;
                elif isinstance(element, tuple):
                    if np.any([element_i.label == label for element_i in element]):
                        ind = (ii, jj)
                        break;

        return ind

    def make_elem_label(self, type, state_label):
        if type is 'H_0' or type is 'mu_q':
            if not isinstance(state_label, str):
                raise TypeError('For type %s, state label %s must be a' +
                                ' string.' % (type,state_label))
            return '<%s|%s|%s>' % (state_label, type, state_label)
        elif type is 'd_q':
            if not isinstance(state_label, list):
                raise TypeError('For type %s, state label %s must be a' +
                                ' list of two strings.' % (type, state_label))
            return '<%s|%s|%s>' % (state_label[0], type, state_label[1])
        else:
            raise ValueError('Matrix element type %s not understood' % type)


    def add_new_row_and_column(self):
        if len(self.blocks) == 0:
            self.blocks = np.empty((1,1), dtype=object)
        else:
            blocks = np.empty((self.blocks.shape[0]+1,
                               self.blocks.shape[1]+1), dtype=object)
            blocks[:-1, :-1] = self.blocks
            self.blocks = blocks


    def add_H_0_block(self, state_label, H_0):
        if H_0.shape[0] != H_0.shape[1]:
            raise ValueError('H_0 must be square.')

        ind_H_0 = self.search_elem_label(self.make_elem_label('H_0', state_label))
        ind_mu_q = self.search_elem_label(self.make_elem_label('mu_q', state_label))

        label = self.make_elem_label('H_0', state_label)
        if not ind_H_0 and not ind_mu_q:
            self.add_new_row_and_column()
            self.blocks[-1, -1] = self.block(label, H_0.astype('complex128'))
            self.state_labels.append(state_label)
            self.ns.append(H_0.shape[0])
        elif ind_mu_q:
            if H_0.shape[0] != self.blocks[ind_H_0].n:
                raise ValueError('Element %s is not the right shape to match mu_q.' % label)
            self.blocks[ind_mu_q] = (self.block(label, H_0.astype('complex128')),
                                     self.blocks[ind_mu_q])
        else:
            raise ValueError('H_0 already added.')

        self.recompute_number_of_states()


    def add_mu_q_block(self, state_label, mu_q):
        if mu_q.shape[0] != 3 or mu_q.shape[1] != mu_q.shape[2]:
            raise ValueError('mu_q must 3xnxn, where n is an integer.')

        ind_H_0 = self.search_elem_label(self.make_elem_label('H_0', state_label))
        ind_mu_q = self.search_elem_label(self.make_elem_label('mu_q', state_label))

        label = self.make_elem_label('mu_q', state_label)
        if not ind_H_0 and not ind_mu_q:
            self.add_new_row_and_column()
            self.blocks[-1, -1] = self.vector_block(label, mu_q.astype('complex128'))
            self.state_labels.append(state_label)
            self.ns.append(mu_q.shape[1])
        elif ind_H_0:
            if mu_q.shape[1] != self.blocks[ind_H_0].n:
                raise ValueError('Element %s is not the right shape to match H_0.' % label)
            self.blocks[ind_H_0] = (self.blocks[ind_H_0],
                                   self.vector_block(label, mu_q.astype('complex128')))
        else:
            raise ValueError('mu_q already added.')

        self.recompute_number_of_states()


    def add_d_q_block(self, label1, label2, d_q):
        ind_H_0 = self.search_elem_label(self.make_elem_label('H_0', label1))
        ind_mu_q = self.search_elem_label(self.make_elem_label('mu_q', label1))

        if ind_H_0 == () and ind_mu_q == ():
            raise ValueError('Label %s not found.' % label1)
        elif ind_H_0 == ():
            ind1 = ind_mu_q[0]
            n = self.blocks[ind_mu_q].n
        elif ind_mu_q == ():
            ind1 = ind_H_0[0]
            n = self.blocks[ind_H_0].n
        else:
            ind1 = ind_H_0[0]
            n = self.blocks[ind_H_0][0].n

        ind_H_0 = self.search_elem_label(self.make_elem_label('H_0', label2))
        ind_mu_q = self.search_elem_label(self.make_elem_label('mu_q', label2))

        if ind_H_0 == () and ind_mu_q == ():
            raise ValueError('Label %s not found.' % label1)
        elif ind_H_0 == ():
            ind2 = ind_mu_q[0]
            m = self.blocks[ind_mu_q].n
        elif ind_mu_q == ():
            ind2 = ind_H_0[0]
            m = self.blocks[ind_H_0].n
        else:
            ind2 = ind_H_0[0]
            m = self.blocks[ind_H_0][0].n

        ind = (ind1, ind2)
        label = self.make_elem_label('d_q', [label1, label2])

        if d_q.shape[1]!=n or d_q.shape[2]!=m:
            raise ValueError('Expected size 3x%dx%d for %s, instead see 3x%dx%d'%
                             (n, m, label, d_q.shape[1], d_q.shape[2]))

        self.blocks[ind] = self.vector_block(label, d_q.astype('complex128'))

        label = self.make_elem_label('d_q', [label2, label1])
        self.blocks[ind[::-1]] = self.vector_block(
            label,
            np.array([np.conjugate(d_q[ii].T) for ii in range(3)]).astype('complex128')
            )

        if ind1<ind2:
            self.laser_keys[label1 + '->' + label2] = ind
        else:
            self.laser_keys[label2 + '->' + label1] = ind[::-1]


    def make_full_matrices(self):
        """
        Returns the full nxn matrices (H_0, mu_q, d_q) that define the
        Hamiltonian.
        """
        # Initialize the field-independent component of the Hamiltonian.
        self.H_0 = np.zeros((self.n, self.n), dtype='complex128')
        self.mu_q = np.zeros((3, self.n, self.n), dtype='complex128')

        n = 0
        m = 0

        # First, return H_0 and mu_q:
        for diag_block in np.diag(self.blocks):
            if isinstance(diag_block, self.vector_block):
                self.mu_q += diag_block.return_block_in_place(n, n, self.n)
                n+=diag_block.n
            elif isinstance(diag_block, self.block):
                self.H_0 += diag_block.return_block_in_place(n, n, self.n)
                n+=diag_block.n
            else:
                self.H_0 += diag_block[0].return_block_in_place(n, n, self.n)
                self.mu_q += diag_block[1].return_block_in_place(n, n, self.n)
                n+=diag_block[0].n

        self.d_q_bare = {}

        # Next, return d_q:
        for ii in range(self.blocks.shape[0]):
            for jj in range(ii+1, self.blocks.shape[1]):
                if not self.blocks[ii, jj] is None:
                    key = self.state_labels[ii] + '->' + self.state_labels[jj]
                    nstart = int(np.sum(self.ns[:ii]))
                    mstart = int(np.sum(self.ns[:jj]))
                    self.d_q_bare[key] = self.blocks[ii, jj].return_block_in_place(nstart, mstart, self.n)

        self.d_q_star = {}
        for key in self.d_q_bare.keys():
            self.d_q_star[key] = np.zeros(self.d_q_bare[key].shape, dtype='complex128')
            for kk in range(3):
                self.d_q_star[key][kk] = np.conjugate(self.d_q_bare[key][kk].T)

        # Finally, put together the full d_q, irrespective of laser beam key:
        self.d_q = np.zeros((3, self.n, self.n), dtype='complex128')
        for key in self.d_q_bare.keys():
            self.d_q += self.d_q_bare[key] + self.d_q_star[key]

        return self.H_0, self.mu_q, self.d_q_bare, self.d_q_star


    def return_full_H(self, Eq, Bq):
        if isattr('H_0', self):
            self.make_full_matrices()

        H = np.zeros((self.n, self.n), dtype='complex128')

        H += self.H_0 + np.tensordot(np.array([-1, 1, -1])*Bq[::-1],
                                    self.mu_q, axes=(0, 0))

        for key in self.d_q.keys():
            H += 0.5*np.tensordot(np.conjugate(Eq[key]),
                                  self.d_q_bare[key], axes=(0, 0)) -\
                 0.5*np.tensordot(Eq[key],
                                  self.d_q_star[key], axes=(0, 0))

        return H


    def diag_static_field(self, B):
        """
        This function diagonalizes the H_0 blocks separately based on the values
        for the static magnetic field, and rotates d_q accordingly.  This is
        necessary for the rate equations.
        """
        already_diagonal = True

        # If it does not already exist, make an empty Hamiltonian that has
        # the same dimensions as this one.
        if not hasattr(self, 'rotated_hamiltonian'):
            self.rotated_hamiltonian = hamiltonian()
            for ii, block in enumerate(np.diagonal(self.blocks)):
                self.rotated_hamiltonian.add_H_0_block(
                    self.state_labels[ii],
                    np.zeros((self.ns[ii], self.ns[ii]), dtype='complex128')
                    )
            for ii, block_row in enumerate(self.blocks):
                for jj, block in enumerate(block_row):
                    if jj>ii:
                        if not block is None:
                            self.rotated_hamiltonian.add_d_q_block(
                                self.state_labels[ii], self.state_labels[jj],
                                block.matrix,
                                )

        # Now we get to the meat of it:
        if isinstance(B, float) or isinstance(B, int):
            Bq = np.array([0, B, 0], dtype=float) # Assume B is along z
        elif isinstance(B, np.ndarray) and B.size == 1:
            Bq = np.array([0, B[0], 0]) # Assume B is along z
        elif isinstance(B, np.ndarray) and B.shape[0] == 3:
            Bq = np.zeros((3,), dtype='complex128')

            Bq[0] = -B[0]/np.sqrt(2)+1j*B[1]/np.sqrt(2)
            Bq[1] = B[2]
            Bq[2] = B[0]/np.sqrt(2)-1j*B[1]/np.sqrt(2)

        U = np.empty((self.blocks.shape[0],), dtype=object)

        for ii, diag_block in enumerate(np.diag(self.blocks)):
            if isinstance(diag_block, tuple):
                H = (diag_block[0].matrix +
                     np.tensordot(np.array([-1, 1, -1])*Bq[::-1],
                                  diag_block[1].matrix, axes=(0, 0))
                     )
            elif isinstance(diag_block, self.vector_block):
                H = np.tensordot(np.array([-1, 1, -1])*Bq[::-1],
                                 diag_block.matrix, axes=(0, 0))
            else:
                H = diag_block.matrix

            # Check to see if it even needs to be diagnalized:
            if np.count_nonzero(H - np.diag(np.diagonal(H))) == 0:
                Es = H
                if isinstance(diag_block, tuple):
                    U[ii] = np.eye(diag_block[0].n)
                else:
                    U[ii] = np.eye(diag_block.n)
            else:
                Es, U[ii] = np.linalg.eig(H)

                # Sort the  output:
                ind_e = np.argsort(Es)
                Es = Es[ind_e]
                U[ii] = U[ii][:, ind_e]

                Es = np.diag(Es)

                already_diagonal=False

            # Check to make sure the diganolization resulted in only real
            # components:
            if np.allclose(np.imag(np.diagonal(Es)), 0.):
                self.rotated_hamiltonian.blocks[ii, ii].matrix = np.real(Es)
            else:
                raise ValueError('You broke the Hamiltonian!')

        # Now, rotate the d_q:
        if not already_diagonal:
            for ii in range(self.blocks.shape[0]):
                for jj in range(ii+1, self.blocks.shape[1]):
                    if not self.blocks[ii, jj] is None:
                        for kk in range(3):
                            self.rotated_hamiltonian.blocks[ii, jj].matrix[kk] = \
                            U[ii].T @ self.blocks[ii,jj].matrix[kk] @ U[jj]

                            self.rotated_hamiltonian.blocks[jj, ii].matrix[kk] = \
                            np.conjugate(self.rotated_hamiltonian.blocks[ii, jj].matrix[kk].T)

                            if (self.rotated_hamiltonian.blocks[ii, jj].matrix.shape !=
                                self.blocks[ii,jj].matrix.shape):
                               raise ValueError("Rotataed d_q not the same size as original.")

        return self.rotated_hamiltonian, already_diagonal


    def diag_H_0(self, B0):
        """
        A method to diagonalize the H_0 basis and recompute d_q and Bijq.  This
        is useful for internal hamilotains that are not diagonal at zero field.
        """
        pass


def forceFromBeamsSimple(R, V, laserBeams, magField, extra_beta = None,
                       normalize=False, average_totbeta=False):
    """
    Returns the approximate force from beams assuming a simple F=0 -> F=1
    transition and even simpler approximations regarding staturation.
    """
    # Do some simple error checking on the inputs
    if not isinstance(R, (list, np.ndarray)):
        raise TypeError("R must be a list or ndarray.")
    if not isinstance(V, (list, np.ndarray)):
        raise TypeError("V must be a list or ndarray.")

    if isinstance(R, list):
        R = np.array(R)
    if isinstance(V, list):
        V = np.array(V)

    if R.shape[0] > 3:
        raise ValueError("dimension 0 of R must be <=3")
    if R.shape != V.shape:
        raise ValueError("length of R must be equal to V")

    # Finally, check the last three arguments:
    if not isinstance(laserBeams, list):
        raise TypeError("laserBeams must be a list.")
    elif not isinstance(laserBeams[0], laserBeam):
        raise TypeError("Each element of laserBeams must be a laserBeam class.")
    if not callable(magField):
        raise TypeError("magField must be a function")
    if extra_beta is not None:
        if not callable(extra_beta):
            raise TypeError("extra_beta must be a function")

    # Do we normalize the force to N infinitely powered laser beams of 1?
    if normalize:
        norm = float(len(laserBeams))
    else:
        norm = 1

    # Number of dimensions:
    dim = R.shape[0]

    # First, go through the beams and calculate the total beta:
    betatot = np.zeros(R.shape[1::])
    for beam in laserBeams:
        betatot += beam.return_beta(R)

    if average_totbeta:
        betatot = betatot/len(laserBeams)

    # If extraBeta is present, call it:
    if extra_beta is not None:
        betatot += extra_beta(R)

    # Calculate the magnetic field:
    B = magField(R)

    # Calculate the Bhat direction:
    # Calculate its magnitude:
    Bmag = np.linalg.norm(B, axis=0)

    # Calculate the Bhat direction:
    Bhat = np.zeros(B.shape)
    Bhat[-1] = 1.0 # Make the default magnetic field direction z
    for ii in range(Bhat.shape[0]):
        Bhat[ii][Bmag!=0] = B[ii][Bmag!=0]/Bmag[Bmag!=0]

    # Preallocate memory for F:
    F = np.zeros(R.shape)

    # Run through all the beams:
    for beam in laserBeams:
        # If kvec is callable, evaluate kvec:
        if callable(beam.kvec):
            kvec = beam.kvec(R)
            # Just like magField(R), beam.kvec(R) is expecting to return a
            # three element vector the same size as R.  It is also expected
            # to be normalized appropriately.
        else:
            kvec = beam.kvec

        # Project the polarization on the quantization axis
        proj = np.abs(beam.project_pol(Bhat))**2

        # Compute the intensity of the beam:
        beta = beam.return_beta(R)

        # Compute the jth component of the force:
        for jj in range(dim):
            # Compute the component from the Delta m = +1,0,-1 component:
            for ii in range(-1, 2, 1):
                F[jj] += norm*proj[ii+1]*kvec[jj]*beta/2\
                         /(1 + betatot + 4*(beam.delta - kvec[jj]*V[jj]\
                           - ii*Bmag)**2)

    # Return the force array:
    return F


# %%
if __name__ == '__main__':
    """
    A simple test of the Hamiltonian class.
    """
    Hg, mugq = hamiltonians.singleF(F=1, muB=1)
    He, mueq = hamiltonians.singleF(F=2, muB=1)
    d_q = pylcp.hamiltonians.dqij_two_bare_hyperfine(1, 2)

    ham1 = hamiltonian()
    ham1.add_H_0_block('g', Hg)
    ham1.add_mu_q_block('g', mugq)
    print(ham1.blocks)
    ham1.add_H_0_block('e', He)
    ham1.add_mu_q_block('e', mueq)
    print(ham1.blocks)
    ham1.add_d_q_block('g', 'e', d_q)
    print(ham1.blocks)

    ham1.make_full_matrices()
    ham1.diag_static_field(np.array([0, 0.5, 0]))
