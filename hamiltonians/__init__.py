"""
author: SPE

Some useful hamiltonians for the laser cooling package.
"""
import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
import scipy.constants as cts
from . import XFmolecules

def wig3j(j1, j2, j3, m1, m2, m3):
    """
    This function redefines the wig3jj in terms of things that I like:
    """
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def wig6j(j1, j2, j3, m1, m2, m3):
    """
    This function redefines the wig6jj in terms of things that I like:
    """
    return float(wigner_6j(j1, j2, j3, m1, m2, m3))


def uncoupled_index(J, I, mJ, mI):
    return int((mI+I)+(2*I+1)*(mJ+J))


def hyperfine_uncoupled(J, I, gJ, gI, Ahfs, Bhfs=0, Chfs=0,
                        muB=(cts.value("Bohr magneton in Hz/T")*1e-4),
                        return_basis = False):
    index = lambda J, I, mJ, mI: uncoupled_index(J, I, mJ, mI)

    num_of_states = int((2*J+1)*(2*I+1))
    H_0 = np.zeros((num_of_states, num_of_states))
    H_Bq = np.zeros((3,num_of_states, num_of_states))

    # First, go through and populate the diagonal (magnetic field) elements:
    for mJ in np.arange(-J, J+1, 1):
        for mI in np.arange(-I, I+1, 1):
            H_Bq[1, index(J, I, mJ, mI), index(J, I, mJ, mI)] += \
                (gJ*muB*mJ + gI*muB*mI)

    # Next, do the J_zI_z diagonal elements of J\dotI operator:
    for mJ in np.arange(-J, J+1, 1):
        for mI in np.arange(-I, I+1, 1):
            H_0[index(J, I, mJ, mI), index(J, I, mJ, mI)] += Ahfs*mJ*mI

    # Now, go through and do the J_+I_- term:
    for mJ in np.arange(-J, J, 1):
        for mI in np.arange(-I+1, I+1, 1):
            H_0[index(J, I, mJ+1, mI-1), index(J, I, mJ, mI)] += \
             0.5*Ahfs*np.sqrt((J-mJ)*(J+mJ+1))*np.sqrt((I+mI)*(I-mI+1))

    # Now, go through and do the J_-I_+ term:
    for mJ in np.arange(-J+1, J+1, 1):
        for mI in np.arange(-I, I, 1):
            H_0[index(J, I, mJ-1, mI+1), index(J, I, mJ, mI)] += \
             0.5*Ahfs*np.sqrt((J+mJ)*(J-mJ+1))*np.sqrt((I-mI)*(I+mI+1))

    if Bhfs > 0:
        Bhfs = Bhfs/(2*I*(2*I-1)*J*(2*J-1))  # rescale, include the denominator
        # Next, do the J_zI_z diagonal elements\
        for mJ in np.arange(-J, J+1, 1):
            for mI in np.arange(-I, I+1, 1):
                H_0[index(J, I, mJ, mI), index(J, I, mJ, mI)] += \
                    Bhfs*(
                        # I_z^2J_z^2 from (I\cdotJ)^2
                        3*mJ**2*mI**2 +
                        # J_-J_+I_+I_- from (I\cdotJ)^2
                        3/4*(J*(J+1)-mJ**2-mJ)*(I*(I+1)-mI**2+mI) +
                        # J_+J_-I_-I_+ from (I\cdotJ)^2
                        3/4*(J*(J+1)-mJ**2+mJ)*(I*(I+1)-mI**2-mI) +
                        # J_zI_z from (I\cdotJ)
                        3/2*mJ*mI -
                        # the rest:
                        I*(I+1)*J*(J+1)
                    )

        # Now, go through and do the J_+I_- terms:
        for mJ in np.arange(-J, J, 1):
            for mI in np.arange(-I+1, I+1, 1):
                H_0[index(J, I, mJ+1, mI-1), index(J, I, mJ, mI)] += Bhfs * \
                 (
                     # J_z I_z J_+I_- + J_+I_-J_z I_z term form 3(I\cdotJ)^2:
                     3/2*((mJ+1)*(mI-1)+mJ*mI)*np.sqrt((J-mJ)*(J+mJ+1) *
                                                       (I+mI)*(I-mI+1)) +
                     # J_+I_- term form 3/2(I\cdotJ):
                     3/4*np.sqrt((J-mJ)*(J+mJ+1)*(I+mI)*(I-mI+1))
                 )

        # Now, go through and do the J_-I_+ terms:
        for mJ in np.arange(-J+1, J+1, 1):
            for mI in np.arange(-I, I, 1):
                H_0[index(J, I, mJ-1, mI+1), index(J, I, mJ, mI)] += Bhfs * \
                 (
                     # J_z I_z J_-I_+ + J_-I_+J_z I_z term form 3(I\cdotJ)^2:
                     3/2*((mJ-1)*(mI+1)+mJ*mI)*np.sqrt((J+mJ)*(J-mJ+1) *
                                                       (I-mI)*(I+mI+1)) +
                     # J_+I_- term form 3/2(I\cdotJ):
                     3/4*np.sqrt((J+mJ)*(J-mJ+1)*(I-mI)*(I+mI+1))
                 )

        # Now, go through and do the J_-J_-I_+I_+ terms:
        for mJ in np.arange(-J+2, J+1, 1):
            for mI in np.arange(-I, I-1, 1):
                H_0[index(J, I, mJ-2, mI+2), index(J, I, mJ, mI)] += 3/4*Bhfs*\
                 np.sqrt((J+mJ-1)*(J+mJ)*(J-mJ+1)*(J-mJ+2)) * \
                 np.sqrt((I-mI-1)*(I-mI)*(I+mI+1)*(I+mI+2))

        # Now, go through and do the J_+J_+I_-I_- terms:
        for mJ in np.arange(-J, J-1, 1):
            for mI in np.arange(-I+2, I+1, 1):
                H_0[index(J, I, mJ+2, mI-2), index(J, I, mJ, mI)] += 3/4*Bhfs*\
                 np.sqrt((J-mJ-1)*(J-mJ)*(J+mJ+1)*(J+mJ+2)) * \
                 np.sqrt((I+mI-1)*(I+mI)*(I-mI+1)*(I-mI+2))

    if return_basis:
        basis = np.zeros((4,num_of_states))

        for mJ in range(-J,J+1):
            for mI in range(-I,I+1):
                basis[index(J, I, mJ, mI)] = np.array([J, I, mJ, mI])

        return H_0, H_Bq, basis
    else:
        return H_0, H_Bq


def coupled_index(F, mF, Fmin):
    if np.abs(mF) > F:
        raise ValueError("Not a good mF value")
    return int(np.sum((2*np.arange(Fmin, F, 1)+1))+(F+mF))


def hyperfine_coupled(J, I, gJ, gI, Ahfs, Bhfs=0, Chfs=0,
                      muB=(cts.value("Bohr magneton in Hz/T")*1e-4),
                      return_basis=False):
    """
    Construct the hyperfine Hamiltonian in the coupled basis.
    """
    # Determine the full number of F's:
    Fmin = np.abs(I-J)
    Fmax = np.abs(I+J)

    index = lambda F, mF: coupled_index(F, mF, Fmin)

    # Make the quantum numbers of the basis states:
    num_of_states = int(np.sum(2*np.arange(Fmin, Fmax+0.5, 1)+1))
    Fs = np.zeros((num_of_states,))
    mFs = np.zeros((num_of_states,))
    for F_i in np.arange(Fmin, Fmax+0.5, 1):
        for mF_i in np.arange(-F_i, F_i+0.5, 1):
            Fs[index(F_i, mF_i)] = F_i
            mFs[index(F_i, mF_i)] = mF_i

    # Now, populate the H_0 matrix (field independent part):
    H_0 = np.zeros((num_of_states, num_of_states))

    # Calculate the diagonal elements:
    Ks = Fs*(Fs+1) - I*(I+1) - J*(J+1)
    diag_elem = 0.5*Ahfs*Ks
    if Bhfs!=0:
        diag_elem += Bhfs*(1.5*Ks*(Ks+1) - 2*I*(I+1)*J*(J+1))/\
        (4*I*(2*I-1)*J*(2*J-1))

    if Chfs!=0:
        diag_elem += Chfs*(5*Ks**2*(Ks/4+1)
                           + Ks*(I*(I+1)+J*(J+1)+3-3*I*(I+1)*J*(J+1))
                           - 5*I*(I+1)*J*(J+1))/\
        (I*(I-1)*(2*I-1)*J*(J-1)*(2*J-1))

    # Insert the diagonal (field indepedent part):
    for ii in range(num_of_states):
        H_0[ii,ii] = diag_elem[ii]

    # Now work on the field dependent part:
    H_Bq = np.zeros((3, num_of_states, num_of_states))

    for ii, q in enumerate(range(-1, 2)):
        # TODO: Verify sign of q!
        for F in np.arange(Fmin, Fmax+0.5, 1):
            for Fp in np.arange(Fmin, Fmax+0.5, 1):
                for mF in np.arange(-F, F+0.5, 1):
                    if np.abs(mF+q)<=Fp:
                        H_Bq[ii, index(F, mF), index(Fp, mF + q)] += gJ*muB*\
                        (-1)**(F-mF)*wig3j(F, 1, Fp, mF, q, -(mF+q))*\
                        np.sqrt((2*Fp+1)*(2*F+1))*(-1)**(J+I+Fp+1)*\
                        wig6j(J, F, I, Fp, J, 1)*\
                        np.sqrt(J*(J+1)*(2*J+1))

                        H_Bq[ii, index(F, mF), index(Fp, mF + q)] += gI*muB*\
                        (-1)**(F-mF)*wig3j(F, 1, Fp, mF, q, -(mF+q))*\
                        np.sqrt((2*Fp+1)*(2*F+1))*(-1)**(J+I+Fp+1)*\
                        wig6j(I, F, J, Fp, I, 1)*\
                        np.sqrt(I*(I+1)*(2*I+1))

    if return_basis:
        return H_0, H_Bq, np.vstack((Fs, mFs))
    else:
        return H_0, H_Bq


def singleF(F, gF=1, muB=(cts.value("Bohr magneton in Hz/T")*1e-4),
            return_basis=False):
    """
    The Hamiltonian for a single F state:
    """
    index = lambda mF: int(F+mF)

    # Initialize the matrix
    H_0 = np.zeros((int(2*F+1), int(2*F+1)))
    H_Bq = np.zeros((3, int(2*F+1), int(2*F+1)))

    # No diagonal elements
    # Off-diagonal elemnts:
    for ii, q in enumerate(np.arange(-1, 2, 1)):
        for mF in np.arange(-F, F+1, 1):
            if np.abs(mF+q) <= F:
                H_Bq[ii, index(mF), index(mF + q)] += gF*muB*\
                    (-1)**(F-mF)*np.sqrt(F*(F+1)*(2*F+1))*\
                    wig3j(F, 1, F, mF, q, -(mF+q))

    if return_basis:
        basis = np.zeros((int(2*F+1), 2))
        basis[:, 0] = F
        basis[:, 1] = np.arange(-F, F+1)

        argout = (H_0, H_Bq, basis)
    else:
        argout = (H_0, H_Bq)

    return argout


def dqij_norm(dqij):
    dqij_norm = np.zeros(dqij.shape)
    for ii in range(dqij.shape[0]):
        for jj in range(dqij.shape[2]):
            dqij_norm[ii, :, jj] = dqij[ii, :, jj]/\
                np.linalg.norm(dqij[:, :, jj])

    return dqij_norm


def dqij_two_hyperfine_manifolds(J1, J2, I, normalize=True, return_basis=False):
    """
    Dipole matrix element matrix elements matrix for a dipole matrix element
    transition.
    """
    def matrix_element(J, F, m_F, Jp, Fp, m_Fp, I, q):
        return (-1)**(F-m_F+J+I+Fp+1)*np.sqrt((2*F+1)*(2*Fp+1))*\
            wig3j(F, 1, Fp, m_F, q, -m_Fp)*wig6j(J, F, I, Fp, Jp, 1)

    # A simple function for addressing the index:
    index = lambda Fmin, F, mF: coupled_index(F, mF, Fmin)

    # What's the minimum F1 and F2?
    F1min = np.abs(I-J1)
    F2min = np.abs(I-J2)

    F1max = np.abs(I+J1)
    F2max = np.abs(I+J2)

    dqij = np.zeros((3, int(np.sum(2*np.arange(F1min, F1max+0.5, 1)+1)),
                     int(np.sum(2*np.arange(F2min, F2max+0.5, 1)+1))))

    for ii, q in enumerate(range(-1, 2)):
        for F1 in np.arange(F1min, F1max+0.5, 1):
            for F2 in np.arange(F2min, F2max+0.5, 1):
                for mF in np.arange(-F1, F1+0.5, 1):
                    if not np.abs(mF+q) > F2:
                        dqij[ii, index(F1min, F1, mF),
                             index(F2min, F2, mF+q)] =\
                        matrix_element(J1, F1, mF, J2, F2, mF+q, I, q)

    if normalize:
        dqij = dqij_norm(dqij)

    if return_basis:
        basis_g = np.zeros((dqij.shape[1], 2))
        basis_e = np.zeros((dqij.shape[2], 2))

        for F in np.arange(F1min, F1max+0.5, 1):
            for mF in np.arange(-F, F+0.5, 1):
                basis_g[index(F1min, F, mF), :] = np.array([F, mF])

        for F in np.arange(F2min, F2max+0.5, 1):
            for mF in np.arange(-F, F+0.5, 1):
                basis_e[index(F2min, F, mF), :] = np.array([F, mF])

        argout = (dqij, basis_g, basis_e)
    else:
        argout = dqij

    return argout


def dqij_two_bare_hyperfine(F1, F2, normalize=True):
    """
    Calculates the dqij matrix for two bare hyperfine states:
    """
    # Make a simple function to return the matrix element:
    def matrix_element(F, m_F, Fp, m_Fp, q):
        return (-1)**(F-m_F)*wig3j(F, 1, Fp, m_F, q, -m_Fp)

    # A simple function for addressing the index:
    index = lambda F, mF: int(F+mF)

    # Initialize the matrix:
    dqij = np.zeros((3, int(2*F1+1), int(2*F2+1)))

    # Populate the matrix:
    for ii, q in enumerate(np.arange(-1, 2, 1)):
        for m_F1 in np.arange(-F1, F1+1, 1):
            if not np.abs(m_F1+q) > F2:
                dqij[ii, index(F1, m_F1), index(F2, m_F1 + q)] =\
                    matrix_element(F1, m_F1, F2, m_F1 + q, q)

    # Normalization
    """
    Normalization involves normalzing each transition |g>->|e> to the norm of all the transitions from the excited state sum(|e>->|g>).   That means summing each column and each
    """
    if normalize:
        dqij = dqij_norm(dqij)

    # Return the matrix:
    return dqij
