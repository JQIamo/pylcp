import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
import scipy.constants as cts
from . import XFmolecules

def wig3j(j1, j2, j3, m1, m2, m3):
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def wig6j(j1, j2, j3, m1, m2, m3):
    return float(wigner_6j(j1, j2, j3, m1, m2, m3))


def uncoupled_index(J, I, mJ, mI):
    return int((mI+I)+(2*I+1)*(mJ+J))


def fine_structure_uncoupled(L, S, I, xi, a_c, a_orb, a_dip, gL, gS, gI,
                             muB=(cts.value("Bohr magneton in Hz/T")*1e-4),
                             return_basis=False):
    """
    Returns the full fine structure manifold in the uncoupled basis.

    Parameters
    ----------
    L : int
        Orbital angular momentum of interest
    S : int or float
        Spin angular momentum of interest
    I : int or float
        Nuclear angular momentum of interest
    xi : float
        Fine structure splitting
    a_c : float
        Contact interaction constant
    a_orb : float
        Orbital interaction constant
    a_dip : float
        Dipole interaction constant
    gL : float
        Orbital g-factor
    gS : float
        Spin g-factor
    gI : float
        Nuclear g-factor
    muB : float, optional
        Bohr magneton.  Default: the CODATA value in Hz/G
    return_basis : bool, optional
        Return the basis vectors as well as gthe

    Returns
    -------
    H_0 : array (NxN)
        Field free Hamiltonian, where N is the number of states
    mu_q : array (3xNxN)
        Zeeman splitting array

    Notes
    -----
    See J.D.Lyons and T.P.Das, Phys.Rev.A,2,2250 (1970) and
    H.Orth et al,Z.Physik A,273,221 (1975) for details of Hamiltonian
    and splitting constants.

    This function is adapted from the one found in Tollet, "Permanent magnetic
    trap for Li atoms", thesis, Rice University, 1994.
    """
    # Set up a basis
    basis = []
    for mL in np.arange(-L, L+1):
        for mS in np.arange(-S, S+1):
            for mI in np.arange(-I, I+1):
                basis.append((mL, mS, mI))

    n_basis = len(basis)
    mu_q = np.zeros((3, n_basis, n_basis))

    # Start with the magnetic field dependent matrices:
    for kk, q in enumerate([-1, 0, 1]):
        for jj, (mLp, mSp, mIp) in enumerate(basis):
            for ii, (mL, mS, mI) in enumerate(basis):
                if mS==mSp and mIp==mI:
                    mu_q[kk, ii, jj] += gL*muB*(-1)**(L-mL)*wig3j(L, 1, L, -mL, q, mLp)*np.sqrt(L*(L+1)*(2*L+1))
                if mL==mLp and mIp==mI:
                    mu_q[kk, ii, jj] += gS*muB*(-1)**(S-mS)*wig3j(S, 1, S, -mS, q, mSp)*np.sqrt(S*(S+1)*(2*S+1))
                if mL==mLp and mSp==mS:
                    mu_q[kk, ii, jj] -= gI*muB*(-1)**(I-mI)*wig3j(I, 1, I, -mI, q, mIp)*np.sqrt(I*(I+1)*(2*I+1))

    # Need to define the OTHER mu_q matrices!

    # Next, fill in the field independent matrix
    # terms, dmL dmS dmI
    # (I)     0   0   0  (diagonal terms)
    # (II)   +1  -1   0
    #        -1  +1   0
    # (III)   0  +1  -1
    #         0  -1  +1
    # (IV)   +1   0  -1
    #        -1   0  +1
    # (V)    +2  -1  -1
    #        -2  +1  +1

    # Initialize field-independent matrix:
    H_0 = np.zeros((n_basis, n_basis))

    # Populate!
    for ii, (mL, mS, mI) in enumerate(basis):
        #  (I)
        if S>0:
            H_0[ii, ii] += mS*mI*(L+S)*a_c/S
        if S>0 and L>0:
            H_0[ii, ii] += (mL*mI*(L+S)*a_orb/L
                            + mS*mI*(3*mL**2 - L*(L+1))*(L+S)*a_dip/(1*S*(2*L-1))
                            + mL*mS*xi)

        # (II)
        if mL+1<=L and mS-1>=-S and L>0:
            t1 = np.sqrt((L-mL)*(L+mL+1)*(S+mS)*(S-mS+1))
            drow = int(np.round(2*S*(2*I+1)))
            H_0[ii+drow, ii] += t1*xi/2 + t1*3*(2*mL+1)*mI*(L+S)*a_dip/(4*L*S*(2*L-1))

        if mS+1<=S and mL-1>=-L and L>0:
            t1 = np.sqrt((S-mS)*(S+mS+1)*(L+mL)*(L-mL+1))
            drow = int(np.round(-2*S*(2*I+1)))
            H_0[ii+drow, ii] += t1*xi/2 + t1*3*(2*mL-1)*mI*(L+S)*a_dip/(4*L*S*(2*L-1))

        # (III)
        if mS+1<=S and mI-1>=-I:
            t1 = np.sqrt((S-mS)*(S+mS+1)*(I+mI)*(I-mI+1))
            drow = int(np.round(2*I));
            if np.abs(a_c)>0.:
                H_0[ii+drow, ii] += t1*(L+S)*a_c/2/S
            if L>0 and np.abs(a_dip)>0.:
                H_0[ii+drow, ii] += - t1*(3*mL**2-L*(L+1))*(L+S)*a_dip/(4*L*S*(2*L-1))

        if mI+1<=I and mS-1>=-S:
            t1 = np.sqrt((I-mI)*(I+mI+1)*(S+mS)*(S-mS+1))
            drow = int(np.round(-2*I))
            if np.abs(a_c)>0.:
                H_0[ii+drow, ii] += t1*(L+S)*a_c/2/S
            if L>0 and np.abs(a_dip)>0.:
                H_0[ii+drow, ii] += - t1*(3*mL**2-L*(L+1))*(L+S)*a_dip/(4*L*S*(2*L-1))

        # (IV)
        if mL+1<=1 and mI-1>=-I and L>0 and (np.abs(a_orb)>0. or np.abs(a_dip)>0.):
            t1 = np.sqrt((L-mL)*(L+mL+1)*(I+mI)*(I-mI+1))
            drow = int((2*I+1)*(2*S+1) - 1)
            H_0[ii+drow, ii] += t1*(1+S)*a_orb/2/L + t1*3*(2*mL+1)*mS*(1+S)*a_dip/(4*L*S*(2*L-1))

        if mI+1<=I and mL-L>=-1 and L>0 and (np.abs(a_orb)>0. or np.abs(a_dip)>0.):
            t1 = np.sqrt((I-mI)*(I+mI+1)*(L+mL)*(L-mL+1))
            drow = int(1 - (2*I+1)*(2*S+1))
            H_0[ii+drow, ii] += t1*(1+S)*a_orb/2/L + t1*3*(2*mL-1)*mS*(1+S)*a_dip/(4*L*S*(2*L-1))

        # (V)
        if mL+2<=L and mS-1>=-S and mI-1>=-I and np.abs(a_dip)>0. and L>0 and S>0:
            t1 = (L-mL)*(L+mL+1)
            t1 = t1*np.sqrt( (S+mS)*(S-mS+1)*(I+mI)*(I-mI+1) )
            drow = int(2*(2*I+1)*(2*S+1) - (2*I+1) - 1)
            H_0[ii+drow, ii] += t1*3*(1+S)*a_dip/(4*L*S*(2*L-1))
        if mL-2>=-L and mS+1<=S and mI+1<=I and np.abs(a_dip)>0. and L>0 and S>0:
            t1 = (L+mL)*(L-mL+1)
            t1 = t1*np.sqrt( (S-mS)*(S+mS+1)*(I-mI)*(I+mI+1) )
            drow = int(-2*(2*I+1)*(2*S+1) + (2*I+1) + 1)
            H_0[ii+drow, ii] += t1*3*(1+S)*a_dip/(4*L*S*(2*L-1))

    if return_basis:
        return H_0, mu_q, basis
    else:
        return H_0, mu_q


def dqij_two_fine_stucture_manifolds_uncoupled(basis_g, basis_e):
    """
    Return the coupling between two fine structure manifolds

    Parameters
    ----------
    basis_g : list or array_like
        A list of the basis vectors for the ground state.  In the uncoupled
        basis, they are of the form :math:`|m_L, m_S, m_I\\rangle`
    basis_e : list or array_like
        A list of the basis vectors for the ground state.  In the uncoupled
        basis, they are of the form :math:`|m_L\', m_S\', m_I\'\\rangle`

    Returns
    -------
    d_q : array with shape (3, N, M)
        The dipole coupling array.  N is the number of ground states and M
        is the number of excited states.
    """
    d_q = np.zeros((3, len(basis_g), len(basis_e)))
    for kk, q in enumerate(range(-1, 2)):
        for ii, (mL, mS, mI) in enumerate(basis_g):
            for jj, (mLp, mSp, mIp) in enumerate(basis_e):
                d_q[kk, ii, jj] = (mI==mIp)*(mS==mSp)*(mL==mLp+q)

    return d_q


def hyperfine_uncoupled(J, I, gJ, gI, Ahfs, Bhfs=0, Chfs=0,
                        muB=(cts.value("Bohr magneton in Hz/T")*1e-4),
                        return_basis = False):
    """
    Construct the hyperfine Hamiltonian in the coupled basis.

    For parameterization of this Hamiltonian, see Steck, Alkali D line data,
    which contains a useful description of the hyperfine Hamiltonian.

    Parameters
    ----------
    J : int or float
        Lower hyperfine manifold :math:`J` quantum number
    I : int or float
        Nuclear spin associated with both manifolds
    gJ : float
        Electronic Lande g-factor
    gI : float
        Nuclear g-factor
    Ahfs : float
        Hyperfine :math:`A` parameter
    Bhfs : float, optional
        Hyperfine :math:`B` parameter. Default: 0.
    Chfs : float, optional
        Hyperfine :math:`C` parameter. Default: 0.
    muB : float, optional
        Bohr magneton.  Default: the CODATA value in Hz/G
    return_basis : boolean, optional
        If true, return the basis.  Default: False

    Returns
    -------
    H_0 : array_like
        Field independent component of the Hamiltonian
    mu_q : array_like
        Magnetic field dependent component of the Hamiltonian
    basis : list
        List of :math:`(J, I, m_J, m_I)` basis states
    """
    index = lambda J, I, mJ, mI: uncoupled_index(J, I, mJ, mI)

    num_of_states = int((2*J+1)*(2*I+1))
    H_0 = np.zeros((num_of_states, num_of_states))
    H_Bq = np.zeros((3,num_of_states, num_of_states))

    # Start with the magnetic field dependent matrices:
    for kk, q in enumerate([-1, 0, 1]):
        for mJ in np.arange(-J, J+0.1, 1):
            for mJp in np.arange(-J, J+0.1, 1):
                for mI in np.arange(-I, I+0.1, 1):
                    for mIp in np.arange(-I, I+0.1, 1):
                        if mIp==mI:
                            mu_q[kk, index(J, I, mJ, mI), index(J, I, mJp, mIp)] += \
                            gJ*muB*(-1)**(J-mJ)*wig3j(J, 1, J, -mJ, q, mJp)*np.sqrt(J*(J+1)*(2*J+1))
                        if mJ==mJp:
                            mu_q[kk, index(J, I, mJ, mI), index(J, I, mJp, mIp)] -= \
                            gI*muB*(-1)**(I-mI)*wig3j(I, 1, I, -mI, q, mIp)*np.sqrt(I*(I+1)*(2*I+1))

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
        raise ValueError("mF=%.1f not a good value for F=%.1f"%(mF, F))
    return int(np.sum((2*np.arange(Fmin, F, 1)+1))+(F+mF))


def hyperfine_coupled(J, I, gJ, gI, Ahfs, Bhfs=0, Chfs=0,
                      muB=(cts.value("Bohr magneton in Hz/T")*1e-4),
                      return_basis=False):
    """
    Construct the hyperfine Hamiltonian in the coupled basis.

    For parameterization of this Hamiltonian, see Steck, Alkali D line data,
    which contains a useful description of the hyperfine Hamiltonian.

    Parameters
    ----------
    J : int or float
        Lower hyperfine manifold :math:`J` quantum number
    I : int or float
        Nuclear spin associated with both manifolds
    gJ : float
        Electronic Lande g-factor
    gI : float
        Nuclear g-factor
    Ahfs : float
        Hyperfine :math:`A` parameter
    Bhfs : float, optional
        Hyperfine :math:`B` parameter. Default: 0.
    Chfs : float, optional
        Hyperfine :math:`C` parameter. Default: 0.
    muB : float, optional
        Bohr magneton.  Default: the CODATA value in Hz/G
    return_basis : boolean, optional
        If true, return the basis.  Default: False

    Returns
    -------
    H_0 : array_like
        Field independent component of the Hamiltonian
    mu_q : array_like
        Magnetic field dependent component of the Hamiltonian
    basis : list
        List of :math:`(F, m_F)` basis states
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
    mu_q = np.zeros((3, num_of_states, num_of_states))

    for ii, q in enumerate(range(-1, 2)):
        for Fp in np.arange(Fmin, Fmax+0.5, 1):
            for F in np.arange(Fmin, Fmax+0.5, 1):
                for mFp in np.arange(-Fp, Fp+0.5, 1):
                    mF = mFp+q
                    if not np.abs(mF)>F:
                        mu_q[ii, index(F, mF), index(Fp, mFp)] -= gJ*muB*\
                        (-1)**np.abs(F-mF)*wig3j(F, 1, Fp, -mF, q, mFp)*\
                        np.sqrt((2*Fp+1)*(2*F+1))*(-1)**(J+I+Fp+1)*\
                        wig6j(J, Fp, I, F, J, 1)*\
                        np.sqrt(J*(J+1)*(2*J+1))

                        mu_q[ii, index(F, mF), index(Fp, mFp)] += gI*muB*\
                        (-1)**np.abs(F-mF)*wig3j(F, 1, Fp, -mF, q, mFp)*\
                        np.sqrt((2*Fp+1)*(2*F+1))*(-1)**(J+I+Fp+1)*\
                        wig6j(I, Fp, J, F, I, 1)*\
                        np.sqrt(I*(I+1)*(2*I+1))

    if return_basis:
        return H_0, mu_q, np.vstack((Fs, mFs))
    else:
        return H_0, mu_q


def singleF(F, gF=1, muB=(cts.value("Bohr magneton in Hz/T")*1e-4),
            return_basis=False):
    """
    Construct the Hamiltonian for a lonely angular momentum state

    Parameters
    ----------
    F : int or float
        Angular momentum quantum number
    gF : float
        Associated Lande g-factor
    muB : float, optional
        Bohr magneton.  Default: the CODATA value in Hz/G
    return_basis : boolean, optional
        If true, return the basis.  Default: False

    Returns
    -------
    H_0 : array_like
        Field independent component of the Hamiltonian
    mu_q : array_like
        Magnetic field dependent component of the Hamiltonian
    basis : list
        List of :math:`(F, m_F)` basis states
    """
    index = lambda mF: int(F+mF)

    # Initialize the matrix
    H_0 = np.zeros((int(2*F+1), int(2*F+1)))
    mu_q = np.zeros((3, int(2*F+1), int(2*F+1)))

    # No diagonal elements
    # Off-diagonal elemnts:
    for ii, q in enumerate(np.arange(-1, 2, 1)):
        for mFp in np.arange(-F, F+1, 1):
            mF = mFp + q
            if not np.abs(mF) > F:
                # The minus sign here comes from the fact that the hyperfine
                # magnetic moment is dominated by the electron, whose magnetic
                # moment points in the opposite direction as the spin.
                mu_q[ii, index(mF), index(mFp)] -= gF*muB*\
                    (-1)**(F-mF)*np.sqrt(F*(F+1)*(2*F+1))*\
                    wig3j(F, 1, F, -mF, q, mFp)

    if return_basis:
        basis = np.zeros((int(2*F+1), 2))
        basis[:, 0] = F
        for mF in np.arange(-F, F+1, 1):
            basis[index(mF), 1] = mF

        argout = (H_0, mu_q, basis)
    else:
        argout = (H_0, mu_q)

    return argout


def dqij_norm(dqij):
    dqij_norm = np.zeros(dqij.shape)
    for ii in range(dqij.shape[0]):
        for jj in range(dqij.shape[2]):
            dqij_norm[ii, :, jj] = dqij[ii, :, jj]/\
                np.linalg.norm(dqij[:, :, jj])

    return dqij_norm


def dqij_two_hyperfine_manifolds(J, Jp, I, normalize=True, return_basis=False):
    """
    Dipole matrix element matrix elements for transitions between two
    hyperfine manifolds.

    Parameters
    ----------
    J : int or float
        Lower hyperfine manifold :math:`J` quantum number
    Jp : int or float
        Upper hyperfine manifold :math:`J\'` quantum number
    I : int or float
        Nuclear spin associated with both manifolds
    normalize : boolean, optional
        Normalize the d_q to one.  Default: True
    return_basis : boolean, optional
        If true, returns the basis states as well as the :math:`d_q`

    Returns
    -------
    d_q : array_like
        Dipole matrix elements between hyperfine manifolds
    basis_g : list
        If return_basis is true, list of (:math:`F`, :math:`m_F`)
    basis_e : list
        If return_basis is true, list of (:math:`F\'`, :math:`m_F\'`)
    """
    def matrix_element(J, F, m_F, Jp, Fp, m_Fp, I, q):
        return (-1)**(F-m_F+J+I+Fp+1)*np.sqrt((2*F+1)*(2*Fp+1))*\
            wig3j(F, 1, Fp, -m_F, q, m_Fp)*wig6j(J, F, I, Fp, Jp, 1)

    # A simple function for addressing the index:
    index = lambda Fmin, F, mF: coupled_index(F, mF, Fmin)

    # What's the minimum F1 and F2?
    Fmin = np.abs(I-J)
    Fpmin = np.abs(I-Jp)

    Fmax = np.abs(I+J)
    Fpmax = np.abs(I+Jp)

    dqij = np.zeros((3, int(np.sum(2*np.arange(Fmin, Fmax+0.5, 1)+1)),
                     int(np.sum(2*np.arange(Fpmin, Fpmax+0.5, 1)+1))))

    for ii, q in enumerate(range(-1, 2)):
        for F in np.arange(Fmin, Fmax+0.5, 1):
            for Fp in np.arange(Fpmin, Fpmax+0.5, 1):
                for m_Fp in np.arange(-Fp, Fp+0.5, 1):
                    m_F = m_Fp+q
                    if not np.abs(m_F) > F:
                        dqij[ii, index(Fmin, F, m_F), index(Fpmin, Fp, m_Fp)] =\
                        matrix_element(J, F, m_F, Jp, Fp, m_Fp, I, q)

    if normalize:
        dqij = dqij_norm(dqij)

    if return_basis:
        basis_g = np.zeros((dqij.shape[1], 2))
        basis_e = np.zeros((dqij.shape[2], 2))

        for F in np.arange(Fmin, Fmax+0.5, 1):
            for m_F in np.arange(-F, F+0.5, 1):
                basis_g[index(Fmin, F, m_F), :] = np.array([F, m_F])

        for Fp in np.arange(Fpmin, Fpmax+0.5, 1):
            for m_Fp in np.arange(-Fp, Fp+0.5, 1):
                basis_e[index(Fpmin, Fp, m_Fp), :] = np.array([Fp, m_Fp])

        argout = (dqij, basis_g, basis_e)
    else:
        argout = dqij

    return argout


def dqij_two_bare_hyperfine(F, Fp, normalize=True):
    """
    Calculates the dqij matrix for two bare hyperfine states.  Specifically,
    it returns the matrix of the operator $d_q$, where a photon is created by
    a transition from the excited state to the ground state.

    Parameters
    ----------
    F : integer or float (half integer)
        Total angular momentum quantum number of the F state.
    Fp : integer or float (half integer)
        Total angular momentum quantum number of the F\' state.
    normalize : boolean
        By default, 'normalize' is True
    """
    # A simple function for addressing the index:
    index = lambda F, mF: int(F+mF)

    # Initialize the matrix.  'Ground' state is represented by rows, 'excited'
    # state by columns.
    dqij = np.zeros((3, int(2*F+1), int(2*Fp+1)))

    # Populate the matrix.  First go through each q:
    for ii, q in enumerate(np.arange(-1, 2, 1)):
        # Go through each m_F2:
        for m_Fp in np.arange(-Fp, Fp+1, 1):
            # m_F1 takes on a q value:
            m_F = m_Fp+q
            if not np.abs(m_F) > F:
                dqij[ii, index(F, m_F), index(Fp, m_Fp)] =\
                    (-1)**(F-m_F)*wig3j(F, 1, Fp, -m_F, q, m_Fp)

    # Normalization involves normalzing each transition |g>->|e> to the norm of
    # all the transitions from the excited state sum(|e>->|g>).   That means
    # summing each column and each
    if normalize:
        dqij = dqij_norm(dqij)

    # Return the matrix:
    return dqij
