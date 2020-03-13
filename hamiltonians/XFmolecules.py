#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: spe, ebn

Hamiltonians for alkaline-earth fluoride molecules.
"""
import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
import scipy.constants as cts

def wig3j(j1, j2, j3, m1, m2, m3):
    """
    This function redefines the wig3jj in terms of things that I like:
    """
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def wig6j(j1, j2, j3, l1, l2, l3):
    """
    This function redefines the wig6jj in terms of things that I like:
    """
    return float(wigner_6j(j1, j2, j3, l1, l2, l3))



def wig9j(j1, j2, j3, l1, l2, l3, n1, n2, n3):
    """
    This function redefines the wig9jj in terms of things that I like:
    """
    return float(wigner_9j(j1, j2, j3, l1, l2, l3, n1, n2, n3))


def ishermitian(A):
    """
    Simple method to check if a matrix is Hermitian.
    """
    return np.allclose(A, np.conjugate(A.T), rtol=1e-10, atol=1e-10)


def isunitary(A):
    """
    Simple method to check if a matrix is Hermitian.
    """
    return np.allclose(np.identity(A.shape[0]), A.T @ A, atol=1e-10)


def Xstate(Lambda=0, N=1, S=1/2, I1=1/2, B=6046.9, gamma=50.697, q=0, b=154.7,
           c=178.5, cc=0, q0=0, q2=0, gS=2.0023193043622, gI=0.00304206,
           muB=cts.value('Bohr magneton in Hz/T')*1e-10, return_basis=False):
    """
    Defines the field free and field dependent Hamiltonian of the ground state.
    Using the basis |Lambda, N, Sigma, J, F, m, P>.  We consider only one N=1 and one Lambda=1
    Default constants are given for $^{14}$MgF:
    B = 6046.9
    gamma = 50.697
    q = 0
    b = 154.7
    c = 178.5
    cc = 0
    q0 = 0
    q2 = 0
    """
    # Update N to an array
    Ns = np.array(N)
    Ns.shape = (Ns.size,)

    # Make the basis states:
    basis = np.empty((0, ),
                     dtype=[('Lambda', 'i4'), ('N', 'i4'), ('J', 'f4'),
                            ('F', 'f4'), ('mF', 'f4'), ('P', 'i4')])

    for N in np.array(Ns):
        for J in np.arange(np.abs(N-S), N+S+1, 1):
            for F in np.arange(np.abs(J-I1), J+I1+1, 1):
                for mF in np.arange(-F, F+1, 1):
                    basis = np.append(
                        basis, np.array([(Lambda, N, J, F, mF, (-1)**N)],
                                         dtype=basis.dtype))


    def spinrotation(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        """
        Spin rotation interaction.  l: Lambda.
        """
        return gamma*(-1)**(NN + J + S)*wig6j(S, NN, J, NN, S, 1)*\
            np.sqrt(S*(S + 1)*(2*S + 1)*NN*(NN + 1)*(2*NN + 1))*\
            (NN == NNp)*(J == Jp)*(F == Fp)*(m == mp)

    def hyperfine(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        """
        Hyperfine interaction.
        """
        return (b + c/3)*(-1)**(Jp + F + I1)*wig6j(I1, Jp, F, J, I1, 1)*\
            (-1)**(NN+J+S+1)*wig6j(S, Jp, NN, J, S, 1)*\
            np.sqrt((2*J+1)*(2*Jp+1)*I1*(I1+1)*(2*I1+1)*S*(S+1)*(2*S+1))*\
            (NN == NNp)*(F == Fp)*(m == mp)

    def dipoledipole(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        """
        Dipole-dipole itneraction.
        """
        return (1/3*c*np.sqrt(30))*(-1)**(Jp + F + I1 + 1 + NN)*\
            wig6j(I1, Jp, F, J, I1, 1)*\
            np.sqrt(I1*(I1 + 1)*(2*I1 + 1))*\
            np.sqrt((2*J+1)*(2*Jp+1))*\
            np.sqrt(S*(S + 1)*(2*S + 1))*\
            wig9j(J, Jp, 1, NN, NNp, 2, S, S, 1)*\
            np.sqrt((2*NN + 1)*(2*NNp + 1))*\
            wig3j(NN, 2, NNp, 0, 0, 0)*\
            (NN == NNp)*(F == Fp)*(m == mp)*(P == Pp)


    def nuclearspinrotation(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        """
        Nuclear spin rotation interaction, Brown and Carrington pg. 458
        """
        return cc*(-1)**(Jp + F + I1)*\
            wig6j(I1, Jp, F, J, I1, 1)*\
            (-1)**(Jp + NN + 1 + S)*np.sqrt((2*Jp + 1)*(2*J + 1))*\
            wig6j(J, NN, S, NN, Jp, 1)*\
            np.sqrt(NN*(NN + 1)*(2*NN + 1))*\
            np.sqrt(I1*(I1 + 1)*(2*I1 + 1))*\
            (NN == NNp)*(F == Fp)*(m == mp)*(P == Pp)

    def rotation(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        """
        Rotation of the molecule
        """
        return (B*(NN*(NN + 1) - l**2) + (-1)**(NN)*(P)*q/2*NN*(NN + 1))*\
            (NN == NNp)*(J == Jp)*(F == Fp)*(m == mp)*(P == Pp)


    def electricquadrupole(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        """
        Electric quadrupole interaction.
        """
        return (-1)**(Jp + I1 + F)*wig6j(I1, J, F, Jp, I1, 2)*\
            (-1)**(NNp + S + J)*wig6j(J, NN, S, NN, Jp, 2)*\
            (-1)**(NN - l)*np.sqrt((2*J+1)*(2*Jp+1)*(2*NN + 1)*(2*NNp + 1))/\
            wig3j(I1, 2, I1, -I1, 0, I1)*\
            (NN == NNp)*(J == Jp)*(F == Fp)*(P == Pp)*(m == mp)*\
            (q0/4*wig3j(NN, 2, NNp, -l, 2, lp) +
             q2/(4*np.sqrt(6))*(-1)**(J - S)*P*\
             wig3j(NN, 2, NNp, -l, 2, -lp))

    # Brown and Carrington 8.183
    def electronspinzeeman(l, NN, J, F, MF, P, lp, NNp, Jp, Fp, MFp, Pp, q):
        """
        Zeeman effect matrix element due to the electron spin.
        """
        return gS*muB*(-1)**(F - MF)*wig3j(F, 1, Fp, -MF, q, MFp)*\
            (-1)**(Fp + J + 1 + I1)*np.sqrt((2*Fp + 1)*(2*F + 1))*\
            wig6j(F, J, I1, Jp, Fp, 1)*\
            (-1)**(J + NN + 1 + S)*np.sqrt((2*Jp + 1)*(2*J + 1))*\
            wig6j(J, S, NN, S, Jp, 1)*\
            np.sqrt(S*(S + 1)*(2*S + 1))*\
            (NN == NNp)*(P == Pp)

    # Brown and Carrington 8.185
    def nuclearspinzeeman(l, NN, J, F, MF, P, lp, NNp, Jp, Fp, MFp, Pp, q):
        """
        Zeeman effect matrix element due to the nuclear spin.
        """
        return gI*muB*(-1)**(F - MF)*wig3j(F, 1, Fp, -MF, q, MFp)*\
            (-1)**(F + J + 1 + I1)*np.sqrt((2*Fp + 1)*(2*F + 1))*\
            wig6j(F, I1, J, I1, Fp, 1)*np.sqrt(I1*(I1 + 1)*(2*I1 + 1))*\
            (NN == NNp)*(P == Pp)

    H0 = np.zeros((basis.shape[0], basis.shape[0]))
    for ii, basis_i in enumerate(basis):
        for jj, basis_j in enumerate(basis):
            args = tuple(basis_i) + tuple(basis_j)
            H0[ii, jj] = spinrotation(*args) + hyperfine(*args) +\
                dipoledipole(*args) + nuclearspinrotation(*args) +\
                rotation(*args)

    Bq = np.zeros((3, basis.shape[0], basis.shape[0]))
    qs = [-1, 0, 1]
    for ll, q_i in enumerate(qs):
        for ii, basis_i in enumerate(basis):
            for jj, basis_j in enumerate(basis):
                args = tuple(basis_i) + tuple(basis_j) + (q_i,)
                Bq[ll, ii, jj] = electronspinzeeman(*args) + \
                    nuclearspinzeeman(*args)


    # Check to see if H0 is diagonal.  If not, diagonalize it:
    if np.count_nonzero(H0 - np.diag(np.diagonal(H0))) > 0:
        if not ishermitian(H0):
            raise ValueError("H0 is not hermitian.")

        # Diagonalize each m_F separately:
        E = np.zeros((basis.shape[0],))
        U = np.zeros((basis.shape[0], basis.shape[0]))
        for mF_i in np.unique(basis['mF']):
            inds = mF_i == basis['mF']
            if sum(inds) > 1:
                inds = np.where(inds)[0]
                E[inds], U[np.ix_(inds,inds)] =\
                    np.linalg.eigh(H0[np.ix_(inds,inds)])
            else:
                E[inds] = H0[inds, inds]
                U[inds, inds] = 1

        # Check the U matrix:
        if not isunitary(U):
            raise ValueError("Something went wrong with diagonalization.")

        ind = np.lexsort((basis['mF'], E))
        U = U[:, ind]

        H0 = U.T @ H0 @ U
        for ii in range(3):
            Bq[ii, :, :] = U.T @ Bq[ii, :, :] @ U
    else:
        U = np.identity(H0.shape[0])

    if return_basis:
        return H0, Bq, U, basis
    else:
        return H0, Bq, U


def Astate(Lambda=1, J=1/2, S=1/2, I1=1/2, P=+1, Ahfs=-1.5, gJ=0.003, gI=0.003,
           muB=cts.value('Bohr magneton in Hz/T')/1e6*1e-4, p=510., q=-51.,
           return_basis=False):
    """
    Defines the field free and static mangetic field versions of the A state.
    the A state is in Hund's case (b): |Lambda, Sigma, J, Omega, F, mF, P>.  Taking guesses as to the parameters for gyromagnetic rations.
    """
    basis = np.empty((0, ),
                     dtype=[('Lambda', 'i4'), ('S','f4'), ('J', 'f4'),
                            ('I1', 'f4'), ('F', 'f4'), ('mF', 'f4'),
                            ('P', 'i4')])

    Ps = np.array(P)
    Ps.shape = (Ps.size,)

    for P in Ps:
        for F in np.arange(np.abs(J-I1), np.abs(J+I1)+1, 1):
            for mF in np.arange(-F, F+1, 1):
                basis = np.append(basis, np.array(
                    [(Lambda, S, J, I1, F, mF, P)],
                     dtype=basis.dtype))

    def lambda_doubling(L, S, J, I, F, mF, P, Lp, Sp, Jp, Ip, Fp, mFp, Pp):
        return -P*(-1)**(J-1/2)*(p+2*q)*(J+1/2)/2*(L==Lp)*(J==Jp)*(I==Ip)*\
            (F==Fp)*(mF==mFp)*(P==Pp)*(S==Sp)

    def hyperfine(L, S, J, I, F, mF, P, Lp, Sp, Jp, Ip, Fp, mFp, Pp):
        """
        Hyperfine interaction: L: Lambda, Sig:Sigma, O:Omega, others obvious
        """
        return Ahfs*F*(F+1)*(L==Lp)*(J==Jp)*(F==Fp)*(mF==mFp)*(P==Pp)

    def zeeman(L, S, J, I, F, mF, P, Lp, Sp, Jp, Ip, Fp, mFp, Pp, q):
        """
        Zeeman interaction: L: Lambda, Sig:Sigma, O:Omega, others obvious
        """
        return gJ*muB*(-1)**(F-mF)*wig3j(F, 1, Fp, mF, q, -mFp)*\
            np.sqrt((2*Fp+1)*(2*F+1))*(-1)**(J+I1+Fp+1)*\
            wig6j(J, F, I1, Fp, J, 1)*\
            np.sqrt(J*(J+1)*(2*J+1)) +\
            gI*muB*(-1)**(F-mF)*wig3j(F, 1, Fp, mF, q, -mFp)*\
            np.sqrt((2*Fp+1)*(2*F+1))*(-1)**(J+I1+Fp+1)*\
            wig6j(I1, F, J, Fp, I1, 1)*\
            np.sqrt(I1*(I1+1)*(2*I1+1))*(L==Lp)*(P==Pp)*(J==Jp)

    H_0 = np.zeros((basis.shape[0], basis.shape[0]))
    for ii, basis_i in enumerate(basis):
        for jj, basis_j in enumerate(basis):
            args = tuple(basis_i) + tuple(basis_j)
            H_0[ii, jj] = hyperfine(*args) + lambda_doubling(*args)

    B_q = np.zeros((3, basis.shape[0], basis.shape[0]))
    for ll, q in enumerate(np.arange(-1, 2, 1)):
        for ii, basis_i in enumerate(basis):
            for jj, basis_j in enumerate(basis):
                args = tuple(basis_i) + tuple(basis_j) + (q,)
                B_q[ll, ii, jj] = zeeman(*args)

    if return_basis:
        return H_0, B_q, basis
    else:
        return H_0, B_q


def dipoleXandAstates(xbasis, abasis, I1=1/2, S=1/2, UX=[],
                      return_intermediate=False):
    """
    Calculate the oscillator strengths between the X and A states.  X state
    basis is assumed to be Hund's case (b) while the A state is assumed to be
    Hund's case (a).  Thus, we need to make an itermediate basis to transform
    between the two.
    """
    dijq = np.zeros((3, xbasis.shape[0], abasis.shape[0]))

    def dipole_matrix_element(L, Sig, O, J, F, mF,
                              Lp, Sigp, Op, Jp, Fp, mFp, q):
        """
        The dipole matrix element, less the reduced matrix element between the X
        and A states.  Shorthand: L=Lambda, O=Omega, P=parity.
        """
        return (-1)**(F-mF)*wig3j(F, 1, Fp, -mF, q, mFp)*(-1)**(Fp+J+I1+1)*\
            np.sqrt((2*F+1)*(2*Fp+1))*wig6j(Jp, Fp, I1, F, J, 1)*\
            (-1)**(J-O)*np.sqrt((2*J+1)*(2*Jp+1))*\
            (wig3j(J, 1, Jp, -O, -1, Op) + wig3j(J, 1, Jp, -O, +1, Op))

    def elements_transform_a_to_b(L, Sig, O, J, F, mF,
                                  Lp, Np, Jp, Fp, mFp, Pp):
        """
        Matrix elements to transform for Hund's case (a) to (b) (Norrgard thesis, pg.)
        """
        return (-1)**(J+Sig+L)*np.sqrt(2*Np+1)*wig3j(S, Np, J, Sig, L, -O)*\
            (L == Lp)*(J == Jp)*(F == Fp)*(mF == mFp)

    def elements_transform_a_to_p(L, S, J, I, F, mF, P,
                                  Lp, Sigp, Op, Jp, Fp, mFp):
        """
        Matrix elements to transform for Hund's case (a) to to parity Eq. 6.234
        """
        if Lp > 0:
            el = 1/np.sqrt(2)*(J == Jp)*(F == Fp)*(mF == mFp)
        else:
            el = 1/np.sqrt(2)*P*(-1)**(J-S)*(J == Jp)*(F == Fp)*(mF == mFp)
        return el

    """ Now, we need to sum over all states to transform between Hund's case (a)
    and (b).  """
    intbasis_ba = np.empty((0, ),
                           dtype=[('Lambda', 'i4'), ('Sigma', 'f4'),
                                  ('Omega', 'f4'), ('J', 'f4'), ('F', 'f4'),
                                  ('mF', 'f4')])

    LambdaX = xbasis['Lambda'][0]
    Ns = np.unique(xbasis['N'])
    for N in Ns:
        for Omega in np.arange(-S, S+1, 1):
            for J in np.arange(np.abs(N-S), np.abs(N+S)+1, 1):
                for F in np.arange(np.abs(J-I1), np.abs(J+I1)+1, 1):
                    for mF in np.arange(-F, F+1, 1):
                        intbasis_ba = np.append(
                            intbasis_ba, np.array([(LambdaX, Omega, Omega,
                                                    J, F, mF)],
                                                  dtype=intbasis_ba.dtype))

    # Make the transformation array to transform from Hund's case (b) to
    # Hund's case (a):
    T_ba = np.zeros((xbasis.shape[0], intbasis_ba.shape[0]))
    for ii, xbasis_i in enumerate(xbasis):
        for jj, intbasis_ba_i in enumerate(intbasis_ba):
            T_ba[ii, jj] = elements_transform_a_to_b(
                *(tuple(intbasis_ba_i) + tuple(xbasis_i))
                )

    """ Now, we need to sum over all states to transform between Hund's case (a)
    and parity states.  """
    intbasis_ap = np.empty((0, ),
                           dtype=[('Lambda', 'i4'), ('Sigma', 'f4'),
                                  ('Omega', 'f4'), ('J', 'f4'), ('F', 'f4'),
                                  ('mF', 'f4')])

    # This implicitly assumes |Omega|=J in the A state
    JA = np.unique(abasis['J'])
    LambdaA = np.unique(abasis['Lambda'])[0] # It's one lambda
    for J in JA:
        Sigma = J-LambdaA
        for F in np.arange(np.abs(J-I1), np.abs(J+I1)+1, 1):
            for mF in np.arange(-F, F+1, 1):
                intbasis_ap = np.append(intbasis_ap, np.array(
                    [(LambdaA, Sigma, Omega, J, F, mF),
                     (-LambdaA, -Sigma, -Omega, J, F, mF)],
                    dtype=intbasis_ap.dtype))

    # Now make the transfer matrix:
    T_pa = np.zeros((abasis.shape[0], intbasis_ap.shape[0]))
    for (ii, abasis_i) in enumerate(abasis):
        for (jj, intbasis_ap_i) in enumerate(intbasis_ap):
            T_pa[ii, jj] = elements_transform_a_to_p(
                *(tuple(abasis_i) + tuple(intbasis_ap_i)))

    T_ap = T_pa.T

    # Make the dipole matrix element operator in this intermediate basis:
    intdijq = np.zeros((3, intbasis_ba.shape[0], intbasis_ap.shape[0]))
    for ii, q in enumerate(np.arange(-1, 2, 1)):
        for jj, intbasis_ba_i in enumerate(intbasis_ba):
            for kk, intbasis_ap_i in enumerate(intbasis_ap):
                intdijq[ii, jj, kk] = dipole_matrix_element(
                    *(tuple(intbasis_ba_i) + tuple(intbasis_ap_i) + (q,))
                    )

    # Finally, did the user pass to us a rotation matrix for case (b) into the
    # eignebasis:
    if UX == []:
        UX = np.identity(xbasis.shape[0])

    # Now transform in Hund's case A basis
    dijq = np.zeros((3, xbasis.shape[0], abasis.shape[0]))
    for ii in range(3):
        dijq[ii] = UX.T @ T_ba @ intdijq[ii] @ T_ap

    if return_intermediate:
        return dijq, T_ap, T_ba, intdijq, intbasis_ap, intbasis_ba
    else:
        return dijq


# %% Run some tests if we are in the main namespace:
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('paper')

    # %%
    """
    Let's focus on the ground manifold first:
    """
    # SrF numbers
    """H0_X, Bq_X, U_X, Xbasis = Xstate(
        N=1, Lambda=0, return_basis=True, B=7487.60, b=97.0834, c=30.268,
        cc=2.876e-2, gamma=74.79485
        )"""

    # CaF numbers: Journal of Molecular Spectroscopy, 86 (2), 365 (1981)
    H0_X, Bq_X, U_X, Xbasis = Xstate(
        N=1, Lambda=0, return_basis=True, B=0, b=109.1893, c=40.1190,
        cc=2.876e-2, gamma=39.65891
        )

    B = np.linspace(0, 20, 101)
    Es_X = np.zeros((B.size, H0_X.shape[0]))
    for ii, B_i in enumerate(B):
        Es_X[ii, :], Us = np.linalg.eig(H0_X+Bq_X[1]*B_i)
        Es_X[ii, :] = np.sort(Es_X[ii, :])

    fig, ax = plt.subplots(1, 1, num="Ground state Zeeman effect")
    ax.plot(B, Es_X, '-', color='C0')

    # Let's see if I get the same thing putitng it in the x-direction:
    for ii, B_i in enumerate(B):
        Es_X[ii, :], Us = np.linalg.eig(H0_X - Bq_X[0]/np.sqrt(2)*B_i +
                                        Bq_X[2]/np.sqrt(2)*B_i)
        Es_X[ii, :] = np.sort(Es_X[ii, :])

    ax.plot(B, Es_X, '--', linewidth=0.75, color='C1')
    ax.set_xlabel('$B$ (G)')
    ax.set_ylabel('$E$ (MHz)')

    # %%
    """
    What does the excited state look like?
    """
    H0_A, Bq_A, Abasis = Astate(P=+1, return_basis=True, Ahfs=2.4, q=0, p=0,
                                gJ=2*-0.021)

    B = np.linspace(0, 20, 101)
    Es_A = np.zeros((B.size, H0_A.shape[0]))
    for ii, B_i in enumerate(B):
        Es_A[ii, :], Us = np.linalg.eig(H0_A+Bq_A[1]*B_i)
        Es_A[ii, :] = np.sort(Es_A[ii,:])

    fig, ax = plt.subplots(1, 1, num="Excited state Zeeman shift")
    ax.plot(B, Es_A, '-', color='C0')

    # Let's see if I get the same thing putitng it in the x-direction:
    for ii, B_i in enumerate(B):
        Es_A[ii, :], Us = np.linalg.eig(H0_A - Bq_A[0]/np.sqrt(2)*B_i +
                                        Bq_A[2]/np.sqrt(2)*B_i)
        Es_A[ii, :] = np.sort(Es_A[ii,:])

    ax.plot(B, Es_A, '--', linewidth=0.75, color='C1')
    ax.set_xlabel('$B$ (G)')
    ax.set_ylabel('$E$ (MHz)')

    # %%
    """
    Let's check the projections:
    """
    # [print(Xbasis[ii], U_X[ii,:]) for ii in range(Xbasis.shape[0])]

    # %%
    """
    Now let's focus on the dipole matrix elements between X and A:
    """
    np.set_printoptions(precision=4, suppress=True)
    dijq, T_pa, T_ba, dijqint, intbasis_ap, intbasis_ba = dipoleXandAstates(
        Xbasis, Abasis, I1=1/2, S=1/2, UX=U_X, return_intermediate=True)

    #np.savetxt('/Users/spe/Desktop/Tba_N1.csv', T_ba,
    #           delimiter=',')
    #[print(Xbasis[ii]) for ii in range(Xbasis.shape[0])]
    #print(dijq[0, :, :]**2)
    #print(np.sum(dijq**2, axis=0))
    #print(np.sum(dijq**2, axis=(0,1)))

    # Try to reproduce Barry and Norrgard table:
    """for ii in range(Xbasis.shape[0])[::-1]:
        big_comp = np.argmax(U_X[:, ii]**2)
        print(np.sum(dijq[:, ii, :]**2,axis=0)[[0, 3, 2, 1]],
              '(', Xbasis['N'][big_comp],
              ',', Xbasis['J'][big_comp],
              ',', Xbasis['F'][big_comp],
              ',', Xbasis['mF'][big_comp], ')')"""
    # print(dijq[0,:,:]**2)

    # Try to reproduce Fig. 3 rates; Tarbutt, PRA 92, 053401 (2015)
    qind = 2
    print('F_g = 1: F_e = 0: {0:.3f} F_e = 1: {1:.3f}'.format(
          np.sum(dijq[qind, 0:3, 0]**2), np.sum(dijq[qind, 0:3, 1::]**2)))
    print('F_g = 0: F_e = 0: {0:.3f} F_e = 1: {1:.3f}'.format(
          np.sum(dijq[qind, 3, 0]**2), np.sum(dijq[qind, 3, 1::]**2)))
    print('F_g = 1: F_e = 0: {0:.3f} F_e = 1: {1:.3f}'.format(
          np.sum(dijq[qind, 4:7, 0]**2), np.sum(dijq[qind, 4:7, 1::]**2)))
    print('F_g = 2: F_e = 0: {0:.3f} F_e = 1: {1:.3f}'.format(
          np.sum(dijq[qind, 7::, 0]**2), np.sum(dijq[qind, 7::, 1::]**2)))
