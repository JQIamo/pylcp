import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
import scipy.constants as cts

def __wig3j(j1, j2, j3, m1, m2, m3):
    """
    This function redefines the wig3jj in terms of things that I like:
    """
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def __wig6j(j1, j2, j3, l1, l2, l3):
    """
    This function redefines the wig6jj in terms of things that I like:
    """
    return float(wigner_6j(j1, j2, j3, l1, l2, l3))



def __wig9j(j1, j2, j3, l1, l2, l3, n1, n2, n3):
    """
    This function redefines the wig9jj in terms of things that I like:
    """
    return float(wigner_9j(j1, j2, j3, l1, l2, l3, n1, n2, n3))


def __ishermitian(A):
    """
    Simple method to check if a matrix is Hermitian.
    """
    return np.allclose(A, np.conjugate(A.T), rtol=1e-10, atol=1e-10)


def __isunitary(A):
    """
    Simple method to check if a matrix is Hermitian.
    """
    return np.allclose(np.identity(A.shape[0]), A.T @ A, atol=1e-10)


def Xstate(N, I, B=0., gamma=0., b=0., c=0., CI=0., q0=0, q2=0,
           gS=cts.value('electron g factor'), gI=cts.value('proton g factor'),
           muB=cts.value('Bohr magneton in Hz/T')*1e-4*1e-6,
           muN=cts.m_e/cts.m_p*cts.value('Bohr magneton in Hz/T')*1e-4*1e-6,
           return_basis=False):

    """
    Defines the field-free and magnetic field-dependent components of the
    :math:`X^2\\Sigma^+` ground state Hamiltonian.

    Parameters
    ----------
        N : int
            Rotational quantum number.
        I : int or float
            Nuclear spin quantum number of the fluorine, usually 1/2
        B : float
            Rotational constant. Default: 0.
        gamma : float
            Electron-spin rotational coupling constant. Default: 0.
        b : float
            Isotropic spin-spin interaction. Default: 0.
        c : float
            Anisotropic spin-spin interaction. Default: 0.
        CI : float
            Nuclear-spin rotational coupling constant. Default: 0.
        q0 : float
            Electron quadrupole constant. Default: 0.
        q2 : float
            Electron quadrupole constant. Default: 0.
        gS : float
            Electron spin g-factor. Default: CODATA value.
        gI : float
            Nuclear (proton) g-factor. Default: CODATA value.
        muB : float, optional
            Bohr Magneton. Default value is the CODATA value in MHz/G.
        muN : float
            Nuclear Magneton. Default value is the CODATA value in MHz/G
        return_basis : boolean, optional
            Boolean to specify whether to return the basis as well as the
            Hamiltonian matrices.  Default: True.

    Notes
    -----
    Assuming Hund's case (b) with basis
    :math:`\\left|\\Lambda, N, \\Sigma, J, F, m_F, P \\right>`, :math:`\\Lambda=0` and
    :math:`\\Sigma=1/2`.  The full Hamiltonian is a combination of a Brown
    and Carrington, *Rotational Spectroscopy of Diatomic Molecules*,
    Eqs. 9.88 (rotation), 9.89 (spin-rotation), 9.90 (hyperfine),
    9.91 (dipole-dipole interaction), 9.53 (electric quadrupole), 8.183
    (electronic spin Zeeman) and 8.185 (nuclear spin Zeeman).  See the comments
    in the code for more details on equations used and approximations made.
    Most Hamiltonian parameters are both keyword arguments and by default zero
    so that the user can easily turn on only the relavent terms easily.
    """
    # -----
    # Converting between hyperfine parameter conventions (thanks to Prof T. Steimle)
    # -----
    # APar=Aiso+2Adip
    # Apar=b+c
    # Aper=Aiso-Adip
    # Aper=b
    # Adip=c/3
    # Aiso= bF = b+c/3

    # These quantum numbers are the same:
    Lambda = 0
    S = 1/2

    # Update N to an array, in case we want multiple N:
    Ns = np.array(N)
    Ns.shape = (Ns.size,)

    # Make the basis states:
    basis = np.empty((0, ),
                     dtype=[('Lambda', 'i4'), ('N', 'i4'), ('J', 'f4'),
                            ('F', 'f4'), ('mF', 'f4'), ('P', 'i4')])

    for N in np.array(Ns):
        for J in np.arange(np.abs(N-S), N+S+1, 1):
            for F in np.arange(np.abs(J-I), J+I+1, 1):
                for mF in np.arange(-F, F+1, 1):
                    basis = np.append(
                        basis, np.array([(Lambda, N, J, F, mF, (-1)**N)],
                                         dtype=basis.dtype))



    # To Do:  rename to B_q to mu_p;  mu for magnetic, p refers to lab frame
    # coordinates for molecules. q refers to molecule frame coordinates and
    # the current labeling is confusing


    # Brown and Carrington 9.89
    def spinrotation(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        return gamma*(-1)**(NN + J + S)*__wig6j(S, NN, J, NN, S, 1)*\
            np.sqrt(S*(S + 1)*(2*S + 1)*NN*(NN + 1)*(2*NN + 1))*\
            (l==lp)*(NN == NNp)*(J == Jp)*(F == Fp)*(m == mp)*(P==Pp)

    # Brown and Carrington 9.90
    def hyperfine(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        return (b + c/3)*(-1)**(Jp + F + I)*__wig6j(I, Jp, F, J, I, 1)*\
            (-1)**(NN+J+S+1)*__wig6j(S, Jp, NN, J, S, 1)*\
            np.sqrt((2*J+1)*(2*Jp+1)*I*(I+1)*(2*I+1)*S*(S+1)*(2*S+1))*\
            (l==lp)*(NN == NNp)*(F == Fp)*(m == mp)*(P==Pp)


    # Brown and Carrington 9.91
    def dipoledipole(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        return (1/3*c*np.sqrt(30))*(-1)**(Jp + F + I + 1 + NN)*\
            __wig6j(I, Jp, F, J, I, 1)*\
            np.sqrt(I*(I + 1)*(2*I + 1))*\
            np.sqrt((2*J+1)*(2*Jp+1))*\
            np.sqrt(S*(S + 1)*(2*S + 1))*\
            __wig9j(J, Jp, 1, NN, NNp, 2, S, S, 1)*\
            np.sqrt((2*NN + 1)*(2*NNp + 1))*\
            __wig3j(NN, 2, NNp, 0, 0, 0)*\
            (NN == NNp)*(F == Fp)*(m == mp)*(P == Pp)

    # Brown and Carrington pg. 458
    def nuclearspinrotation(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        return CI*(-1)**(Jp + F + I)*\
            __wig6j(I, Jp, F, J, I, 1)*\
            (-1)**(Jp + NN + 1 + S)*np.sqrt((2*Jp + 1)*(2*J + 1))*\
            __wig6j(J, NN, S, NN, Jp, 1)*\
            np.sqrt(NN*(NN + 1)*(2*NN + 1))*\
            np.sqrt(I*(I + 1)*(2*I + 1))*\
            (NN == NNp)*(F == Fp)*(m == mp)*(P == Pp)

    #Brown and Carrington 9.88; rotattion
    def rotation(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        return B*NN*(NN + 1)*\
            (NN == NNp)*(J == Jp)*(F == Fp)*(m == mp)*(P == Pp)

    #Brown and Carrington 9.53, adapted to Hund's case b
    #see also Brown and Carrington 9.94 (Electronic quadrupole interaction)
    def electricquadrupole(l, NN, J, F, m, P, lp, NNp, Jp, Fp, mp, Pp):
        return (-1)**(Jp + I + F)*__wig6j(I, J, F, Jp, I, 2)*\
            (-1)**(NNp + S + J)*__wig6j(J, NN, S, NN, Jp, 2)*\
            (-1)**(NN - l)*np.sqrt((2*J+1)*(2*Jp+1)*(2*NN + 1)*(2*NNp + 1))/\
            __wig3j(I, 2, I, -I, 0, I)*\
            (NN == NNp)*(J == Jp)*(F == Fp)*(P == Pp)*(m == mp)*\
            (q0/4*__wig3j(NN, 2, NNp, -l, 2, lp) +
             q2/(4*np.sqrt(6))*(-1)**(J - S)*P*\
             __wig3j(NN, 2, NNp, -l, 2, -lp))

    # Brown and Carrington 8.183
    def electronspinzeeman(l, NN, J, F, MF, P, lp, NNp, Jp, Fp, MFp, Pp, p):
        return gS*muB*(-1)**(F - MF)*__wig3j(F, 1, Fp, -MF, p, MFp)*\
            (-1)**(Fp + J + 1 + I)*np.sqrt((2*Fp + 1)*(2*F + 1))*\
            __wig6j(F, J, I, Jp, Fp, 1)*\
            (-1)**(J + NN + 1 + S)*np.sqrt((2*Jp + 1)*(2*J + 1))*\
            __wig6j(J, S, NN, S, Jp, 1)*\
            np.sqrt(S*(S + 1)*(2*S + 1))*\
            (l==lp)*(NN == NNp)*(P == Pp)

    # Brown and Carrington 8.185
    def nuclearspinzeeman(l, NN, J, F, MF, P, lp, NNp, Jp, Fp, MFp, Pp,p):
        return -gI*muN*(-1)**(F - MF)*__wig3j(F, 1, Fp, -MF, p, MFp)*\
            (-1)**(F + J + 1 + I)*np.sqrt((2*Fp + 1)*(2*F + 1))*\
            __wig6j(F, I, J, I, Fp, 1)*np.sqrt(I*(I + 1)*(2*I + 1))*\
            (NN == NNp)*(P == Pp)*(J==Jp)


    H0 = np.zeros((basis.shape[0], basis.shape[0]))
    for ii, basis_i in enumerate(basis):
        for jj, basis_j in enumerate(basis):
            args = tuple(basis_i) + tuple(basis_j)
            H0[ii, jj] = spinrotation(*args) + hyperfine(*args) +\
                dipoledipole(*args) + nuclearspinrotation(*args)
            if I >=1:
                 H0[ii, jj] += electricquadrupole(*args)
            if Ns.size >= 2:
                H0[ii,jj] +=  rotation(*args)

    mu_p = np.zeros((3, basis.shape[0], basis.shape[0]))
    qs = [-1, 0, 1]
    for ll, q_i in enumerate(qs):
        for ii, basis_i in enumerate(basis):
            for jj, basis_j in enumerate(basis):
                args = tuple(basis_i) + tuple(basis_j) + (q_i,)
                mu_p[ll, ii, jj] = electronspinzeeman(*args) + \
                    nuclearspinzeeman(*args)


    # Check to see if H0 is diagonal.  If not, diagonalize it:
    if np.count_nonzero(H0 - np.diag(np.diagonal(H0))) > 0:
        if not __ishermitian(H0):
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
        if not __isunitary(U):
            raise ValueError("Something went wrong with diagonalization.")

        ind = np.lexsort((basis['mF'], E))
        U = U[:, ind]

        H0 = U.T @ H0 @ U
        for ii in range(3):
            mu_p[ii, :, :] = U.T @ mu_p[ii, :, :] @ U
    else:
        U = np.identity(H0.shape[0])

    if return_basis:
        return H0, mu_p, U, basis
    else:
        return H0, mu_p, U


def Astate(J, I, P, B=0., D=0., H=0., a=0., b=0., c=0., eQq0=0., p=0., q=0.,
           gS=2.002, gL=1, gl=0, glprime=0, gr=0, greprime=0, gN=0,
           muB=cts.value('Bohr magneton in Hz/T')*1e-4*1e-6,
           muN=cts.m_e/cts.m_p*cts.value('Bohr magneton in Hz/T')*1e-4*1e-6,
           return_basis=False):
    """
    Defines the field-free and magnetic field-dependent components of the excited
    :math:`A^2\Pi_{1/2}` state Hamiltonian.

    Parameters
    ----------
        J : int
            Rotational quantum number(s)
        I : int or float
            Nuclear spin quantum number
        P : int or float
            Parity quantum number (:math:`\\pm1`)
        B : float
            Rotational constant. Default: 0.
        D : float
            First non-rigid rotor rotational constant. Default: 0.
        H : float
            Second non-rigid rotor rotational constant. Default: 0.
        a : float
            Frosch and Foley :math:`a` parameter. Default: 0.
        b : int or float
            Frosch and Foley :math:`b` parameter. Default: 0.
        c : int or float
            Frosch and Foley :math:`c` parameter. Default: 0.
        eQq0: int or float
            electric quadrupole hyperfine constant (only valid for I>=1).
            Default: 0.
        p : float
            Lambda-doubling constant. Default: 0.
        q : float
            Lambda-doubling constant. Default: 0.
        muB : float
            Bohr Magneton.  Default value is the CODATA value in MHz/G.
        muN : float
            Nuclear Magneton.  Default value is the CODATA value in MHz/G.
        gS : float
            Electron spin g-factor. Default: :math:`g_S = 2.002`.
        gL : float
            Orbital g-factor. Note that it may deviate slightly from 1 due to
            relativistic, diamagnetic, and non-adiabatic contributions.
            Default: :math:`g_L = 1`.
        gr : float
            Rotational g-factor.  Default: 0.
        gl : float
            Anisotropic electron spin g-factor. Default: 0.
        glprime : float
            Parity-dependent anisotropic electron spin g-factor.
            A reasonable approximation is that :math:`g_l' \\sim p/2B`.
            Default: 0.
        greprime : float
            Parity-dependent electron contribution to rotational g-factor.
            A resonable approximation is that :math:`g_{re}' \\sim -q/B`.
            Default: 0.
        gN : float
            Nuclear spin g-factor.  It changes negligibly in most molecules.
            Default: 0.
        return_basis : boolean, optional
            Boolean to specify whether to return the basis as well as the
            Hamiltonian matrices.  Default: True.

    Notes
    -----
    Assumes the A state is in Hund's case (a), namely
    :math:`\\left|\\Lambda, J, \\Omega, I, F, m_F, P \\right>`.  By definition,
    :math:`\\Sigma = \\Omega - \\Lambda`.  For the A state,
    :math:`\\Sigma=1/2`, :math:`\\Lambda=1`, and then :math:`\\Omega=1/2`.
    The full Hamiltonian is a combination of a Brown
    and Carrington, *Rotational Spectroscopy of Diatomic Molecules*,
    Eqs. 6.196 (rotation), 8.401 (:math:`\Lambda`-doubling),
    8.372 (nuclear spin-orbit coupling), 8.374 (Fermi contact interaction),
    8.506 (quadrupole), 9.57, 9.58, 9.59, 9.60, 9.70, and 9.71 (Zeeman
    interaction).  See the comments in the code for more details on equations
    used and approximations made.  Most Hamiltonian parameters are
    both keyword arguments and by default zero so that the user can easily
    turn on only the relavent terms easily.
    """
    basis = np.empty((0, ),
                     dtype=[('Lambda', 'i4'), ('S','f4'), ('J', 'f4'),('O','f4'),
                            ('I', 'f4'), ('F', 'f4'), ('mF', 'f4'),('P', 'i4')])

    Ps = np.array(P)
    Ps.shape = (Ps.size,)

    Js = np.array(J)
    Js.shape = (Js.size,)
    S=1/2
    Lambda=1
    O=1/2

    for P in Ps:
        for J in Js:
            for F in np.arange(np.abs(J-I), np.abs(J+I)+1, 1):
                for mF in np.arange(-F, F+1, 1):
                    basis = np.append(basis, np.array(
                        [(Lambda, S, J, O, I, F, mF, P)],
                        dtype=basis.dtype))

    # Brown and Carrington, 6.196:
    def rotation(L, S, J, O, I, F, mF, P, Lp, Sp, Jp, Op, Ip, Fp, mFp, Pp):
        return (B*J*(J+1) - D*J**2*(J+1)**2 + H*J**3*(J+1)**3)*(L==Lp)*(S==Sp)*(J==Jp)*(O==Op)*(I==Ip)*(F==Fp)*(mF==mFp)*(P==Pp)  #6.196

    # Only correct for $^2\Pi_{1/2}$ states, see table on page 531 of Brown
    # and Carrington. See 8.401 for full Lambda doubling Hamiltonian.
    def lambda_doubling(L, S, J, O, I, F, mF, P, Lp, Sp, Jp, Op, Ip, Fp, mFp, Pp):
        return -P*(-1)**(J-1/2)*(p+2*q)*(J+1/2)/2*(L==Lp)*(S==Sp)*(J==Jp)*(O==Op)*(I==Ip)*\
            (F==Fp)*(mF==mFp)*(P==Pp)

    # Brown and Carrington 8.372
    def nuclearspinorbit(L, S, J, O, I, F, mF, P, Lp, Sp, Jp, Op, Ip, Fp, mFp, Pp):
        return a*L*(-1)**(Jp+F+I)*__wig6j(I,Jp,F,J,I,1)*np.sqrt(I*(I+1)*(2*I+1))*(-1)**(J-O)*\
        __wig3j(J, 1, Jp, -O, 0, Op)*np.sqrt((2*J+1)*(2*Jp+1))*\
        (L==Lp)*(S==Sp)*(I==Ip)*(F==Fp)*(P==Pp)*(mF==mFp)

    # Brwon and Carrington 8.374, Using Omega = Lambda + Sigma, or Sigma = Omega-Labmda
    def fermicontact(L, S, J, O, I, F, mF, P, Lp, Sp, Jp, Op, Ip, Fp, mFp, Pp):
        return (b+c/3)*(-1)**(Jp+F+I+J-O+S-(O-L))*__wig6j(I, Jp, F, J, I, 1)*np.sqrt(I*(I+1)*(2*I+1)*(2*J+1)*(2*Jp+1)*S*(S+1)*(2*S+1))*\
        (S==Sp)*(I==Ip)*(F==Fp)*(mF==mFp)*(P==Pp)*(-1)**(J+S-2*O+L)*\
        0.5*(
        __wig3j(J, 1, Jp, -O, -1, Op)*__wig3j(S, 1, Sp, -(O-L),-1, (Op-Lp))  +  P*(-1)**(J-S)*__wig3j(J, 1, Jp, +O, -1, Op)*__wig3j(S, 1, Sp, +(O-L),-1, (Op-Lp))  + Pp*(-1)**(Jp-Sp)*__wig3j(J, 1, Jp, -O, -1, -Op)*__wig3j(S, 1, Sp, -(O-L),-1, -(Op-Lp))  + P*Pp*(-1)**(J-S+Jp-Sp)*__wig3j(J, 1, Jp, +O, -1, -Op)*__wig3j(S, 1, Sp, +(O-L),-1, -(Op-Lp))+\
        __wig3j(J, 1, Jp, -O,  0, Op)*__wig3j(S, 1, Sp, -(O-L), 0, (Op-Lp))  +  P*(-1)**(J-S)*__wig3j(J, 1, Jp, +O,  0, Op)*__wig3j(S, 1, Sp, +(O-L), 0, (Op-Lp))  + Pp*(-1)**(Jp-Sp)*__wig3j(J, 1, Jp, -O,  0, -Op)*__wig3j(S, 1, Sp, -(O-L), 0, -(Op-Lp))  + P*Pp*(-1)**(J-S+Jp-Sp)*__wig3j(J, 1, Jp, +O,  0, -Op)*__wig3j(S, 1, Sp, +(O-L), 0, -(Op-Lp))+\
        __wig3j(J, 1, Jp, -O,  1, Op)*__wig3j(S, 1, Sp, -(O-L), 1, (Op-Lp))  +  P*(-1)**(J-S)*__wig3j(J, 1, Jp, +O,  1, Op)*__wig3j(S, 1, Sp, +(O-L), 1, (Op-Lp))  + Pp*(-1)**(Jp-Sp)*__wig3j(J, 1, Jp, -O,  1, -Op)*__wig3j(S, 1, Sp, -(O-L), 1, -(Op-Lp))  + P*Pp*(-1)**(J-S+Jp-Sp)*__wig3j(J, 1, Jp, +O,  1, -Op)*__wig3j(S, 1, Sp, +(O-L), 1, -(Op-Lp))
        )

    # Brown and Carrington 8.506.  This ignores eQq2, which couples states with \Delta\Omega = \pm 2, see Brown and Carrington 8.382
    def quadrupole(L, S, J, O, I, F, mF, P, Lp, Sp, Jp, Op, Ip, Fp, mFp, Pp):
        return -(-1)**(Jp+I+F)*__wig6j(I, J, F, Jp, I, 2)/__wig3j(I, 2, I, -I, 0, I)*(-1)**(J-O)*np.sqrt((2*J+1)*(2*Jp+1))*\
        (L==Lp)*(S==Sp)*(O==Op)*(I==Ip)*(F==Fp)*(mF==mFp)*(P==Pp)*\
        eQq0/4*__wig3j(J, 2, Jp, -O, 0, Op)

    def zeeman(L, S, J, O, I, F, mF, P, Lp, Sp, Jp, Op, Ip, Fp, mFp, Pp):
        reduced_matrix_elements = 0

        # 9.57, orbital Zeeman + rotational Zeeman term:
        reduced_matrix_elements += muB*(gL+gr)*L*(-1)**(Fp+J+I+1+J-O)*__wig6j(J,F,I, Fp,Jp,1)* __wig3j(J,1,Jp, -O,0,Op)* np.sqrt((2*F+1)*(2*Fp+1)*(2*J+1)*(2*Jp+1))*(L==Lp)*(O==Op)*(S==Sp)*(I==Ip)*(P==Pp)

        # 9.58, electron spin Zeeman + rotational Zeeman term:
        reduced_matrix_elements += muB*(gS+gr+gl)*(-1)**(Fp+J+I+1+J-O+S-(O-L))*__wig6j(J,F,I, Fp,Jp,1)*np.sqrt((2*F+1)*(2*Fp+1)*(2*J+1)*(2*Jp+1)*S*(S+1)*(2*S+1))*(L==Lp)*(S==Sp)*(I==Ip)*(P==Pp)*\
        0.5*(
        __wig3j(J, 1, Jp, -O, -1, Op)*__wig3j(S, 1, Sp, -(O-L),-1, (Op-Lp))  +  P*(-1)**(J-S)*__wig3j(J, 1, Jp, +O, -1, Op)*__wig3j(S, 1, Sp, +(O-L),-1, (Op-Lp))  + Pp*(-1)**(Jp-Sp)*__wig3j(J, 1, Jp, -O, -1, -Op)*__wig3j(S, 1, Sp, -(O-L),-1, -(Op-Lp))  + P*Pp*(-1)**(J-S+Jp-Sp)*__wig3j(J, 1, Jp, +O, -1, -Op)*__wig3j(S, 1, Sp, +(O-L),-1, -(Op-Lp))+\
        __wig3j(J, 1, Jp, -O,  0, Op)*__wig3j(S, 1, Sp, -(O-L), 0, (Op-Lp))  +  P*(-1)**(J-S)*__wig3j(J, 1, Jp, +O,  0, Op)*__wig3j(S, 1, Sp, +(O-L), 0, (Op-Lp))  + Pp*(-1)**(Jp-Sp)*__wig3j(J, 1, Jp, -O,  0, -Op)*__wig3j(S, 1, Sp, -(O-L), 0, -(Op-Lp))  + P*Pp*(-1)**(J-S+Jp-Sp)*__wig3j(J, 1, Jp, +O,  0, -Op)*__wig3j(S, 1, Sp, +(O-L), 0, -(Op-Lp))+\
        __wig3j(J, 1, Jp, -O,  1, Op)*__wig3j(S, 1, Sp, -(O-L), 1, (Op-Lp))  +  P*(-1)**(J-S)*__wig3j(J, 1, Jp, +O,  1, Op)*__wig3j(S, 1, Sp, +(O-L), 1, (Op-Lp))  + Pp*(-1)**(Jp-Sp)*__wig3j(J, 1, Jp, -O,  1, -Op)*__wig3j(S, 1, Sp, -(O-L), 1, -(Op-Lp))  + P*Pp*(-1)**(J-S+Jp-Sp)*__wig3j(J, 1, Jp, +O,  1, -Op)*__wig3j(S, 1, Sp, +(O-L), 1, -(Op-Lp))
        )

        #9.59, Nuclear spin zeeman:
        reduced_matrix_elements += -muN*gN*(-1)**(Fp+J+I+1)*__wig6j(I,F,J, Fp,I,1)*np.sqrt(I*(I+1)*(2*I+1)*(2*F+1)*(2*Fp+1))*(L==Lp)*(S==Sp)*(J==Jp)*(O==Op)*(I==Ip)*(P==Pp)

        #9.60, third and final rotational Zeeman term:
        reduced_matrix_elements += -muB*gr*(-1)**(Fp+J+I+1)*__wig6j(J,F,I, Fp,Jp,1)*np.sqrt(J*(J+1)*(2*J+1)*(2*F+1)*(2*Fp+1))*(L==Lp)*(S==Sp)*(J==Jp)*(O==Op)*(I==Ip)*(P==Pp)

        #cf 9.58 and 9.71, anisotropic electron spin Zeeman:
        reduced_matrix_elements += -muB*(gl)*(O-L)*(-1)**(Fp+J+I+1+J-O)*__wig6j(J,F,I, Fp,Jp,1)* __wig3j(J,1,Jp, -O,0,Op)* np.sqrt((2*F+1)*(2*Fp+1)*(2*J+1)*(2*Jp+1))*(L==Lp)*(S==Sp)*(O==Op)*(I==Ip)*(P==Pp)

        #cf 9.70,9.71 parity-dependent Zeeman terms:
        reduced_matrix_elements += -muB*(glprime-greprime)*(-1)**(Fp+J+I+1+J-O)*__wig6j(J,F,I, Fp,Jp,1)* __wig3j(J,1,Jp, -O,0,Op)* np.sqrt((2*F+1)*(2*Fp+1)*(2*J+1)*(2*Jp+1)*S*(S+1)*(2*S+1))*\
        0.5*(
        __wig3j(J, 1, Jp, -O, +1, Op)*__wig3j(S, 1, Sp, -(O-L),-1, (Op-Lp))  +  P*(-1)**(J-S)*__wig3j(J, 1, Jp, +O, +1, Op)*__wig3j(S, 1, Sp, +(O-L),-1, (Op-Lp))  + Pp*(-1)**(Jp-Sp)*__wig3j(J, 1, Jp, -O, +1, -Op)*__wig3j(S, 1, Sp, -(O-L),-1, -(Op-Lp))  + P*Pp*(-1)**(J-S+Jp-Sp)*__wig3j(J, 1, Jp, +O, +1, -Op)*__wig3j(S, 1, Sp, +(O-L),-1, -(Op-Lp))+\
        __wig3j(J, 1, Jp, -O, -1, Op)*__wig3j(S, 1, Sp, -(O-L),+1, (Op-Lp))  +  P*(-1)**(J-S)*__wig3j(J, 1, Jp, +O, -1, Op)*__wig3j(S, 1, Sp, +(O-L),+1, (Op-Lp))  + Pp*(-1)**(Jp-Sp)*__wig3j(J, 1, Jp, -O, -1, -Op)*__wig3j(S, 1, Sp, -(O-L),+1, -(Op-Lp))  + P*Pp*(-1)**(J-S+Jp-Sp)*__wig3j(J, 1, Jp, +O, -1, -Op)*__wig3j(S, 1, Sp, +(O-L),+1, -(Op-Lp))
        )

        # In essence, apply Wigner-Eckart Theorem:
        mu_q = np.zeros((3,))
        for kk, p in enumerate([-1, 0, 1]):
            mu_q[kk] = (-1)**p *(-1)**(F-mF)* __wig3j(F,1,Fp, -mF, p, mFp)*reduced_matrix_elements  #check if we need the (-1)**p factor here!!!
        return mu_q

    H_0 = np.zeros((basis.shape[0], basis.shape[0]))
    for ii, basis_i in enumerate(basis):
        for jj, basis_j in enumerate(basis):
            args = tuple(basis_i) + tuple(basis_j)
            H_0[ii, jj] = nuclearspinorbit(*args) +fermicontact(*args)
            if Ps.size !=1:
                H_0[ii,jj]+= lambda_doubling(*args)
            if I >= 1:
                H_0[ii,jj] +=quadrupole(*args)
            if Js.size >= 2:  #if only considering a single J, ignore energy offset due to rotation
                H_0[ii, jj] += rotation(*args)

    mu_p = np.zeros((3, basis.shape[0], basis.shape[0]))
    for ii, basis_i in enumerate(basis):
        for jj, basis_j in enumerate(basis):
            args = tuple(basis_i) + tuple(basis_j)
            mu_p[:, ii, jj] = zeeman(*args)

    if return_basis:
        return H_0, mu_p, basis
    else:
        return H_0, mu_p


def dipoleXandAstates(xbasis, abasis, I=1/2, S=1/2, UX=[],
                      return_intermediate=False):
    """
    Calculate the oscillator strengths between the X and A states.

    Parameters
    ----------
        xbasis : list or array_like
            List of basis vectors for the X state
        abasis : list or array_like
            List of basis vectors for the A state
        I : int or float
            Nuclear spin angular momentum.  Default: 1/2.
        S : int or float
            :math:`\\Sigma` quantum number.  Default: 1/2.
        UX : two-dimensional array, optional
            a rotation matrix for case (b) into the intermediate eigenbasis.
            Default: empty
        return_intermediate : boolean, optional
            Argument to return the intermediate bases and transformation
            matrices.

    Notes
    ----
    The X state is assumed to be Hund's case (b) while the A state is assumed
    to be Hund's case (a).  Thus, this function makes an intermediate basis to
    transform between the two.
    """
    dijq = np.zeros((3, xbasis.shape[0], abasis.shape[0]))

    def dipole_matrix_element(L, Sig, O, J, F, mF,
                              Lp, Sigp, Op, Jp, Fp, mFp, q):
        """
        The dipole matrix element, less the reduced matrix element between the X
        and A states.  Shorthand: L=Lambda, O=Omega, P=parity.
        """
        return (-1)**(F-mF)*__wig3j(F, 1, Fp, -mF, q, mFp)*(-1)**(Fp+J+I+1)*\
            np.sqrt((2*F+1)*(2*Fp+1))*__wig6j(Jp, Fp, I, F, J, 1)*\
            (-1)**(J-O)*np.sqrt((2*J+1)*(2*Jp+1))*\
            (__wig3j(J, 1, Jp, -O, -1, Op) + __wig3j(J, 1, Jp, -O, +1, Op))

    def elements_transform_a_to_b(L, Sig, O, J, F, mF,
                                  Lp, Np, Jp, Fp, mFp, Pp):
        """
        Matrix elements to transform for Hund's case (a) to (b) (Norrgard thesis, pg.)
        """
        return (-1)**(J+Sig+L)*np.sqrt(2*Np+1)*__wig3j(S, Np, J, Sig, L, -O)*\
            (L == Lp)*(J == Jp)*(F == Fp)*(mF == mFp)

    def elements_transform_a_to_p(L, S, J, O, I, F, mF, P,
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
                for F in np.arange(np.abs(J-I), np.abs(J+I)+1, 1):
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
        for F in np.arange(np.abs(J-I), np.abs(J+I)+1, 1):
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
    #plt.style.use('paper')
    plt.close('all')
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
        N=1, Lambda=0, S=1/2, I=1/2, return_basis=True, B=10303.98670, b=109.1893, c=40.1190,
        CI=2.876e-2, gamma=39.65891        )

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

    ax.plot(B, Es_X, linewidth=0.75, color='C1')
    ax.set_xlabel('$B$ (G)')
    ax.set_ylabel('$E$ (MHz)')

    # %%
    """
    What does the excited state look like?
    """
    H0_A, Bq_A, Abasis = Astate(J=1/2,I=1/2,P=+1, a=3/2*4.8,  p=-1313.091, B= 10456.19, glprime=-3*.0211, return_basis=True)
    #for CaF, Lambda-doubling parameter  p=-1313.091
    print(H0_A, Bq_A)

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
        Xbasis, Abasis, I=1/2, S=1/2, UX=U_X, return_intermediate=True)

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
