#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:43:11 2017

@author: spe
"""
import scipy.constants as cts
import numpy as np
from numpy import pi


class state():
    """
    The quantum state and its parameters for an atom.

    Parameters
    ----------
        n : integer
            Principal quantum number of the state.
        S : integer or float
            Total spin angular momentum of the state.
        L : integer or float
            Total orbital angular momentum of the state.
        J : integer or float
            Total electronic angular momentum of the state.
        lam : float, optional
            Wavelength, in meters, of the photon necessary to excite the state
            from the ground state.  electronic angular momentum of the state.
        E : float, optional
            Energy of the state above the ground state in cm:math:`^{-1}`.
        tau : float, optioanl
            Lifetime of the state in s.  If not specified, it is assumed to
            be infinite (the ground state).
        gJ : float
            Total angular momentum Lande g-factor.
        Ahfs : float
            A hyperfine coefficient.
        Bhfs : float
            B hyperfine coefficient.
        Chfs : float
            C hyperfine coefficient.

    Attributes
    ----------
    All the parameters above are saved as attirbutes.

    Additional attributes:
        gamma : float
            Lifetime in s:math:`^{-1}`
        gammaHz : float
            Corresponding linewidth in Hz, given by :math:`\gamma/2\pi`.
        energy : float
            The energy in cm:math:`^{-1}`

    Notes
    -----
    One can define the energy of the state either through `lam` or through `E`.
    One of these two optional variables must be specified.

    This construction of the state assumes L-S coupling.
    """
    def __init__(self, n=None, S=None, L=None, J=None, lam=None, E=None,
                 tau=np.inf, gJ=1, Ahfs=0, Bhfs=0, Chfs=0):
        self.n = n

        self.L = L
        self.S = S
        self.J = J

        self.gJ = gJ

        if lam:
            self.energy = 0.01/lam  # cm^-1
        elif E:
            self.energy = E
        else:
            raise ValueError("Need to specify energy of the state somehow.")

        self.tau = tau  # s
        self.gamma = 1/self.tau  # s^{-1}
        self.gammaHz = self.gamma/2/pi  # Hz

        self.Ahfs = Ahfs
        self.Bhfs = Bhfs
        self.Chfs = Chfs


class transition():
    """
    A transition is between two states and has interesting properties for laser
    cooling.
    """
    def __init__(self, state1, state2, mass):
        self.k = state2.energy - state1.energy  # cm^{-1}
        self.lam = 0.01/self.k # m
        self.nu = cts.c/self.lam  # Hz
        self.omega = 2*np.pi*self.nu

        # Typical definition from Fermi's Golden rule:
        self.Isat = cts.hbar*self.omega**3*state2.gamma/(12*np.pi*cts.c**2) # W/m^2
        self.Isat *= 1000/1e4

        # Maximum acceleration on this transition:
        self.a0 = cts.hbar*(2*np.pi*100*self.k)*state2.gamma/2/mass # cm/s^2 ->
        self.v0 = state2.gamma/(2*np.pi*100*self.k) # cm/s
        self.x0 = self.v0**2/self.a0
        self.t0 = self.v0/self.a0

        # Save B gamma:
        self.Bgamma = state2.gammaHz/cts.value('Bohr magneton in Hz/T')/1e-4


class atom():
    """
    A class containing reference data for laser cooling alkali atoms

    Parameters
    ----------
        species : string
            The isotope number and species of alkali atom.  For lithium-7,
            species can be eiter "7Li" or "Li7", for example.  Supported species
            are "6Li", "7Li", "23Na", "85Rb", and "87Rb"

    Attributes
    ----------
        I : float
            Nuclear spin of the isoptope
        gI : float
            Nuclear g-factor of the isotope.  Note that the nuclear g-factor
            is specified relative to the Bohr magneton, not the nuclear
            magneton.
        mass : float
            Mass, in kg, of the atom.
        states : list of pylcp.atom.state
            States of the atom useful for laser cooling, in order of increasing
            energy.
        transitions : list of pylcp.atom.transition
            Transitions in the atom useful for laser cooling.  All transitions
            are from the ground state.
    """
    def __init__(self, species="7Li"):
        # Prepare to add in some useful electronic states:
        self.state = []

        if species == "6Li" or species == "Li6":
            self.I = 1   # nuclear spin
            self.gI = -0.0004476540  # nuclear magnetic moment
            self.mass = 6.0151214*cts.value('atomic mass constant')  # kg

            # TODO: FIX all of these numbers so they are actually lithium 6:
            # Ground state:
            self.state.append(state(n=2, L=0, S=1/2, J=1/2, lam=np.inf,
                                    tau=np.inf, gJ=-2.0023010,
                                    Ahfs=152.1368407e6))
            # D1 line (2P_{1/2})
            self.state.append(state(n=2, L=1, S=1/2, J=1/2,
                                    lam=670.976658173e-9, tau=27.109e-9,
                                    gJ=-0.6668, Ahfs=17.375e6))
            # D2 line (2P_{3/2})
            self.state.append(state(n=2, L=1, S=1/2, J=3/2,
                                    lam=670.961560887e-9, tau=27.102e-9,
                                    gJ=-1.335, Ahfs=-1.155e6, Bhfs=-0.40e6))

        elif species == "7Li" or species == "Li7":
            self.I = 3/2  # nuclear spin
            self.gI = -0.0011822130  # nuclear magnetic moment
            self.mass = 7.0160045*cts.value('atomic mass constant')  # kg

            # Ground state:
            self.state.append(state(n=2, L=0, J=1/2, lam=np.inf, tau=np.inf,
                                    gJ=2.0023010, Ahfs=401.7520433e6, S=1/2))
            # D1 line (2P_{1/2})
            self.state.append(state(n=2, L=1, J=1/2, lam=670.976658173e-9,
                                    tau=27.109e-9, gJ=0.6668, Ahfs=45.914e6,
                                    S=1/2))
            # D2 line (2P_{3/2})
            self.state.append(state(n=2, L=1, J=3/2, lam=670.961560887e-9,
                                    tau=27.102e-9, gJ=1.335, Ahfs=-3.055e6,
                                    Bhfs=-0.221e6, S=1/2))
            # 3P_{1/2}
            self.state.append(state(n=3, L=1, J=1/2, lam=323.3590e-9,
                                    tau=998.4e-9, gJ=2/3, Ahfs=13.5e6, S=1/2))
            # 3P_{1/2}
            self.state.append(state(n=3, L=1, J=3/2, lam=323.3590e-9,
                                    tau=998.4e-9, gJ=4/3, Ahfs=-0.965e6,
                                    Bhfs=-0.019e6, S=1/2))

        elif species == "23Na" or species == "Na23":
            self.I = 3/2  # nuclear spin
            self.gI = -0.00080461080  # nuclear magnetic moment
            self.mass = 22.9897692807*cts.value('atomic mass constant')  # kg

            # Ground state:
            self.state.append(state(n=2, L=0, J=1/2, lam=np.inf, tau=np.inf,
                                    gJ=2.00229600, Ahfs=885.81306440e6, S=1/2))
            # D1 line (2P_{1/2})
            self.state.append(state(n=2, L=1, J=1/2, lam=589.7558147e-9,
                                    tau=16.299e-9, gJ=0.66581, Ahfs=94.44e6,
                                    S=1/2))
            # D2 line (2P_{3/2})
            self.state.append(state(n=2, L=1, J=3/2, lam=589.1583264e-9,
                                    tau=16.2492e-9, gJ=1.33420, Ahfs=18.534e6,
                                    Bhfs=2.724e6, S=1/2))

        elif species == "85Rb" or species == "Rb85":
            self.I = 5/2  # nuclear spin
            self.gI = -0.00029364000  # nuclear magnetic moment
            self.mass = 84.911789732*cts.value('atomic mass constant')

            # Ground state:
            self.state.append(state(n=5, L=0, J=1/2, lam=np.inf, tau=np.inf,
                                    gJ=2.0023010, Ahfs=1.0119108130e9, S=1/2))
            # D1 line (2P_{1/2})
            self.state.append(state(n=5, L=1, J=1/2, lam=780.241e-9,
                                    tau=27.679e-9, gJ=0.6668, Ahfs=120.527e6,
                                    S=1/2))
            # D2 line (2P_{1/2})
            self.state.append(state(n=5, L=1, J=3/2, lam=780.241e-9,
                                    tau=26.2348e-9, gJ=1.335, Ahfs=25.0020e6,
                                    Bhfs=25.79e6, S=1/2))

        elif species == "87Rb" or species == "Rb87":
            self.I = 3/2  # nuclear spin
            self.gI = -0.0009951414  # nuclear magnetic moment
            self.mass = 86.909180527*cts.value('atomic mass constant')

            # Ground state:
            self.state.append(state(n=5, L=0, J=1/2, lam=np.inf, tau=np.inf,
                                    gJ=2.00233113, Ahfs=3.417341305452145e9,
                                    S=1/2))
            # D1 line (5P_{1/2})
            self.state.append(state(n=5, L=1, J=1/2, lam=794.978851156e-9,
                                    tau=27.679e-9, gJ=0.666, Ahfs=407.24e6,
                                    S=1/2))
            # D2 line (5P_{3/2})
            self.state.append(state(n=5, L=1, J=3/2, lam=780.241209686e-9,
                                    tau=26.2348e-9, gJ=1.3362, Ahfs=84.7185e6,
                                    Bhfs=12.4965e6, S=1/2))
        else:
            raise ValueError("Atom {0:s} not recognized.".format(species))

        # Take the states and make transitions:
        self.__make_transitions()


    def __sort_states(self):
        """
        Sorts the states by energy.
        """
        # TODO: implement
        pass


    def __make_transitions(self):
        """
        Take subtractions of energies to generate transitions from ground
        state
        """
        self.transition = []
        for ii, state_i in enumerate(self.state):
            if ii > 0:
                self.transition.append(transition(self.state[0], state_i,
                                                  self.mass))
