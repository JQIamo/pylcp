"""
Created on Fri Oct 27 09:51:32 2017

@author: spe

This example shows how to calculate relevant numbers for real atomic species.
"""
import pylcp.atom as atom

# Let's work out some numbers:
atom_Li = atom.atom("7Li")
atom_Rb = atom.atom("87Rb")

print("Li: v0 = {0:.3f} m/s, a_max = {1:.3e} m/s^2, v0^2/a_max = {2:.3f} mm"\
        .format(atom_Li.transition[1].v0, atom_Li.transition[1].a0,
                1e3*atom_Li.transition[1].x0))
print("B_\gamma' = {0:.3f} G/cm ".format(atom_Li.transition[1].Bgamma))
print("Rb: v0 = {0:.3f} m/s, a_max = {1:.3e} m/s^2, v0^2/a_max = {2:.3f} mm"\
        .format(atom_Rb.transition[1].v0, atom_Rb.transition[1].a0,
                1e3*atom_Rb.transition[1].x0))
print("B_\gamma' = {0:.3f} G/cm ".format(atom_Rb.transition[1].Bgamma))
