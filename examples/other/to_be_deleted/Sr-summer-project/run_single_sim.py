import numpy as np
import os
import pylcp
from pylcp.gratings import (infiniteGratingMOTBeams, maskedGaussianGratingMOTBeams)
import datetime, time
import dill

# Define event functions for evolve_motion:
def origin_return(t, y, length_scale, trap_center): # args are passed to events!
    v = y[-6:-3]
    r = (y[-3:]-trap_center)*length_scale
    return (np.sqrt(np.dot(r, r)+np.dot(v, v)) - 0.1)
# Make a function to check overlap of MOT beams:
def beam_overlap(t, y, beams):
    # beams is a laserBeams object:
    r = y[-3:]
    olap = np.prod(beams.beta(r, t), axis=0, keepdims=True)
    olap[olap == 0] = -1
    return olap

with open('parameters.pkl', 'rb') as input:
    trap = dill.load(input)
    z_0 = dill.load(input)
    (species, scale, angle_list, zchip_list, th_m, phi_m) = dill.load(input)

if species == "$^{88}$Sr":
    fstr = '88Sr'
elif species == "$^{87}$Sr":
    fstr = '87Sr'
elif species == "$^{174}$Yb":
    fstr = '174Yb'
elif species == "$^{171}$Yb":
    fstr = '171Yb'
for dd, thd_i in enumerate(angle_list):
    for zz, zchip_i in enumerate(zchip_list):
        path = 'sims/sim_%s_%s_%s.pkl' % (fstr, str(dd), str(zz))
        if os.path.isfile(path) or np.isnan(z_0[(thd_i,zchip_i)]):
            #skip files that already have a worker.
            continue
        else:
            # run the simulation:
            print(path)
            with open(path, 'wb') as output:
                sols = {}
                # get beam geometry
                laserBeams = trap[(thd_i,zchip_i)].laserBeams['g->e']
                # get the trap center:
                r_0 = np.array([0,0,z_0[(thd_i,zchip_i)]])
                print(r_0)
                # define trap escape event:
                escape = lambda t, y: beam_overlap(t, y, laserBeams)
                escape.terminal = True
                escape.direction = -1
                # define recapture event:
                capture = lambda t, y: origin_return(t, y, scale, r_0)
                capture.terminal = True
                capture.direction = -1
                for ii, th_i in enumerate(th_m):
                    ts = time.time()
                    for phi_i in phi_m:
                        print('starting theta = '+str(th_i)+', phi = '+str(phi_i))
                        # initial guess for escape velocity:
                        v_esc = 0
                        esc_sol = None
                        v_guess = 1.25
                        v_step = v_guess/2
                        levels = 5
                        for kk in np.arange(levels):
                            v_g = np.array([v_guess*np.sin(th_i)*np.cos(phi_i),
                                            v_guess*np.sin(th_i)*np.sin(phi_i),
                                            v_guess*np.cos(th_i)])
                            trap[(thd_i,zchip_i)].set_initial_position_and_velocity(r_0, v_g)
                            trap[(thd_i,zchip_i)].set_initial_pop_from_equilibrium()
                            trap[(thd_i,zchip_i)].evolve_motion([0, 2e6],
                                                                recoil_velocity=scale,
                                                                method='RK45',
                                                                t_eval=np.linspace(0,2e6,200),
                                                                vectorized=False,
                                                                events=(capture, escape),
                                                                rtol=1e-3,
                                                                atol=1e-6)
                            if trap[(thd_i,zchip_i)].sol.t_events[1].size == 0:
                                # atom did not escape, so go to higher velocity:
                                v_esc = v_guess
                                esc_sol = trap[(thd_i,zchip_i)].sol
                                print('recaptured ' + str(v_guess))
                                v_guess += v_step
                            else:
                                # atom left the beam overlap region, so go to lower velocity:
                                print('lost ' + str(v_guess))
                                v_guess -= v_step
                            v_step = v_step/2
                            # Don't integrate below the cutoff:
                            if v_guess <= 0.1:
                                break
                        # store result:
                        sols[(th_i,phi_i)] = esc_sol
                    # print in same way as for old simulation:
                    te = time.time()
                    print('Theta '+str(ii)+' took '+str((te-ts)/3600)+' hours')
                # save the results:
                dill.dump(sols, output)
            tc = time.time()
            print('Saving took '+str((tc-te)/3600)+' hours')
