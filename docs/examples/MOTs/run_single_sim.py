import numpy as np
import pylcp
from pylcp.gratings import (infiniteGratingMOTBeams, maskedGaussianGratingMOTBeams)
import datetime, time
import dill

num_of_sims = 12

with open('parameters.pkl', 'rb') as input:
    obe = dill.load(input)
    args = dill.load(input)
    kwargs = dill.load(input)
    (rscale, roffset, vscale, voffset) = dill.load(input)

laserBeams = obe.laserBeams

for jj in range(num_of_sims):
    tic = time.time()
    obe.set_initial_position(rscale*np.random.randn(3) + roffset)
    obe.set_initial_velocity(vscale*np.random.randn(3) + voffset)
    obe.set_initial_rho_from_rateeq()
    obe.evolve_motion(*args, **kwargs)
    
    now = datetime.datetime.now()
    with open('sims/sim_%s.pkl' % now.strftime('%Y%m%d_%H%M%S_%f'), 'wb') as output:
        dill.dump(obe.sol, output)

    toc = time.time() 
    print('completed solution %d in %.2f s.' % (jj, toc-tic))
