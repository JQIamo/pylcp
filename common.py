import time
import numpy as np

# Define a progress bar for use in the next section of code:
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1,
                     length = 100, fill = '█',remaining_time = 0):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if remaining_time > 0:
        time_str = "{0:d}:{1:02d}:{2:02d}".format(
                np.floor(remaining_time/3600.0).astype(int),
                np.floor((remaining_time/60.0)%60.0).astype(int),
                np.floor((remaining_time%60.0)).astype(int))
        print('\r%s |%s| %s%% %s; est. time remaining: %s' %
              (prefix, bar, percent, suffix, time_str), end = '\r')
    else:
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

class progressBar(object):
    def __init__(self, decimals=1, fill='█', prefix='Progress:',
                 suffix='complete', length=40):
        self.tic = time.time()
        self.decimals = decimals
        self.fill = fill
        self.length = length
        self.prefix = prefix
        self.suffix = suffix

    def update(self, percentage):
        toc = time.time()
        percent = ("{0:." + str(self.decimals) + "f}").format(100*percentage)
        filledLength = int(self.length*percentage)
        bar = self.fill*filledLength + '-'*(self.length - filledLength)
        remaining_time = np.round((1-percentage)*((toc-self.tic)/percentage))
        if remaining_time>0 and percentage>0:
            time_str = "%2d:%02d:%02d" % (min(remaining_time/3600.0, 99),
                                          (remaining_time/60.0)%60.0,
                                          remaining_time%60.0)
            print('\r%s |%s| %s%% %s; est. time remaining: %s' %
                  (self.prefix, bar, percent, self.suffix, time_str), end='\r')
        else:
            print('\r%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix), end='\r')

        # Print New Line on Complete
        if percentage >= 1:
            print()

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
