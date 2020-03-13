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

def random_vector():
    a, b = np.random.rand(2)
    th = np.arccos(2*b-1)
    phi = 2*np.pi*a

    return np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi),
                     np.cos(th)])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vectors = []
    for n in range(500):
        vectors.append(random_vector())

    vectors = np.array(vectors)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vectors[:,0], vectors[:,1], vectors[:,2])
    ax.view_init(elev=-90., azim=0.)
