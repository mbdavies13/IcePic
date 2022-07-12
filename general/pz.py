"""
Code for doing pz analysis on systems.
Purpose: to define contact layers for xy images.
"""

def calc_pz_hist(z, norm=False, frames=None, boxarea=None, bin_width=0.01, trough_prom=5):
    from scipy.signal import find_peaks
    import numpy as np

    bins1 = np.arange(np.amin(z)-0.1,np.amax(z)+0.1,bin_width)
    hist_1, bin_edges1 = np.histogram(z,bins1)
    if norm == False:
        hist1 = hist_1
    elif norm == True:
        hist1 = hist_1 / (frames * boxarea * bin_width * 0.1 * 0.1) # 0.1*0.1 convert A to nm
    bincenters1 = 0.5*(bin_edges1[1:]+bin_edges1[:-1])

    # peak analysis
    minushist1 = -1*hist1
    peaks1, properties1= find_peaks(hist1, height=0,distance=1, prominence=5)

    # trough analysis
    troughs1, tro_properties1 = find_peaks(minushist1, height=(-1000,0),distance=1, prominence=trough_prom)

    return bincenters1, hist1, peaks1, troughs1

def plot_pz_hist(z, bincenters, hist, peaks, troughs,
                 lw=3,
                 show_bool=True):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rc('font', size=15)
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    ax = fig.add_subplot(111)

    ax.grid(True)

    ##plot histograms##
    if np.amin(z) == np.amax(z):
        ax.plot(bincenters, hist, 'b', label="substrate - 2D", lw=lw)
    else:
        ax.plot(bincenters, hist, 'b', label="substrate", lw=lw)

    ##plot peaks and troughs analysis##
    ax.scatter(bincenters[peaks], hist[peaks], color="darkorange", marker='x', lw=lw, s=100, zorder=10)
    ax.scatter(bincenters[troughs], hist[troughs], facecolor="None", edgecolor="darkorchid", lw=lw,  s=100, zorder=10)

    # axis
    ax.set_xlabel('z (nm)')
    ax.set_ylabel('p(z) $nm^{-3}$')

    if show_bool == True:
        plt.show()
    return fig, ax



