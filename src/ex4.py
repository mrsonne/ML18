"""
Visualize the data using suitable visualization approaches taking into account the
ACCENT principles, i.e. apprehension, clarity, consistency, effciency, necessity and
truthfulness as well as Tufte's guidelines. Argue for the choices you make when
visualizing the data. When visualizing the data consider the following:

* Are there issues with outliers in the data,
* do the attributes appear to be normal distributed,
* are variables correlated,
* does the primary machine learning modeling aim appear to be feasible based on your visualizations.

Provide a discussion explaining what you have learned about the data up until now.
Summarize here the most important things you have learned about the data and
give also your thoughts on whether your primary modeling task(s) appears to be
feasible based on your visualization. This will complete the first part of the course
report covering "Data: Feature extraction and visualization"
"""


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import zscore
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

MAXATTRLENGTH = 7
FS_TICKLABEL = 14
FS_AXLABEL = 16
FS_LEGEND = 12


def test():
    pass


def boxplot(X, y, classnames, attributenames, show=False, save=True):
    """
    Modified from Exercise 4.1.4
    """
    fs_ticks = 18
    fs_title = 18

    nclasses = len(classnames)
    fig, axs = plt.subplots(1, nclasses, figsize=(14,7), sharey=True)
    for idxclass, classname in enumerate(classnames):
        class_mask = (y == idxclass) # binary mask to extract elements of class c        
        axs[idxclass].boxplot(X[class_mask,:])
        axs[idxclass].set_title('Class: {}'.format(classname), fontsize=fs_title)
        axs[idxclass].set_xticks(range(1, len(attributenames)+1))
        axs[idxclass].set_xticklabels([a[:MAXATTRLENGTH] for a in attributenames], rotation=45, fontsize=fs_ticks)
        y_up = X.max() + (X.max() - X.min())*0.1 
        y_down = X.min() - (X.max() - X.min())*0.1
        axs[idxclass].set_ylim(y_down, y_up)

        for tick in axs[idxclass].yaxis.get_major_ticks():
            tick.label.set_fontsize(fs_ticks) 

    if save:
        fname = "../images/breast_cancer/boxplot.png"
        plt.savefig(fname, bbox_inches='tight')

    if show:
        plt.show()

    return fig, axs

def hist(X, y, classnames, attributenames):
    """
    Modified from Exercise 4.1.2
    """
    nobservations = X.shape[0]
    nattrs = len(attributenames)
    u = int(np.floor(np.sqrt(nattrs)))
    v = int(np.ceil(float(nattrs)/u))
    fig, axs = plt.subplots(u, v, figsize=(8,7))
    axs = axs.flat
    for idxattr, ax in enumerate(axs):
        bin_edges = 'auto'

        # we need to fix the range since we are plotting multiple classes on top of each other
        print(attributenames[idxattr], X[:, idxattr].min(), X[:, idxattr].max())
        rmin, rmax = X[:, idxattr].min(), X[:, idxattr].max()
        for idxclass, classname in enumerate(classnames):
            class_mask = (y == idxclass) # binary mask to extract elements of class c
            histdata, bin_edges = np.histogram(X[class_mask, idxattr], bins=bin_edges, range=(rmin, rmax))
            width = 0.7 * (bin_edges[1] - bin_edges[0])
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(centers, histdata, alpha=0.5, 
                   label=classname, align='center', width=width)
            ax.set_xlabel(attributenames[idxattr], fontsize=15)
            ax.set_ylim(0, float(nobservations)/4)

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15) 

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(15) 



    # axs[1].legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 2.25),
    #               fancybox=True, shadow=True)

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, 'upper center', fontsize=14, ncol=len(classnames))


    fig.tight_layout(pad=4., h_pad=1., w_pad=1.)
    fname = "../images/breast_cancer/histogram.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def scatterplot_matrix(X, y, classnames, attributenames, labels=None, odir='./', fileid=''):
    """
    Modified from Exercise 4.1.5
    """
    N = len(y)
    M = len(attributenames)
    C = len(classnames)
    fig, axs = plt.subplots(M, M, figsize=(18,14))
    iplt = 0
    axs = axs.flat
    for m1 in range(M):
        for m2 in range(M):
            for c, classname in enumerate(classnames):
                class_mask = (y==c)
                axs[iplt].plot(np.array(X[class_mask,m2]),
                               np.array(X[class_mask,m1]),
                               '.', label=classname)
                if m1 == M - 1:
                    axs[iplt].set_xlabel(attributenames[m2][:MAXATTRLENGTH], fontsize=20)
                else:
                    axs[iplt].set_xticks([])

                if m2 == 0:
                    axs[iplt].set_ylabel(attributenames[m1][:MAXATTRLENGTH], fontsize=20)
                else:
                    axs[iplt].set_yticks([])

                for tick in axs[iplt].xaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 

                for tick in axs[iplt].yaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 



                    # annonate with labels

            if labels is not None:
                xs = X[:, m2].take(labels)
                ys = X[:, m1].take(labels)
                # axs[iplt].annotate('{}'.format(idobs),
                #                    xy=coords, xycoords='data',
                #                    xytext=(7.5, 7.5), textcoords='offset points',
                #                    # arrowprops=dict(shrink=0.005),
                #                    horizontalalignment='center', verticalalignment='center')
                axs[iplt].plot(xs, ys, 'x', label='Misclassified', color='black')


            iplt += 1
    
    lines, labels = axs[0].get_legend_handles_labels()

    if labels is not None:
        ncol = C + 1
    else:
        ncol = C 

    fig.legend(lines, labels, 'upper center', fontsize=18, ncol=ncol)


    fname = os.path.join(odir, "scatter_matrix{}.png".format(fileid))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def matrix(X, y, classnames, attributenames):
    """
    Modified from Exercise 4.1.7
    """
    nattrs = len(attributenames)
    X_standarized = zscore(X, ddof=1)

    fig, axs = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    fig.suptitle('Data matrix of Z-scores')

    for idxcls, clsname in enumerate(classnames):
        class_mask = (y==idxcls)
        n = class_mask.astype('int').sum()
        im = axs[idxcls].imshow(X_standarized[class_mask], interpolation='none', aspect=(4./n), cmap=cm.gray)
        axs[idxcls].set_title(clsname)
        axs[idxcls].set_ylabel('Data objects')

    # Axis for the colorbar
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])  
    cb = plt.colorbar(im, cax=cbaxes)

    axs[-1].set_xticks(range(nattrs))
    axs[-1].set_xticklabels([a[:MAXATTRLENGTH] for a in attributenames], rotation=45)
    axs[-1].set_xlabel('Attributes')
    fname = "../images/breast_cancer/zscore_matrix.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
