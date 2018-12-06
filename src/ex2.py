"""
## Excercise 2
For your report please complete the following two sections:
Provide a description of your Dataset:

* What the problem of interest is (i.e. what is your data about),
* Where did you obtained the data,
* What the primary machine learning modeling aim is for the data, i.e.    
attributes you feel are relevant when carrying out a classication, a regression,
a clustering, an association mining, and an anomaly detection, and what you
hope to accomplish using these techniques. For instance, which attribute do
you wish to explain in the regression based on which other attributes? Which
class label will you predict based on which other attributes in the classication
task? If you need to transform the data to admit these tasks, explain roughly
how you might do this (but don't transform the data now!).


* Describe for the attributes of your data if they are discrete/continous, 
Nominal/Ordinal/Interval/Ratio,

* Give an account of whether there are data issues (i.e. missing values or 
corrupted data) and briefy describe them if so. 

Carry out a principal component analysis (PCA) on your data. There are three 
aspects that needs to be described when you carry out the PCA analysis:

* The amount of variation explained as a function of the number of PCA 
components included,

* the principal directions of the considered PCA components (either find a 
way to plot them or interpret them in terms of the features),

* the data projected onto the considered principal components.

If your attributes have very different scales it may be relevant to standardize 
the data prior to the PCA analysis.
"""

from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

MAXATTRLENGTH = 7
FS_TICKLABEL = 20
FS_AXLABEL = 22
FS_LEGEND = 18

def pca(X, plot=False):
    """
    Modified from Exercise 2.1.3
    """
    fs_ticks = 24
    fs_axlabels = 26
    fs_legend = 22
    nobservations = X.shape[0]

    # Subtract mean value from data
    # Y = X - np.ones((nobservations, 1))*X.mean(axis=0)
    Y = StandardScaler(with_std=False).fit_transform(X)

    # PCA by computing SVD of Y Y = U*S*VT
    U, S, VT = svd(Y, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum() 

    # Plot variance explained
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(range(1, len(rho) + 1), rho, 'ko-')
        # ax.set_title('Variance explained by principal components')
        ax.set_xlabel('Principal component', fontsize=fs_axlabels)
        ax.set_ylabel('Variance explained', fontsize=fs_axlabels)
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fs_ticks) 
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fs_ticks) 

        fname = "../images/breast_cancer/var_from_pcs.png"
        plt.savefig(fname, bbox_inches='tight')
        print(os.path.abspath(fname))
        plt.show()

    return U, S, VT


def pca_directions(X, attributenames, components=[0]):
    """
    Plot contribution to principal components
    """
    fs_ticks = 24
    fs_axlabels = 26
    fs_legend = 22

    ncomponents = len(components)
    nattrs = len(attributenames)
    U, S, VT = pca(X)
    fig, axs = plt.subplots(ncomponents, 1, figsize=(8, 8), sharex=True)
    axs = np.atleast_1d(axs)
    # fig.suptitle('PCA directions')
    for idxcomp, ax in zip(components, axs):
        ax.set_title('PC{}'.format(idxcomp + 1), fontsize=fs_axlabels)
        ax.bar(range(nattrs), VT.T[:,idxcomp], width=0.4,
               label='PC{}'.format(idxcomp + 1), fill=False, color='black')
        ax.axhline(0, color='black')
    
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs_ticks) 


    axs[-1].set_xticks(range(nattrs))
    axs[-1].set_xticklabels([a[:MAXATTRLENGTH] for a in attributenames], rotation=45, fontsize=fs_ticks)
    # axs[-1].set_xlabel('Attributes', fontsize=fs_axlabels)

    fig.tight_layout(pad=3.)
    fname = "../images/breast_cancer/directions_pcs.png"
    plt.savefig(fname, bbox_inches='tight')
    print(os.path.abspath(fname))
    plt.show()


def project_on_pca(X, y, classnames, components=[0, 1], labels=None, plot=True, odir='./', fileid=''):
    """
    Modified from Exercise 2.1.4
    """
    U, S, VT = pca(X)

    # Project the centered data onto principal component space
    # Subtract mean value from data
    nobservations = X.shape[0]
    # Y = X - np.ones((nobservations, 1))*X.mean(axis=0)
    Y = StandardScaler(with_std=False).fit_transform(X)
    Z = np.dot(Y, VT.T)

    # Indices of the principal components to be plotted

    # Plot PCA of the data
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        # fig.suptitle('PCA projection')

        for idxcls, classname in enumerate(classnames):
            class_mask = y == idxcls
            ax.plot(Z[class_mask, components[0]],
                    Z[class_mask, components[1]],
                    'o', label=classname)

        # annonate with labels
        if labels is not None:
            for idobs in labels:
                coords = Z[idobs, components[0]], Z[idobs, components[1]]
                ax.annotate('{}'.format(idobs),
                            xy=coords, xycoords='data',
                            xytext=(7.5, 7.5), textcoords='offset points',
                            # arrowprops=dict(shrink=0.005),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=16)

        leg = ax.legend(fancybox=True, loc='upper right', 
                        fontsize=FS_LEGEND, handletextpad=0.05)
        leg.get_frame().set_alpha(0.6)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(FS_TICKLABEL) 

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(FS_TICKLABEL) 


        ax.set_xlabel('PC{0}'.format(components[0] + 1), fontsize=FS_AXLABEL)
        ax.set_ylabel('PC{0}'.format(components[1] + 1), fontsize=FS_AXLABEL)

        fname = os.path.join(odir, "data_projected_on_pcs{}.png".format(fileid))
        plt.savefig(fname, bbox_inches='tight')
        plt.show()

    return np.stack( [Z[:, c] for c in components], axis=-1 )


