"""
Create now a section in the report entitled clustering. In this part of the report you
should attempt to cluster your data and evaluate how well your clustering reflects 
the labeled information. If your data is a regression problem define two or more
classes by dividing your output into intervals defining two or more classes as you
did in the classication part of the report.

* Perform a hierarchical clustering of your data using a suitable dissimilarity
measure and linkage function. Try to interpret the results of the hierarchical
clustering.

* Evaluate the quality of the clustering in terms of your label information for
different cut-offs in the dendrogram of number of clusters.
"""


# exercise 10.2.1
from __future__ import print_function
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mltools.toolbox_02450 import clusterplot, clusterval
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import k_means
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
#cluster_metrics.supervised.normalized_mutual_info_score

import os
import numpy as np

def get_data(idir='./', fname='synth1.mat'):
    """
    Based on Excercise 10.2.1
    """
    # Load Matlab data file and extract variables of interest
    mat_data = loadmat(os.path.join(idir, fname))
    X = mat_data['X']
    y = mat_data['y'].squeeze()
    attributenames = [name[0] for name in mat_data['attributeNames'].squeeze()]
    classnames = [name[0][0] for name in mat_data['classNames']]
    # N, M = X.shape
    # C = len(classnames)
    return X, y, attributenames, classnames


def cluster(X, maxclust, y=None, plot=False, odir='./', fileid=''):
    """
    Based on Excercise 10.2.1

    Nice blog: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    """
    # Perform hierarchical/agglomerative clustering on data matrix
    # https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
    # method = 'single'
    # method = "complete"
    # method = "average"
    # method = "weighted"
    # method = "centroid"
    # method = "median"
    # method = 'ward'

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    # metric = 'euclidean'
    # metric = 'cosine'
    # metric = 'mahalanobis' 
    # metric = "correlation"
    # metric = "minkowski"

    # method, metric = "average", "correlation"
    # method, metric = "weighted", "correlation"
    # method, metric = "complete", "euclidean"
    method, metric = "complete", "correlation"
    # method, metric = "complete", "cosine"
    # method, metric = "complete", "minkowski"
    # method, metric = "ward", "euclidean"
    # method, metric = "single", "euclidean"

    Z = linkage(X, method=method, metric=metric)

    # Compute and display clusters by thresholding the dendrogram
    clusterids = fcluster(Z, criterion='maxclust', t=maxclust)

    # make cluster ids zero-based
    clusterids -= 1
    print(clusterids)

    # Cophenetic Correlation Coefficient
    # c, coph_dists = cophenet(Z, pdist(X))
    # print('Cophenetic Correlation Coefficient', c)

    if plot:
        plt.figure(1)
        clusterplot(X, clusterids.reshape(clusterids.shape[0], 1), y=y)
        fname = os.path.join(odir, "clustering_classes_{}_{}{}.png".format(method, metric, fileid))
        plt.savefig(fname, bbox_inches='tight')

        # Display dendrogram
        max_display_levels=4
        fs_ticks = 16
        fig, ax = plt.subplots(1, 1, figsize=(10,4))
        ax.set_xlabel('Sample index or (cluster size)', fontsize=fs_ticks)
        ax.set_ylabel('Distance', fontsize=16)
        dendrogram(Z, truncate_mode='level', p=max_display_levels, leaf_font_size=14., ax=ax)

        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fs_ticks) 

        # dendrogram(Z,
        #            leaf_rotation=90.,  # rotates the x axis labels
        #            leaf_font_size=8.,  # font size for the x axis labels)
        #            )

        # dendrogram(
        #     Z,
        #     truncate_mode='lastp',  # show only the last p merged clusters
        #     p=12,  # show only the last p merged clusters
        #     leaf_rotation=90.,
        #     leaf_font_size=12.,
        #     show_contracted=True,  # to get a distribution impression in truncated branches
        # )

        fname = os.path.join(odir, "clustering_dendrogram_{}_{}{}.png".format(method, metric, fileid))
        plt.savefig(fname, bbox_inches='tight')
        plt.show()

    return clusterids


def nclusters(X, y, odir='./', fileid=''):
    """
    Based on 10.1.3
    """
    # Maximum number of clusters:
    K = 10

    # Allocate variables:
    Rand = np.empty((K,))
    Jaccard = np.empty((K,))
    NMI = np.empty((K,))
    adjusted_rand = np.empty((K,))
    jacc = np.empty((K,))
    # anmi = np.empty((K,))

    for k in range(K):
        print(k + 1)
        clusterids = cluster(X, k + 1)
        # _, clusterids, _ = k_means(X, k + 1)
        print(y)

        # compute cluster validities:
        Rand[k], Jaccard[k], NMI[k] = clusterval(y, clusterids)
        adjusted_rand[k] = adjusted_rand_score(y, clusterids)
        jacc[k] = jaccard_similarity_score(y, clusterids)
        NMI[k] = normalized_mutual_info_score(y, clusterids)
        # anmi[k] = adjusted_mutual_info_score(y, clusterids)

    # Plot results:

    fig, ax = plt.subplots(1, 1)
    # ax.set_title('Cluster validity', fontsize=20)
    # ax.plot(np.arange(K) + 1, Rand, label='Rand')
    # ax.plot(np.arange(K) + 1, Jaccard, label='Jaccard')
    ax.plot(np.arange(K) + 1, NMI, label='NMI', color='black')
    ax.plot(np.arange(K) + 1, adjusted_rand, label='Adjusted Rand', color='magenta')
    ax.plot(np.arange(K) + 1, jacc, label='Jaccard', color='green')
    # ax.plot(np.arange(K) + 1, anmi, label='ANMI')
    ax.set_xlabel('Number of clusters', fontsize=20)
    ax.set_ylabel('Similarity', fontsize=20)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, K + 1))
    # ax.set_ylim(-1.05, 1.05)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18) 

    ax.legend(loc='upper right', fontsize=20)
    fname = os.path.join(odir, "clustering_validity{}.png".format(fileid))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
