"""
Analyze your data by the GMM and use cross-validation to estimate the number
of clusters. (If you encounter the problem "Ill-conditioned covariance" set the reg-
ularization parameter min_covar of gmm to a small constant, say 10-6). Evaluate
how well the clusters of the GMM model correspond to class labels using the cluster
validity measures from last week by assigning observations to the cluster having
highest probability. 

Apply the outlier scoring methods from last exercise in order to
rank all the observations in terms of the Gaussian Kernel density (using the efficient
leave-one-out density estimation approach), KNN density, and KNN average relative
density for some suitable K. Discuss whether it seems there may be outliers in your
data according to the three scoring methods.
"""
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn import model_selection
from mltools.toolbox_02450 import clusterplot, clusterval, gausKernelDensity
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from scipy.stats.kde import gaussian_kde
import pandas as pd
import ex2, ex4

pd.options.display.float_format = '{:,.4f}'.format


def get_data_11_3_1():
    # Draw samples from mixture of gaussians (as in exercise 11.1.1), add outlier
    N = 1000; M = 1
    x = np.linspace(-10, 10, 50)
    X = np.empty((N,M))
    m = np.array([1, 3, 6]); s = np.array([1, .5, 2])
    c_sizes = np.random.multinomial(N, [1./3, 1./3, 1./3])
    for c_id, c_size in enumerate(c_sizes):
        X[c_sizes.cumsum()[c_id]-c_sizes[c_id]:c_sizes.cumsum()[c_id],:] = np.random.normal(m[c_id], np.sqrt(s[c_id]), (c_size,M))
    X[-1,0]=-10 # added outlier
    return X


def get_data(idir='./', fname='synth1.mat'):
    """
    Based on Excercise 11.1.1
    """
    # Load Matlab data file and extract variables of interest
    mat_data = loadmat(os.path.join(idir, fname))
    X = mat_data['X']
    # X[:,1] /= 500
    y = mat_data['y'].squeeze()
    attributenames = [name[0] for name in mat_data['attributeNames'].squeeze()]
    classnames = [name[0][0] for name in mat_data['classNames']]
    return X, y, attributenames, classnames


def cluster(X, classnames, maxclust, y=None, plot=False, odir='./', fileid=''):
    """
    Based on Excercise 11.1.1
    """

    N, M = X.shape
    C = len(classnames)
    # Number of clusters
    K = maxclust
    cov_type = 'full'
    # type of covariance, you can try out 'diag' as well. 
    # JAS: Can be a way around sigular covariance matrices 
    reps = 10
    # number of fits with different initalizations, best result will be kept
    # Fit Gaussian mixture model
    gmm = GaussianMixture(n_components=maxclust, covariance_type=cov_type, n_init=reps, tol=1e-6, max_iter=100).fit(X)
    clusterids = gmm.predict(X)
    print(clusterids[:3])
    print(gmm.predict_proba(X)[:3,:])

    # extract cluster labels
    cds = gmm.means_        

    # extract cluster centroids (means of gaussians)
    covs = gmm.covariances_
    # extract cluster shapes (covariances of gaussians)
    if cov_type.lower() == 'diag':
        new_covs = np.zeros([K, M, M])    
        
        for count, elem in enumerate(covs):
            # temp_m = np.zeros([M, M])
            new_covs[count] = np.diag(elem)

        covs = new_covs

    if plot:
        fig = plt.figure(figsize=(14,9))
        if M <= 2:
            clusterplot(X, clusterid=clusterids, centroids=cds, y=y, covars=covs)
        else:
            ## In case the number of features != 2, then a subset of features most be plotted instead.
            idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
            clusterplot(X[:,idx], clusterid=clusterids, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])

        fname = os.path.join(odir, "gmm_plot{}.png".format(fileid))
        plt.savefig(fname, bbox_inches='tight')
        plt.show()

    return clusterids


def clustervalidity(y, clusterids):
    # compute cluster validities:
    # Rand, Jaccard, NMI = clusterval(y, clusterids)
    print(y)
    print(clusterids)
    adjusted_rand = adjusted_rand_score(y, clusterids)
    jacc = jaccard_similarity_score(y, clusterids)
    nmi = normalized_mutual_info_score(y, clusterids)

    print('Adjusted Rand {}\nJaccard {}\nNMI {}'.format(adjusted_rand, jacc, nmi))


def cv_gmm(X, classnames, plot=False, odir='./', fileid=''):
    # Range of K's to try
    KRange = range(1,6)
    T = len(KRange)

    covar_type = 'full'     # you can try out 'diag' as well
    reps = 10           # number of fits with different initalizations, best result will be kept

    # Allocate variables

    # Apart from cross-validation the optimal number of clusters are sometimes derived
    # by penalizing model complexity based on the Bayesian Information Criteria (BIC)
    # or Akaike's Information Criteria (AIC)

    BIC = np.zeros((T,))
    AIC = np.zeros((T,))

    # cross validation error (-log-likelihood)
    CVE = np.zeros((T,))

    # K-fold crossvalidation
    CV = model_selection.KFold(n_splits=10, shuffle=True)

    for t,K in enumerate(KRange):
            print('Fitting model for K={0}'.format(K))

            # Fit Gaussian mixture model
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)

            # Get BIC and AIC
            BIC[t,] = gmm.bic(X)
            AIC[t,] = gmm.aic(X)

            # For each crossvalidation fold
            for train_index, test_index in CV.split(X):

                # extract training and test set for current CV fold
                X_train = X[train_index]
                X_test = X[test_index]

                # Fit Gaussian mixture model to X_train
                gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

                # compute negative log likelihood of X_test
                CVE[t] += -gmm.score_samples(X_test).sum()
                print(K)
                print(X_test.shape)
                print(gmm.predict_proba(X_test).shape)
                print('Class', gmm.predict(X_test))
                print(gmm.score_samples(X_test).shape)
                # print(gmm.score_samples(X_test))
                print("")
                


    # Plot results

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(KRange, BIC,'-*b', label='BIC')
        ax.plot(KRange, AIC,'-xr', label='AIC')
        ax.plot(KRange, 2*CVE,'-ok', label='Cross-validation error (-log-likelihood)')
        ax.set_xticks(KRange)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14) 

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14) 

        ax.legend(fontsize=13)
        ax.set_xlabel('Cross-validation fold', fontsize=16)

        fname = os.path.join(odir, "gmm_validity{}.png".format(fileid))
        plt.savefig(fname, bbox_inches='tight')

        plt.show()

    return KRange[CVE.argmin()]


def plot_kde_score(n, scores, xticklabels, odir, fileid):
    fs_ticks = 14
    fs_labels = 16
    fig, ax = plt.subplots(1,1, figsize=(0.5*n, 6))
    ax.bar(range(n), scores[:n], fill=False, color='black')
    ax.set_ylabel('Outlier score', fontsize=fs_labels)
    ax.set_xticks(range(n))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Data point number', fontsize=fs_labels)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fs_ticks) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs_ticks) 

    fname = os.path.join(odir, "outlier_scores{}.png".format(fileid))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_kde_boxplot(X, y, classnames, attrnames, df, odir, fileid):
    fig, axs = ex4.boxplot(X, y, classnames, attrnames, show=False, save=False)
    # add outliers to a boxplot
    for icls, _ in enumerate(classnames):
        _df = df.loc[df['Class'] == icls]
        axs[icls].plot(_df["Attribute idx"] + 1, _df["Value"], 'ro', markersize=10, fillstyle='none')

    fname = os.path.join(odir, "boxplot_outliers{}.png".format(fileid))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def df_kde_scores(n, idxs, scores, X, y, attrnames):
    """
    One density per attribute per data objct
    """
    results = {'Score':[], 'Data index':[], 'Attribute':[], 'Value':[], "Class":[], "Attribute idx":[]}
    for i in range(n):
        # print('Density rank {} with density {}'.format(i + 1, scores[i]))
        idx_data, idx_attr = np.unravel_index(idxs[i], X.shape)
        results['Score'].append(scores[i])
        results['Data index'].append(idx_data)
        results['Attribute idx'].append(idx_attr)
        results['Attribute'].append(attrnames[idx_attr])
        results['Value'].append(X[idx_data, idx_attr])
        results['Class'].append(y[idx_data])

    df = pd.DataFrame.from_dict(results)
    return df


def df_kde_scores2(n, idxs, scores):
    """
    One density per data object
    """
    results = {'Score':[], 'Data index':[]}
    for i in range(n):
        results['Score'].append(scores[i])
        results['Data index'].append(idxs[i])

    df = pd.DataFrame.from_dict(results)
    return df


def kde_scores(X, y, attrnames, n=None, classnames=None, plot=True, odir='./', odir_data='./', fileid=''):
    """
    Modified from 11.3.1
    """

    # bw_method = 5.
    # bw_method = 'silverman'
    bw_method = 'scott'

    # Compute kernel density estimate (row major i.e. first N elements are the properties of first subject)
    kde = gaussian_kde(X.ravel())#, bw_method=bw_method)

    scores = kde.evaluate(X.ravel())
    # JAS: last score corresponds to the outlier 
    # print(scores)
    print(np.argmin(scores), len(X[:,0]))
    idxs = scores.argsort()
    scores.sort()

    # use all
    if n is None:
        n = X.shape[0]

    df = df_kde_scores(n, idxs, scores, X, y, attrnames)



    # Plot kernel density estimate
    if plot:
        plot_kde_score(n, scores, df["Data index"], odir, fileid)
        plot_kde_boxplot(X, y, classnames, attrnames, df, odir, fileid)

    df["Class"] = df["Class"].replace(to_replace={1 : "Patient", 0 : "Healty"})
    print(df.to_string(index=False))

    fname = "kde_scores{}.txt".format(fileid)
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write(df.drop("Attribute idx", axis=1).to_latex(index=False))


    return df['Data index']


def xval_kde_score(X, y, attrnames, n=None, classnames=None, plot=True, odir='./', odir_data='./', fileid=''):
    """
    Modified from 11.3.2
    """
    widths = 2.0**np.arange(-10, 10)
    logP = np.zeros(np.size(widths))
    for i, w in enumerate(widths):
        f, log_f = gausKernelDensity(X, w)
        logP[i] = log_f.sum()

    val = logP.max()
    ind = logP.argmax()

    width=widths[ind]
    print('Optimal estimated width is: {0}'.format(width))

    # Estimate density for each observation not including the observation
    # itself in the density estimate
    densities, log_density = gausKernelDensity(X, width)

    # Sort the densities
    idxs = (densities.argsort(axis=0)).ravel()
    densities = densities[idxs].flatten()

    print(idxs)

    # use all
    if n is None:
        n = X.shape[0]


    df = df_kde_scores2(n, idxs, densities)

    print(df)

    # Display the index of the lowest density data object
    print('Lowest density: {0} for data object: {1}'.format(densities[0], idxs[0]))
    print('Second lowest density: {0} for data object: {1}'.format(densities[1], idxs[1]))

    fname = "kde_scores_xval{}.txt".format(fileid)
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write(df.drop("Attribute idx").to_latex())


    # Plot density estimate of outlier score
    if plot:
        plot_kde_score(n, densities, idxs[:n], odir, fileid)

    plt.figure(2)
    plt.plot(logP)
    plt.title('Optimal width')
    plt.show()

    return df['Data index']



def knn_scores(X, y, attrnames, n=None, classnames=None, plot=True, odir='./', odir_data='./', fileid=''):
    """
    K-neighbors density estimator

    Modified from Exercise 11.4.1
    """

    ### K-neighbors density estimator
    # Neighbors to use:
    K = 5

    # Find the k nearest neighbors
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    distances, i = knn.kneighbors(X)

    densities = 1./(distances.sum(axis=1)/K)

    # Sort the scores
    idxs = densities.argsort()
    densities = densities[idxs]


    # use all
    if n is None:
        n = X.shape[0]

    df = df_kde_scores2(n, idxs, densities)

    print(df)

    fname = "kde_scores{}.txt".format(fileid)
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write(df.to_latex())


    if plot:
        plot_kde_score(n, densities, idxs[:n], odir, fileid)

    return df['Data index']

def knn_avg_scores(X, y, attrnames, n=None, classnames=None, plot=True, odir='./', odir_data='./', fileid=''):
    """
    K-nearest neigbor average relative density

    Modified from Exercise 11.4.1
    """
    # Neighbor to use:
    K = 5

    knn = NearestNeighbors(n_neighbors=K).fit(X)
    
    distances, idxs = knn.kneighbors(X)
    print(idxs[:5,:])
    print(distances[:5,:])

    density = 1./(distances.sum(axis=1)/K)
    avg_rel_density = density/(density[idxs[:,1:]].sum(axis=1)/K)

    # Sort the avg.rel.densities
    i_avg_rel = avg_rel_density.argsort()
    avg_rel_density = avg_rel_density[i_avg_rel]

    # use all
    if n is None:
        n = X.shape[0]

    df = df_kde_scores2(n, i_avg_rel, avg_rel_density)

    print(df)

    fname = "kde_scores{}.txt".format(fileid)
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write(df.to_latex())


    if plot:
        plot_kde_score(n, avg_rel_density, i_avg_rel[:n], odir, fileid)

    return df['Data index']


def find_outliers(x_scaled, y, attributenames, n=None, classnames=None, odir='./', odir_data='./'):

    idxs_data_knn = knn_scores(x_scaled, y, attributenames, n=n,
                               classnames=classnames, odir=odir, odir_data=odir_data, fileid='_knn_5')

    idxs_data_knn_avg = knn_avg_scores(x_scaled, y, attributenames, n=n,
                                       classnames=classnames, odir=odir, odir_data=odir_data, fileid='_knn_avg5')

    idxs_data_kde = xval_kde_score(x_scaled, y, attributenames, n=n,
                                   classnames=classnames, odir=odir, odir_data=odir_data, fileid='_kde_xval')


    _outliers = sorted(list(set(idxs_data_knn).intersection(set(idxs_data_kde)).intersection(set(idxs_data_knn_avg))))
    print('Number of outliers {} of {}'.format(len(_outliers), n))
    print('Outliers', _outliers)
    fname = "outliers.txt"
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write(', '.join([str(idx) for idx in _outliers]))

    fname = "outlier_n.txt"
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write(str(n))

    ex2.project_on_pca(x_scaled, y, classnames, components=[0, 1], labels=_outliers, odir=odir, fileid='_outliers')


    return _outliers