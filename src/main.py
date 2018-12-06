from __future__ import print_function
# from mltools.tmgsimple import TmgSimple
print('HELLO')
import numpy as np
import os 
import ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex10, ex11, ex12
import discussion
import common 
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

def load_data(filename="../data/breast_cancer/dataR2.csv"):
    data = np.genfromtxt(filename, delimiter=',', names=True, dtype=float)
    x = data.view((float, len(data.dtype.names)))

    # convert to 0 and 1 corresponding to indices
    y = x[:, -1].astype('int') - 1 
    classnames = 'Healthy control', 'Patient'

    # Don't use the classification column 
    x = x[:, :-1]
    attributenames = data.dtype.names[:-1]
    return x, y, attributenames, classnames



if __name__ == '__main__':

    # Data loading
    x, y, attributenames, classnames = load_data()
    # print(x)
    print(y)
    print(x.shape)
    print(attributenames)
    print(classnames)


    # Data: Feature extraction and visualization
    x_scaled = zscore(x, ddof=1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_atpc = ex2.project_on_pca(x_scaled, y, classnames, components=list(range(len(attributenames))), plot=False)

    # EXERCISE 2
    # U, S, VT = ex2.pca(x_scaled, plot=True)
    # ex2.pca_directions(x_scaled, attributenames, components=[0,1])
    # ex2.project_on_pca(x_scaled, y, classnames, components=[0, 1], odir = "../images/breast_cancer/")

    # EXERCISE 3
    # ex3.describe(x, attributenames, "../../report/tables/")
    # ex3.corrcoef(x, y, classnames, attributenames)

    # EXERCISE 4
    # ex4.boxplot(x_scaled, y, classnames, attributenames)
    # ex4.hist(x, y, classnames, attributenames)
    # ex4.scatterplot_matrix(x, y, classnames, attributenames, odir="../images/breast_cancer/", fileid='')
    # ex4.matrix(x_scaled, y, classnames, attributenames)

    # EXERCISE 5
    # dtc = ex5.tree_fit(x, y, attributenames, criterion="gini", show=True)
    # idxperson = 50
    # idxperson = 2
    # ex5.tree_predict(x[idxperson,:], ex5.tree_fit(x, y, attributenames, criterion="gini"), 
    #                  classnames, attributenames)


    odir = "../images/tmp/"
    fileid = '_logreg'
    ex5.logreg_fit(x_scaled, y, classnames, attributenames, plot=True, odir=odir, fileid=fileid)

    # transforms = None
    # ex5.linreg_fit(x_scaled, attributenames, 'HOMA', transforms=transforms,
    #                odir="../images/breast_cancer/", fileid='_scaled_notransform')

    # transforms = {'Glucose' : ex5.pow2}  
    # ex5.linreg_fit(x_scaled, attributenames, 'HOMA', transforms=transforms,
    #                odir="../images/breast_cancer/", fileid='_scaled_glucose2')

    # obs_misclassified = ex5.logreg_fit(x, y, classnames, attributenames, plot=True)
    # ex2.project_on_pca(x_scaled, y, classnames, components=[0, 1], labels=obs_misclassified)
    # ex4.correlation(x, y, classnames, attributenames, labels=obs_misclassified)


    # EXERCISE 6
    # classification
    # odir = "../images/breast_cancer/"
    # odir_data = "../../report/tables/"
    # odir = "../images/tmp/"
    # odir_data = "../../report/tmp/"
    # x, y, attributenames, classnames = ex6.testdata_6_1_2()
    # ex6.classification(x, y, attributenames, classnames, odir=odir, odir_data=odir_data)

    # transforms = {'Glucose' : ex5.pow2}
    # odir = "../images/breast_cancer/"
    # odir_data = "../../report/tables/"
    # odir = "../images/tmp/"
    # odir_data = "../../report/tmp/"
    # ex6.linreg_fit_fsel(x_scaled, attributenames, "HOMA", transforms=transforms,
    #                     plot=True, odir=odir, fileid='_fsel_scaled_glucose2', odir_data=odir_data)

    # EXERCISE 7
    # x, y, attributenames, classnames = ex7.get_data_7_1_1(filename='../Data/synth1.mat')
    # ex7.classification(x, y, attributenames, classnames, model='tree')
    # components = None
    # components = [0, 1]
    # ex7.cv_classification(components, x, y, attributenames, classnames)
    # odir = "../images/breast_cancer/"
    # odir_data = "../../report/tables/"
    # odir = "../images/tmp/"
    # odir_data = "../../report/tmp/"
    # ex7.cv_nested(x, y, odir=odir, odir_data=odir_data, fileid='_classification')
    # ex7.cv_nested(x_scaled, y, odir=odir, odir_data=odir_data, fileid='_classification')

    # EXERCISE 8
    # x, y, attributenames, classnames = ex8.get_data_8_3_1()
    # ex8.cv_ann(x, y)
    # ex8.cv_ann(x, y, components=[0, 1])
    # ex8.cv_logreg(x, y)
    # ex8.cv_logreg(x, y, components=[0, 1])
    # odir = "../images/tmp/"
    # ex8.cv_nested(x, y, odir=odir, odir_data=odir_data, fileid='_regression')


    # EXERCISE 10
    # x, y, attributenames, classnames = ex10.get_data(idir='../../../02450Toolbox_Python/Data', fname='synth1.mat')
    # maxclust = 2
    # ex10.cluster(x_scaled, maxclust, y=y, plot=True, odir="../images/tmp/", fileid='_scaled')
    # ex10.cluster(x_atpc, maxclust, y=y, plot=True, odir="../images/tmp/", fileid='_atpca')
    # ex10.nclusters(x, y, odir="../images/tmp/", fileid='_org')
    # ex10.nclusters(x_scaled, y, odir="../images/tmp/", fileid=' _scaled')
    # ex10.nclusters(x_atpc, y, odir="../images/tmp/", fileid=' _scaled')

    # EXERCISE 11
    # x, y, attributenames, classnames = ex11.get_data(idir='../../../02450Toolbox_Python/Data', fname='synth1.mat')
    # maxclust = 4
    # maxclust = 2
    # ex11.cluster(x, classnames, maxclust, y=y, plot=True, odir="../images/tmp/")
    # ex11.cv_gmm(x, classnames, plot=True, odir="../images/tmp/")
    # x = ex11.get_data_11_3_1()
    # ex11.kde_scores(x, n=20)

    # maxclust = 2
    # clusterids = ex11.cluster(x_scaled, classnames, maxclust, y=y, plot=True, odir="../images/tmp/", fileid='_scaled')
    # ex11.clustervalidity(y, clusterids)
    # clusterids = ex11.cluster(x_atpc, classnames, maxclust, y=y, plot=True, odir="../images/tmp/", fileid='_atpc')
    # ex11.clustervalidity(y, clusterids)

    # clusterids = ex11.cluster(x_atpc, classnames, maxclust, y=y, plot=True, odir="../images/tmp/", fileid='_atpc')
    # best_ncluster = ex11.cv_gmm(x_scaled, classnames, plot=True, odir="../images/tmp/", fileid='_scaled')
    # ex11.cv_gmm(x_atpc, classnames, plot=True, odir="../images/tmp/", fileid='_atpca')


    # clusterids = ex11.cluster(x_atpc, classnames, best_ncluster, y=y, plot=True, odir="../images/tmp/", fileid='_atpc')
    # clusterids = ex11.cluster(x_scaled, classnames, best_ncluster, y=y, plot=True, odir="../images/tmp/", fileid='_scaled')
    # ex11.clustervalidity(y, clusterids)


    # odir = "../images/tmp/"
    # odir_data = "../../report/tmp/"
    # n = 20
    # ex11.kde_scores(x_scaled, y, attributenames, n=n,
    #                 classnames=classnames, odir=odir, odir_data=odir_data, fileid='_kde_default')

    # ex11.find_outliers(x_scaled, y, attributenames, n=n, classnames=classnames, odir=odir, odir_data=odir_data,)


    # EXERCISE 12
    # x_b, y, attributenames_b, classnames = ex12.get_data_12_1_5(idir='../../../02450Toolbox_Python/Data')
    # attributenames_b = ex12.rename_attributes(attributenames_b)
    # odir_data = "../../report/tmp/"
    # x_b, attributenames_b = ex12.binarize_data(x, attributenames)
    # attributenames_b = ex12.rename_attributes(attributenames_b)
    # x_b, attributenames_b = ex12.append_onehot_classlabels(x_b, y, attributenames_b, classnames)
    # ex12.write_apriorifile(x_b, attributenames_b, odir=odir_data)
    # bindir = 'C:/Users/sonnej/Dropbox/MachineLearning24-28September2018/02450Toolbox_Python/Tools'
    # apriorifilepath = os.path.join(odir_data, 'AprioriFile.txt')
    # ..\Data\courses.txt
    # ex12.run_mining(apriorifilepath, bindir=bindir)

    # DISCUSSION
    # odir = "../images/tmp/"
    # odir_data = "../../report/tmp/"
    # args = discussion.get_data_synth()
    # discussion.probabilities(*args)

    # confmat, p_S, n = discussion.get_data_breastcancer()
    # confmat = discussion.fit(x, y, attributenames, classnames, odir=odir, odir_data=odir_data, fileid='tree_maxdepth3')
    # discussion.probabilities(confmat, p_S, n)

