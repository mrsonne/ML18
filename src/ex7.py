"""
For the classication part of the report:

1. Classify your data not only using Decision trees but also considering KNN and
Naive Bayes. (Use cross-validation to select relevant parameters (i.e., pruning
level for the decision tree and number of neighbors for KNN ) in an inner crossvalidation
loop and give in a table the performance results for the methods
evaluated on the same cross-validation splits on the outer cross-validation loop,
i.e. you should use two levels of cross-validation).

2. Statistically compare the performance of the two best performing models (i.e.,
considering the credibility interval equivalent to the use of a paired t-test as
described in lecture and exercise 6). Compare in addition if the performance
of your models are better than simply predicting all outputs to be the largest
class in the training data.

If your data has previously been analyzed by classication in the literature, please
report what methods have been used previously as well as their performance and
relate your results to these previous results.

Notice, if the analysis of your data is too computationally demanding for choosing
parameters in the inner cross-validation loop we suggest you use the hold-out method
instead of K-fold cross-validation.
"""
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from scipy.io import loadmat
import ex6
import common
import numpy as np
import sys
import ex2


def get_data_ex7_2_3():
    # Load list of names from files
    fmale = open('../Data/male.txt','r')
    ffemale = open('../Data/female.txt','r')
    mnames = fmale.readlines(); fnames = ffemale.readlines();
    names = mnames + fnames
    gender = [0]*len(mnames) + [1]*len(fnames)
    fmale.close(); ffemale.close();

    # Extract X, y and the rest of variables. Include only names of >4 characters.
    X = np.zeros((len(names),4))
    y = np.zeros((len(names),1))
    n=0
    for i in range(0,len(names)):
        name = names[i].strip().lower()
        if len(name)>3:
            X[n,:] = [ord(name[0])-ord('a')+1, ord(name[1])-ord('a')+1, ord(name[-2])-ord('a')+1, ord(name[-1])-ord('a')+1]
            y[n,0] = gender[i]
            n+=1
    X = X[0:n,:]; y = y[0:n,:];

    N, M = X.shape; C = 2
    attributeNames = ['1st letter', '2nd letter', 'Next-to-last letter', 'Last letter']
    classNames = ['Female', 'Male'];

    return X, y.squeeze(), attributeNames, classNames


def get_data_7_1_1(filename='../Data/synth2.mat'):
    mat_data = loadmat(filename)
    X = mat_data['X']
    y = mat_data['y'].squeeze()
    attributenames = [name[0] for name in mat_data['attributeNames'].squeeze()]
    classnames = [name[0][0] for name in mat_data['classNames']]
    return X, y, attributenames, classnames


def classification_fit(attributenames, classnames, cross_validator, plot=True, show=True, plot_on_pcs=False, fit=True,
                       odir='./', odir_data='./', fileid=''):
    """
    Modified from Exercise 7.1.1'

    Use a cross validator....
    """


    # extract raw model
    model = cross_validator.basemodel
    if fit:
        X_train, X_test, y_train, y_test, _ = cross_validator.splitter.split()
        cross_validator.fit(model, X_train, y_train)
    else:
        X_train = None
        y_train = None
        X_test = cross_validator.splitter.X
        y_test = cross_validator.splitter.y

    _, _, objs_misclassified, confmat, y_pred_test = cross_validator.xval_error(model, X_test, y_test, X_train, y_train)
    title = cross_validator.name


    if plot_on_pcs:
        _X = StandardScaler().fit_transform(cross_validator.splitter.X)
        U, S, VT = ex2.pca(_X, plot=False)

        _X_test = StandardScaler().fit_transform(X_test)
        _X_test = np.dot(_X_test, VT.T)

        if X_train is not None:
            _X_train = StandardScaler().fit_transform(X_train)
            _X_train = np.dot(_X_train, VT.T)
        else:
            _X_train = X_train


        y = None
        components = range(len(attributenames))

        _attributenames = ['PC{}'.format(icomp + 1) for icomp in components]
    else:
        _attributenames = attributenames
        _X_train = X_train
        _X_test = X_test


    if plot:
        common.classification_plot(classnames, _attributenames, _X_train, _X_test, y_train, confmat, 
                                   y_pred_test, objs_misclassified, title, show, 
                                   odir=odir, odir_data=odir_data, fileid=fileid)


    return objs_misclassified, confmat


def _tree_xvalidator(x, y, k_inner):
    basemodel = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=5, min_samples_leaf=5)
    attrname , attr_values = "max_depth", np.arange(1, 10, 1)
    splitter = common.KfoldSplitter(x, y, k_inner)
    name = '{}'.format(basemodel.__class__.__name__).replace('Classifier', '')
    return common.CrossValidator(splitter, attrname, attr_values, basemodel, name=name)


def _knn_xvalidator(x, y, k_inner):
    basemodel = KNeighborsClassifier(n_neighbors=1,
                                      algorithm='brute',
                                    #   p=2,
                                      metric='mahalanobis',
                                      metric_params={"V":np.array([[1,0],[0,1]])},
                                     )
    attrname , attr_values = "n_neighbors", np.arange(1, 5, 1)
    splitter = common.KfoldSplitter(x, y, k_inner)
    name = '{}'.format(basemodel.__class__.__name__).replace('Classifier', '')
    return common.CrossValidator(splitter, attrname, attr_values, basemodel, name=name)

def _knn_p2_xvalidator(x, y, k_inner):
    basemodel = KNeighborsClassifier(n_neighbors=1,
                                      algorithm='brute',
                                      p=2,
                                     )
    attrname , attr_values = "n_neighbors", np.arange(1, 5, 1)
    splitter = common.KfoldSplitter(x, y, k_inner)
    name = '{}'.format(basemodel.__class__.__name__ + '_p2').replace('Classifier', '')
    return common.CrossValidator(splitter, attrname, attr_values, basemodel, name=name)


def _mnb_xvalidator(x, y, k_inner):
    basemodel = MultinomialNB(alpha=0.01, fit_prior=True)
    attrname , attr_values = "alpha", np.linspace(0.01, 1, 10)
    splitter = common.KfoldSplitter(x, y, k_inner)
    name = '{}'.format(basemodel.__class__.__name__).replace('Classifier', '')
    return common.CrossValidator(splitter, attrname, attr_values, basemodel, name=name)
    

def _dummy_xvalidator(x, y, k_inner):
    basemodel = DummyClassifier(strategy="most_frequent")
    attrname, attr_values = "strategy", ["most_frequent"]
    splitter = common.KfoldSplitter(x, y, k_inner)
    name = '{}'.format(basemodel.__class__.__name__).replace('Classifier', '')
    return common.CrossValidator(splitter, attrname, attr_values, basemodel, name=name)


def _get_xvalidators(k_inner, x, y):
    """
    Get cross validators for nested validation
    """
    return (
            f(x, y, k_inner),
            _knn_xvalidator(x, y, k_inner),
            # _knn_p2_xvalidator(x, y, k_inner),
            _mnb_xvalidator(x, y, k_inner),
            _dummy_xvalidator(x, y, k_inner),
           )


def cv_classification(components, x=None, y=None, attributenames=None, classnames=None):
    """
    Non-nested test for debugging
    """
    if x is None or y is None:
        filename = '../Data/synth2.mat'
        x, y, attributenames, classnames = get_data_7_1_1(filename)

    
    if components is not None:
        # requires disabling _mnb_xvalidator
        print('Using principal components {}'.format(components))
        x_scaled = StandardScaler().fit_transform(x)
        _x = ex2.project_on_pca(x_scaled, y, classnames, components=components, plot=False)
    else:
        _x = x

    k_inner = 10
    xvalidators = _get_xvalidators(k_inner, _x, y)
        

    for xval in xvalidators:
        best_model, best_value = xval.run()

    xvalidators[0].plot()

    xvalidators[0].splitter = common.HoldoutSplitter(_x, y)
    classification_fit(attributenames, classnames, xvalidators[0])


def cv_nested(x, y, odir="./", odir_data="./", fileid=''):
    """
    Nested cross validation
    """
    k_inner, k_outer = 10, 10
    # init with empty splitters to get aan error if the data is not set properly
    xvalidators = _get_xvalidators(k_inner, [], [])
    outer_splitter = common.KfoldSplitter(x, y, k_outer)
    xval_nested = common.NestedCrossValidator(outer_splitter, xvalidators)
    xval_nested.run()
    xval_nested.plot(odir, fileid)
    xval_nested.write_tables(odir_data, fileid)


def classification(x, y, attributenames, classnames, model='tree', odir='./', odir_data='./', fileid=''):
    holdout_splitter = common.HoldoutSplitter(x, y)

    if model == 'tree':
        model = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=5, min_samples_leaf=5)
    elif model == 'tree_maxdepth3':
        model = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=5, min_samples_leaf=5, max_depth=3)
    elif model == 'knn_mahalanobis':
        model = KNeighborsClassifier(n_neighbors=1,
                                     algorithm='brute',
                                     metric='mahalanobis',
                                     metric_params={"V":np.array([[1,0],[0,1]])},
                                    )
    elif model == 'knn_p2':
        model = KNeighborsClassifier(n_neighbors=1, p=2)
    else:
        print('Unknown model')
        sys.exit()

    modelname = '{} model'.format(model.__class__.__name__) 
    attrname, attr_values = None,  []
    xval = common.CrossValidator(holdout_splitter, attrname, attr_values, model, name=modelname)
    _, confmat = classification_fit(attributenames, classnames, xval, odir=odir, odir_data=odir_data, fileid=fileid)
    return confmat