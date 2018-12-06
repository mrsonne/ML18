"""
For the regression part of the report do:
* Fit apart from the linear regresssion model also an artificial neural network
(ANN) model to the data and estimate in an inner cross-validation loop the
number of hidden units. Give in a table the performance results for the methods
evaluated on the same cross-validation splits on the outer cross-validation
loop, i.e. you should use two levels of cross-validation).

* Statistically evaluate if there is a signicant performance difference between
the tted ANN and linear regression models (i.e., considering the credibility
interval equivalent to the use of a paired t-test as described in lecture and
exercise 6 ). Compare in addition if the performance of your models are
better than simply predicting the output to be the average of the training
data output.

If your data has previously been analyzed by regression in the literature, please
report what methods have been used previously as well as their performance and
relate your results to these previous results.
Notice, if the analysis of your data is too computationally demanding for choosing
parameters in the inner cross-validation loop we suggest you use the hold-out method
instead of K-fold cross-validation. Furthermore, if analyzing the data by ANN is
too computationally demanding you can consider only analyzing a subset of your
data by ANN.
"""
import common
import sklearn.neural_network as nn
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from mltools.toolbox_02450 import dbplotf
import ex2


def get_data_8_3_1(filename='../Data/synth1.mat'):
    mat_data = loadmat(filename)
    X = mat_data['X']
    y = mat_data['y'].squeeze()
    attributenames = [name[0] for name in mat_data['attributeNames'].squeeze()]
    classnames = [name[0][0] for name in mat_data['classNames']]
    return X, y, attributenames, classnames

def _dummy_xvalidator(x, y, k_inner):
    basemodel = DummyClassifier(strategy="most_frequent")
    attrname, attr_values = "strategy", ["most_frequent"]
    splitter = common.KfoldSplitter(x, y, k_inner)
    name = '{}'.format(basemodel.__class__.__name__).replace('Classifier', '')
    return common.CrossValidator(splitter, attrname, attr_values, basemodel, name=name)


def _ann_xvalidator(x, y, k_inner):
    basemodel = nn.MLPClassifier(solver='lbfgs', alpha=1e-4,
                                  hidden_layer_sizes=(1,),
                                  random_state=1)

    splitter = common.KfoldSplitter(x, y, k_inner)
    attrname = "hidden_layer_sizes" 
    # using tuple currently messes up my pandas tables
    # attr_values = [(n,) for n in np.arange(1, 10, 1)]
    # for one layer model we can just use integers
    attr_values = np.arange(1, 10, 1)
    name = '{}'.format(basemodel.__class__.__name__)
    return common.CrossValidator(splitter, attrname, attr_values, basemodel, name=name)


def _logreg_xvalidator(x, y, k_inner):
    basemodel = lm.LogisticRegression(solver='lbfgs', 
                                      multi_class='multinomial',
                                      tol=1e-4, 
                                      random_state=1)

    splitter = common.KfoldSplitter(x, y, k_inner)
    attrname , attr_values = "C", np.logspace(-2, 2, 20)
    name = '{}'.format(basemodel.__class__.__name__)
    return common.CrossValidator(splitter, attrname, attr_values, basemodel, name=name)


def _cv_reg(method, x, y, components, plot=True):
    """
    General cross validation method
    """
    if components is not None:
        print('Using principal components {}'.format(components))
        classnames = None
        x_scaled = StandardScaler().fit_transform(x)
        _x = ex2.project_on_pca(x_scaled, y, classnames, components=components, plot=False)
    else:
        _x = x

    k_inner = 10
    xval = method(_x, y, k_inner)
    best_model, best_value = xval.run()

    if plot and _x.shape[1] == 2:
        def neval(xloc):
            return np.argmax(best_model.predict_proba(xloc), 1)

        plt.figure(1)
        dbplotf(_x, y, neval, 'auto')
        plt.show()


def cv_logreg(x, y, components=None, plot=True):
    """
    Cross validate multinomial regression
    """
    _cv_reg(_logreg_xvalidator, x, y, components, plot=True)


def cv_ann(x, y, components=None, plot=True):
    """
    Cross validate ANN regression
    """
    _cv_reg(_ann_xvalidator, x, y, components, plot=True)


def _get_xvalidators(k_inner):
    """
    Get cross validators for nested validation
    """
    return (_logreg_xvalidator([], [], k_inner),
            _ann_xvalidator([], [], k_inner),
            _dummy_xvalidator([], [], k_inner))


def cv_nested(x, y, odir="./", odir_data="./", fileid=''):
    k_inner, k_outer = 10, 10
    xvalidators = _get_xvalidators(k_inner)
    outer_splitter = common.KfoldSplitter(x, y, k_outer)
    xval_nested = common.NestedCrossValidator(outer_splitter, xvalidators)
    xval_nested.run()
    xval_nested.plot(odir, fileid)
    xval_nested.write_tables(odir_data, fileid)
