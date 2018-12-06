"""
For the regression part of the report do:
    1. Apply linear regression with forward selection and consider if transforming or
    combining attributes potentially may be useful. For linear regression, plotting
    the residual error vs. the attributes can give some insight into whether in-
    cluding a transformation of a variable can improve the model, i.e. potentially
    describe parts of the residuals.

    2. Explain how a new data observation is predicted according to the estimated
    model. I.e. what are the effects of the selected attributes in terms of predicting
    the data. (Notice, if you interpret the magnitude of the estimated coefficients this in
    general requires that each attribute be normalized prior to the analysis.).

For the classication part do:
    1. Apply Decision Trees and use cross-validation to select an adequate pruning
    level.

    2. Interpret how a new data observation is classified according to the fitted tree.
    (If you have multiple models fitted, (i.e., one for each cross-validation split)
    either focus on one of these fitted models or consider fitting one model for
    the optimal setting of the parameters estimated by cross-validation to all the
    data.)
"""
from __future__ import print_function
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mpltransforms
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from sklearn import model_selection#, tree
import sklearn.linear_model as lm
from sklearn import tree
import numpy as np
from mltools.toolbox_02450 import feature_selector_lr
import common
from shutil import copyfile
from itertools import chain
import ex4, ex5, ex7

def bmplot(yt, xt, X, ax):
    ''' 
    Function plots matrix X as image with lines separating fields.

    Modified from mltools  
    '''
    im = ax.imshow(X, interpolation='none', cmap='bone')
    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels(xt, fontsize=16)
    ax.set_yticks(range(len(yt)))
    ax.set_yticklabels(yt, rotation=45, fontsize=16)

    for i in range(0, len(yt)):
        ax.axhline(i - 0.5, color='black')

    for i in range(0, len(xt)):
        ax.axvline(i - 0.5, color='black')

    return im

def testdata_6_1_2():
    """
    Test data from Exercise 6.1.2. For validation purposes only
    """
    from scipy.io import loadmat
    mat_data = loadmat('../Data/wine2.mat')
    x = mat_data['X']
    y = mat_data['y'].squeeze()
    attributenames = [name[0] for name in mat_data['attributeNames'][0]]
    classnames = [name[0][0] for name in mat_data['classNames']]
    return x, y, attributenames, classnames


def testdata_6_2_1():
    """
    Test data from Exercise 6.2.1. For validation purposes only
    """
    from scipy.io import loadmat
    mat_data = loadmat('../Data/body.mat')
    x = mat_data['X']
    y = mat_data['y'].squeeze()
    attributenames = [name[0] for name in mat_data['attributeNames'][0]]
    return x, y, attributenames


def classification(x, y, attributenames, classnames, odir="./", odir_data="./",):
    attrname , attr_values = "max_depth", np.arange(2, 21, 1)
    splitter = common.KfoldSplitter(x, y, 10)
    model = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=5, min_samples_leaf=5)
    xval = common.CrossValidator(splitter, attrname, attr_values, model, name='Cross validation DTC')
    best_model, best_value = xval.run()
    xval.plot(odir)
    xval.write_tables(odir=odir_data, fileid="_dtc")
    # plot example of classification for the best model
    xval.basemodel = best_model
    xval.splitter = common.HoldoutSplitter(x, y, test_proportion=.99)
    objs_misclassified, _ = ex7.classification_fit(attributenames, classnames, xval,
                                                plot=True, show=True, plot_on_pcs=True, fit=False)
    ex4.scatterplot_matrix(x, y, classnames, attributenames,
                           labels=objs_misclassified,
                           odir=odir, fileid='_dtc')
    fname = "tree"
    ex5.render_tree(best_model, attributenames, odir=odir, fname=fname)



def linreg_fit_fsel(X, attributenames, yname, transforms=None, plot=True, show=False, odir="./", fileid='', odir_data='./'):
    fs_labels = 18
    fs_axlabels = 18
    fs_ticks = 16
    fs_title = 20

    _X, y, tranformstrs, ytransformstr, _attributenames = ex5.prepare_data(X, attributenames, yname, transforms)

    N, M = _X.shape

    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 5
    CV = model_selection.KFold(n_splits=K,shuffle=True)

    # Feature selections
    internal_cross_validation = 10

    # Initialize variables
    Features = np.zeros((M, K))

    # Error with all fratures included in model
    error_train = np.empty((K, 1))
    error_test = np.empty((K, 1))

    error_train_fs = np.empty((K, 1))
    error_test_fs = np.empty((K, 1))

    # Average square deviation in the data. Used in R^2 calculation
    error_train_nofeatures = np.empty((K, 1))
    error_test_nofeatures = np.empty((K, 1))
    fnames = []

    split_generator = CV.split(_X)
    for k in range(K):
        
        train_index, test_index = next(split_generator) 

        # extract training and test set for current CV fold
        X_train = _X[train_index,:]
        y_train = y[train_index]
        X_test = _X[test_index,:]
        y_test = y[test_index]
        
        # Compute squared error without using the input data at all
        error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum()/y_train.shape[0]
        error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum()/y_test.shape[0]

        # Compute squared error with all features selected (no feature selection)
        m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        error_train[k] = np.square(y_train - m.predict(X_train)).sum()/y_train.shape[0]
        error_test[k] = np.square(y_test - m.predict(X_test)).sum()/y_test.shape[0]

        # Compute squared error with feature subset selection
        #textout = 'verbose';
        textout = ''
        (selected_features,
        features_record,
        loss_record) = feature_selector_lr(X_train,
                                           y_train,
                                           internal_cross_validation,
                                           display=textout)

        print()
        print(loss_record)
        print(features_record)

        Features[selected_features, k] = 1
        # .. alternatively you could use module sklearn.feature_selection
        if len(selected_features) is 0:
            print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
        else:
            m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
            error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
            error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
        
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(range(1, len(loss_record)), loss_record[1:], 'k-o')
            axs[0].plot([], [], color='none', label='Fold {}'.format(k + 1))
            axs[0].set_xticks(range(1, len(loss_record)))
            axs[0].set_xlabel('Iteration', fontsize=20)
            axs[0].set_ylabel('Squared error (crossvalidation)', fontsize=20)

            for tick in axs[0].xaxis.get_major_ticks():
                tick.label.set_fontsize(16) 

            for tick in axs[0].yaxis.get_major_ticks():
                tick.label.set_fontsize(16) 


            leg = axs[0].legend(loc='upper right', fontsize=fs_labels)
            leg.get_frame().set_linewidth(0.0)
            leg.get_frame().set_facecolor('none')

            im = bmplot(_attributenames, range(1, features_record.shape[1]), -features_record[:,1:], axs[1])
            im.set_clim(-1.5, 0)
            axs[1].xaxis.set_ticks_position('none') 
            axs[1].set_xlabel('Iteration', fontsize=20)
            fig.tight_layout()
            fname = "cv_feature_selection_fold{}{}.png".format(k + 1, fileid)
            fname = os.path.join(odir, fname)
            fnames.append(fname)
            plt.savefig(fname, bbox_inches='tight')



        print('Cross validation fold {0}/{1}'.format(k + 1, K))
        print("Selected feature indices", selected_features)
        print("Selected feature names", [_attributenames[idx] for idx in selected_features])
        # print('Train indices: {0}'.format(train_index))
        # print('Test indices: {0}'.format(test_index))
        print('Feature count: {0}\n'.format(selected_features.size))


    kmin = np.argmin(error_test_fs)

    # Display results
    fstr = "- {0:15}: {1}"
    print('\nLinear regression without feature selection:')
    print(fstr.format('Training error', error_train.mean()))
    print(fstr.format("Test error", error_test.mean()))
    print(fstr.format("R^2 train", (error_train_nofeatures.sum()-error_train.sum())/error_train_nofeatures.sum()))
    print(fstr.format("R^2 test", (error_test_nofeatures.sum()-error_test.sum())/error_test_nofeatures.sum()))

    print('\nLinear regression with feature selection:')
    print(fstr.format("Training error", error_train_fs.mean()))
    print(fstr.format("Test error", error_test_fs.mean()))
    print(fstr.format("R^2 train", (error_train_nofeatures.sum()-error_train_fs.sum())/error_train_nofeatures.sum()))
    print(fstr.format("R^2 test", (error_test_nofeatures.sum()-error_test_fs.sum())/error_test_nofeatures.sum()))


    print('Optimal features', Features[:, kmin])
    print('For fold', kmin + 1)
    # Which feature were selected in each on the K folds
    fig, ax = plt.subplots(1, 1)
    im = bmplot(_attributenames, range(1, Features.shape[1] + 1), -Features, ax)

    # mark best features
    mpl.rcParams['hatch.linewidth'] = 1.5 
    rects = [Rectangle((kmin - 0.5, idx - 0.5), 1, 1) for idx, val in enumerate(Features[:, kmin]) if val == 1]
    pc = PatchCollection(rects, facecolor='none', edgecolor='black', linewidth=0, hatch='//')
    ax.add_collection(pc)
    # https://matplotlib.org/users/transforms_tutorial.html#blended-transformations
    trans = mpltransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for idx, err in enumerate(error_test_fs.flatten()):
        testerrstr = '{:5.2e}'.format(err)
        t = ax.annotate(testerrstr,
                        xy=(idx, 1.02), # imshow has y=0 at top
                        xycoords=trans,
                        fontsize=16,
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        xytext=(0, 0), 
                        textcoords='offset points',
                        rotation=45,
                        clip_on=False)




    # im = bmplot(_attributenames, range(1, Features.shape[1] + 1), -best_features, ax)
    im.set_clim(-1.5, 0)
    ax.set_xlabel('Crossvalidation fold', fontsize=20)
    # ax.set_ylabel('Attribute', fontsize=20)
    
    fname = "fsel_best_fold.txt"
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write(str(kmin + 1))

    fname = "fsel_err.txt"
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write('{:5.2e}'.format(error_test_fs.flatten()[kmin]))

    fname = "fsel_fname.txt"
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write(fnames[kmin])

    new_name = fnames[kmin].replace('fold{}'.format(kmin+1), "fold_best")
    copyfile(fnames[kmin], new_name)
    

    fname = "fsel_best_features.txt"
    selected_features = Features[:, kmin].nonzero()[0]
    selected_features = ['"{}"'.format(_attributenames[int(idx)]) for idx in selected_features]
    separators = [", "]*(len(selected_features) - 2) + [" and ", ""]
    fpath = os.path.join(odir_data, fname)
    with open(fpath, "w") as handle:
        handle.write(''.join(list(chain(*zip(selected_features, separators)))))

    fname = os.path.join(odir, "features_selected{}.png".format(fileid))
    plt.savefig(fname, bbox_inches='tight')
    # plt.savefig(fname)


    # Inspect selected feature coefficients effect on the entire dataset and
    # plot the fitted model residual error as function of each attribute to
    # inspect for systematic structure in the residual

    f = kmin  # cross-validation fold to inspect
    ff = Features[:, f].nonzero()[0]
    nfeatures = len(ff)
    if nfeatures == 0:
        print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        model = lm.LinearRegression(fit_intercept=True).fit(_X[:, ff], y)
        
        y_est = model.predict(_X[:, ff])
        residual = y - y_est

        coeffstrs = []
        for name, coef, transf in zip(np.array(_attributenames)[ff], model.coef_, np.array(tranformstrs)[ff]):
            coeffstrs.append('{:11} trf={}, coef={:+8.2e}'.format(name, transf, coef))
        coeffstrs.append('Intercept {}'.format(model.intercept_))
        coeffstrs.append('SSQ {}'.format(np.power(residual, 2).sum()))
        coeffstr = '\n'.join(coeffstrs)
        print(coeffstr)


        fig, axs = plt.subplots(1, nfeatures, figsize=(3*nfeatures, 3), sharey=True)
        # # fig.suptitle('Features selected in fold {0}'.format(f + 1), fontsize=fs_title)
        for ifeature, ax in enumerate(axs.flat):
            ax.plot(_X[:, ff[ifeature]], residual, '.', color='black')
            ax.set_xlabel(_attributenames[ff[ifeature]], fontsize=fs_axlabels)

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fs_ticks) 

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fs_ticks) 

            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect(abs(x1-x0)/abs(y1-y0))

        axs[0].set_ylabel('Residual error', fontsize=fs_axlabels)
        fig.tight_layout()
        fname = os.path.join(odir, "residuals_feature_selection{}.png".format(fileid))
        plt.savefig(fname, bbox_inches='tight')

        ex5.plotprediction(_X[:, ff], y, y_est, residual, coeffstr, 
                           yname, np.array(tranformstrs)[ff], ytransformstr,
                           np.array(_attributenames)[ff], odir, fileid, show=show)

    
    if show:
        plt.show()

