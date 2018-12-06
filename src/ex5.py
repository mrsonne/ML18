"""
Create two sepearate sections in your report, one on regression and one on classifi-
cation and explain in each section which regression and classication problem you
intent to solve. If your data does not naturally form a classication problem you can
consider thresholding one of the attributes to turn it into two classes. If your data
does not naturally form a regression problem you can (as we did for the wine data
in today's exercise) consider predicting one of the attributes from the remaining
attributes using linear regression.
"""
from __future__ import print_function
import numpy as np
from sklearn import tree
import sklearn.linear_model as lm
import graphviz
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib
from pandas_ml import ConfusionMatrix
from itertools import product
import matplotlib.patheffects as path_effects
import matplotlib as mpl 
import os

def pow2(x):
    return np.power(x, 2)


def pow3(x):
    return np.power(x, 3)


def pow32(x):
    return np.power(x, 1.5)


def log(x):
    return np.log(x)


def sqrt(x):
    return np.sqrt(x)



def render_tree(dtc, attributenames, odir='../', fname='tree'):
    out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributenames)
    src = graphviz.Source.from_file('tree_gini.gvz')
    fpath = os.path.join(odir, fname)
    print('Tree rendered in {}'.format(fpath))
    src.format = "png"
    src.render(fpath, view=True)


def tree_fit(X, y, attributenames, criterion='gini', show=False):
    """
    Modified from Exercise 5.1.2

    The parameter criterion can be used to choose the splitting 
    criterion (gini or entropy).

    The parameters min_samples_split, min_samples_leaf, and max_depth 
    influence the stopping criterion.
    """
    # Fit regression tree classifier, Gini split criterion, no pruning
    dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=15)
    dtc = dtc.fit(X, y)

    if show:
        render_tree(dtc, attributenames)

    return dtc


def tree_predict(x, dtc, classnames, attributenames):
    """
    Modified from Exercise 5.1.4
    """
    # Data array
    _x = x.reshape(1, -1)

    # Evaluate the classification tree for the new data object
    idclass = dtc.predict(_x)[0]
    proba = dtc.predict_proba(_x)
    
    # Print results
    print('\nAttributes:')
    for name, val in zip(attributenames, _x[0]):
        print('  {:15}{}'.format(name, val))

    print('\nClassification result:')
    print('  Class: {}'.format(classnames[idclass]))
    print('  p: {}'.format(proba[0][idclass]))


def prepare_data(X, attributenames, yname, transforms):
    """
    Transform X and extract y-column based on yname
    """
    # Transform features
    Xtranformed = X.copy()
    if transforms is not None:
        for name, fun in transforms.items():
            idx = attributenames.index(name)
            newvals = fun(X[:,idx])
            Xtranformed[:,idx] = newvals

    # Split dataset into features and target vector
    y_idx = attributenames.index(yname)
    _attributenames = list(attributenames)
    _attributenames.remove(yname)
    y = Xtranformed[:, y_idx]

    if transforms is not None:
        tkeys = transforms.keys()
        tranformstrs = [transforms[name].__name__ if name in tkeys else None for name in _attributenames]
        ytransformstr = transforms[yname].__name__ if yname in tkeys else None
    else:
        tranformstrs = [None]*len(_attributenames)
        ytransformstr = None

    X_cols = list(range(0, y_idx)) + list(range(y_idx + 1,len(attributenames)))
    _X = Xtranformed[:, X_cols]

    return _X, y, tranformstrs, ytransformstr, _attributenames


def linreg_fit(X, attributenames, yname, transforms=None, plot=True, odir='./', fileid=''):
    """
    Modified from Exercise 5.2.2 and 5.2.5
    """

    _X, y, tranformstrs, ytransformstr, _attributenames = prepare_data(X, attributenames, yname, transforms)

    print('XXXXX', ytransformstr)

    model = lm.LinearRegression(fit_intercept=True).fit(_X, y)
    # Compute model output:
    y_est = model.predict(_X)
    residual = y_est - y
    coeffstrs = []
    for name, coef, transf in zip(_attributenames, model.coef_, tranformstrs):
        coeffstrs.append('{:11} trf={}, coef={:+8.2e}'.format(name, transf, coef))
    coeffstrs.append('Intercept {}'.format(model.intercept_))
    coeffstrs.append('SSQ {}'.format(np.power(residual, 2).sum()))
    coeffstr = '\n'.join(coeffstrs)
    print(coeffstr)
    if plot:
        plotprediction(_X, y, y_est, residual, coeffstr, yname, tranformstrs, ytransformstr, _attributenames, odir, fileid)


def plotprediction(_X, y, y_est, residual, coeffstr, yname, tranformstrs, ytransformstr, _attributenames, odir, fileid, show=True):
    fs_label = 18
    fs_ticks = 15

    ymin = min(np.min(y), np.min(y_est))
    ymax = max(np.max(y), np.max(y_est))
    fig = plt.figure(figsize=(12, 5.))
    gs = grd.GridSpec(1, 2, height_ratios=[1,], width_ratios=[1, 1], wspace=0.3)
    axs = [fig.add_subplot(g) for g in gs]
    axs[0].plot(y, y_est, 'k.')
    axs[0].plot([ymin, ymax], [ymin, ymax], 'k-')
    axs[0].set_xlim((ymin,ymax))
    axs[0].set_ylim((ymin,ymax))
    x0, x1 = axs[0].get_xlim()
    y0, y1 = axs[0].get_ylim()
    axs[0].set_aspect(abs(x1-x0)/abs(y1-y0))

    if ytransformstr is None:
        axs[0].set_xlabel('True {}'.format(yname), fontsize=fs_label)
        axs[0].set_ylabel('Estimated {}'.format(yname), fontsize=fs_label)
    else:
        axs[0].set_xlabel('True {}({})'.format(ytransformstr, yname), fontsize=fs_label)
        axs[0].set_ylabel('Estimated {}({})'.format(ytransformstr, yname), fontsize=fs_label)

    for tick in axs[0].xaxis.get_major_ticks():
        tick.label.set_fontsize(fs_ticks) 

    for tick in axs[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(fs_ticks) 

    mpl.rcParams['font.family'] = ['monospace']
    t = axs[0].text(0.01, 0.99, coeffstr,
                    transform=axs[0].transAxes,
                    fontsize=8.5,
                    horizontalalignment='left', 
                    verticalalignment='top',
                    color='black')

    # t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

    mpl.rcParams.update(mpl.rcParamsDefault)

    axs[1].hist(residual, 40, fill=False, color='black')
    for tick in axs[1].xaxis.get_major_ticks():
        tick.label.set_fontsize(fs_ticks) 

    for tick in axs[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(fs_ticks) 

    x0, x1 = axs[1].get_xlim()
    y0, y1 = axs[1].get_ylim()
    axs[1].set_xticks( np.linspace(int(x0 - 1), int(x1+1), 4))
    x0, x1 = axs[1].get_xlim()
    axs[1].set_aspect(abs(x1-x0)/abs(y1-y0))

    if ytransformstr is None:
        axs[1].set_xlabel('Estimated {1} - True {1}'.format(ytransformstr, yname), fontsize=fs_label)
    else:
        axs[1].set_xlabel('Estimated {0}({1}) - True {0}({1})'.format(ytransformstr, yname), fontsize=fs_label)
    axs[1].set_ylabel('Count', fontsize=fs_label)



    fname = os.path.join(odir, "lin_reg{}.png".format(fileid))
    plt.savefig(fname, bbox_inches='tight')

    if show:
        plt.show()


    nattrs = len(_attributenames)
    u = int(np.floor(np.sqrt(nattrs)))
    v = int(np.ceil(float(nattrs)/u))
    fig, axs = plt.subplots(u, v, figsize=(9, 4.5), sharey=True)
    axs = np.atleast_2d(axs)

    for idxattr, (ax, name, tranformstr) in enumerate(zip(axs.flat, _attributenames, tranformstrs)):
        ax.plot(_X[:, idxattr], residual, '.k')

        if tranformstr is not None:
            ax.set_xlabel('{}({})'.format(tranformstr, name), fontsize=fs_label)
        else:
            ax.set_xlabel('{}'.format(name), fontsize=fs_label)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fs_ticks) 

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs_ticks) 


        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))

    for ax in axs[:,0]:
        ax.set_ylabel('Residual', fontsize=fs_label)

    
    fig.tight_layout()
    fname = os.path.join(odir, "lin_reg_attribute_residuals{}.png".format(fileid))
    plt.savefig(fname, bbox_inches='tight')

    if show:
        plt.show()


def logreg_fit(X, y, classnames, attributenames, plot=True, odir='./', fileid=''):
    """
    Modified from Exercise 5.2.6

    Fit using logistic regression
    """
    model = lm.logistic.LogisticRegression()
    model = model.fit(X, y)
    idxref = 0

    # Classify wine as helthy/patient (0/1) and assess probabilities
    y_est = model.predict(X)
    y_est_clsref = model.predict_proba(X)[:, idxref] 
    print(model.coef_)
    for i, attr in enumerate(attributenames):
        print('{:20}: coef={:+8.2e} exp(coeff)={:8.2e}'.format(attr, model.coef_[0][i], np.exp(model.coef_[0][i])))


    # Evaluate classifier's misclassification rate over entire training data
    misclass_rate = sum(np.abs(y_est - y)) / float(len(y_est))

    # misclassified observations
    obs_misclassified = np.nonzero(y_est - y)[0]

    # Display classification results
    # print('\nProbability of given sample being a white wine: {0:.4f}'.format(x_class))
    str_miscls = 'Overall misclassification rate: {0:.3f}'.format(misclass_rate)
    print(str_miscls)

    confusion_matrix = ConfusionMatrix(y, y_est)
    print("Confusion matrix:\n{}".format(confusion_matrix))

    df = confusion_matrix.to_dataframe(normalized=False, calc_sum=True)
    print(df[False][False])
    print(df[False])



    if plot:
        fs_ticks = 16
        fs_labels = 18
        fig, axs = plt.subplots(1,2, figsize=(11,6))
        plt.bar(attributenames, model.coef_[0], fill=False, color='black')
        plt.axhline(0, color='black')
        axs[1].set_xticklabels(attributenames, rotation=60, ha='right', fontsize=fs_ticks)
        axs[1].set_ylabel('Coefficient', fontsize=fs_labels)

        for tick in axs[1].yaxis.get_major_ticks():
            tick.label.set_fontsize(fs_ticks) 


        # fname = os.path.join(odir, 'coef{}'.format(fileid))
        # plt.savefig(fname, bbox_inches='tight')
        # plt.show()


        # fig, ax = plt.subplots(1,1)

        for idx, name in enumerate(classnames):
            class_ids = np.nonzero(y == idx)[0].tolist()
            axs[0].plot(class_ids, y_est_clsref[class_ids], 'o', markersize=8, label=name)

        axs[0].plot(obs_misclassified, y_est_clsref[obs_misclassified], 'ko', markersize=13, fillstyle='none', label='Misclassified')
        

        # axs[0].set_title(str_miscls)
        axs[0].axhline(0.5, color='black')
        axs[0].set_xlabel('Data object index', fontsize=fs_labels)
        axs[0].set_ylabel('Predicted prob. of "{}"'.format(classnames[idxref]), fontsize=fs_labels)

        for tick in axs[0].xaxis.get_major_ticks():
            tick.label.set_fontsize(fs_ticks) 

        for tick in axs[0].yaxis.get_major_ticks():
            tick.label.set_fontsize(fs_ticks) 

        # ax.set_ylim((0, 1))
        leg = axs[0].legend(fontsize=fs_labels)
        leg.get_frame().set_alpha(0.6)
        fig.tight_layout()

        fname = os.path.join(odir, 'prob_class{}'.format(fileid))
        plt.savefig(fname, bbox_inches='tight')
        plt.show()
        
        colornormalize = mpl.colors.Normalize(vmin=0, vmax=1)
        axconfusion = confusion_matrix.plot(normalized=True, colornormalize=colornormalize)

        pred = [item.get_text() for item in axconfusion.get_xticklabels()]
        actual = [item.get_text() for item in axconfusion.get_yticklabels()]
        mapstr = {'False': False, 'True': True}
        for p, a in product(pred, actual):
            colval = mapstr[p]
            rowval = mapstr[a]
            axconfusion.annotate(df[colval][rowval],
                                xy=(int(colval), int(rowval)), xycoords='data',
                                xytext=(0, 0), textcoords='offset points',
                                path_effects=[
                                              path_effects.PathPatchEffect(edgecolor='white', linewidth=1., facecolor='0.25')],
                                fontsize=24,
                                horizontalalignment='center', verticalalignment='center')

        plt.show()

    return obs_misclassified