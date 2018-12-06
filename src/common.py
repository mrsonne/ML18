from __future__ import print_function
import os
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patheffects as path_effects
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import abc
from sklearn import model_selection
import pandas as pd
from itertools import combinations
from scipy import stats

pd.options.display.float_format = '{:,.2f}'.format

def paired_ttest(vals1, vals2, alpha=0.05):
    z = vals1 - vals2
    n = len(z)
    zb = z.mean()
    nu = n - 1
    sig =  (z - zb).std()  / np.sqrt(nu)
    zL = zb + sig * stats.t.ppf(alpha*0.5, nu)
    zH = zb + sig * stats.t.ppf(1-alpha*0.5, nu)

    pval = pvalue(vals1, vals2)

    if zL <= 0 and zH >= 0:
        different = False
    else:
        different = True

    return pval, different

def pvalue(vals1, vals2):
    tstatistic, pvalue = stats.ttest_rel(vals1, vals2)#, equal_var=False)
    return pvalue


class Splitter(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def split(self):
        """
        Should return X_train, X_test, y_train, y_test
        """
        pass


class HoldoutSplitter(Splitter):

    def __init__(self, X, y, test_proportion=0.5):
        self.K = 1
        self.X = X
        self.y = y
        self.ntot = len(self.y)
        self.test_proportion = test_proportion


    def set_data(self, X, y):
        self.X = X
        self.y = y
        self.ntot = len(self.y)


    def split(self):
        X_train, X_test, y_train, y_test, = model_selection.train_test_split(self.X,
                                                                             self.y,
                                                                             test_size=self.test_proportion)
        return X_train, X_test, y_train, y_test, 1.


class KfoldSplitter(Splitter):
    def __init__(self, X, y, K):
        print('Initializing splitter')
        self.K = K
        self.CV = model_selection.KFold(n_splits=self.K, shuffle=True)

        self.X = X
        self.y = y
        self.ntot = len(self.y)
        self.cvsplit = self.CV.split(X)


    def set_data(self, X, y):
        self.X = X
        self.y = y
        self.cvsplit = self.CV.split(X)
        self.ntot = len(self.y)


    def split(self):
        train_index, test_index = next(self.cvsplit) 
        X_train, y_train = self.X[train_index,:], self.y[train_index]
        X_test, y_test = self.X[test_index,:], self.y[test_index]
        weight = float(len(y_test))/self.ntot
        return X_train, X_test, y_train, y_test, weight


def get_empty_kfold_splitters(nsplitters, k_inner):
    """
    Splitters contain no data since this will be set in outer cross validation loop
    """
    return [common.KfoldSplitter([], [], k_inner) for _ in range(nsplitters)]


class NestedCrossValidator(object):
    """
    TODO: generalize and merge with CrossValidator
    """
    def __init__(self, splitter, xvalidators, name='Outer cross validation'):
        print('Initializing {}'.format(name))
        self.xvalidators = xvalidators
        self.nvals = len(self.xvalidators)
        self.itervals = iter(xvalidators)
        self.name = name
        self.splitter = splitter

        self.error_train = None
        self.error_test = None
        self.error_test_mean = None
        self.error_train_mean = None

        # results dataframes
        self.df = None
        self.df_describe = None
        self.df_pval = None



    def set_itervals(self):
        """
        For (re)settimng iterator
        """
        self.itervals = iter(self.xvalidators)


    def get_currentmodel(self):
        """
        Choose the model attribute of the next model
        """
        return next(self.itervals)


    def xval_error(self, model, X_test, y_test, X_train, y_train):
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        misclass_rate_test = sum(np.abs(y_pred_test - y_test)) / float(len(y_pred_test))
        misclass_rate_train = sum(np.abs(y_pred_train - y_train)) / float(len(y_pred_train))

        objs_misclassified = np.nonzero(y_pred_test - y_test)[0]
        confmat = confusion_matrix(y_test, y_pred_test)
        return misclass_rate_test, misclass_rate_train, objs_misclassified, confmat, y_pred_test


    def run(self):
        """
        """
        print('\nStart "{}"'.format(self.name))
        print('with classification models {} for each fold.'.format([xval.name for xval in self.xvalidators]))
        # Initialize variables
        self.error_train = np.atleast_2d(np.empty((self.nvals, self.splitter.K)))
        self.error_test = np.atleast_2d(np.empty((self.nvals, self.splitter.K)))
        # self.best_values = np.atleast_2d(np.empty((self.nvals, self.splitter.K)))
        # nested list can store mixed types
        self.best_values = [[None for k in range(self.splitter.K)] for i in range(self.nvals)]
        weights = np.empty(self.splitter.K)

        for k in range(self.splitter.K):
            print('Fold {} of {} in "{}"'.format(k + 1, self.splitter.K, self.name))

            # train = D_par, test = D_test
            X_train, X_test, y_train, y_test, weights[k] = self.splitter.split()

            self.set_itervals()
            for i in range(self.nvals):
                xval = self.get_currentmodel()
                # set data on splitter for current (outer) fold-loop
                xval.set_splitter_data(X_train, y_train)

                # get the best model (fitted to D_par) and the corresponding model attribute 
                bestmodel, best_value = xval.run()

                (self.error_test[i, k],
                 self.error_train[i, k],
                 _, _, _) = self.xval_error(bestmodel, X_test, y_test, X_train, y_train)

                # self.best_values[i, k] = best_value
                self.best_values[i][k] = best_value



        # https://pandas.pydata.org/pandas-docs/stable/advanced.html
        ncol = 2
        headers = [np.array([[xval.name]*ncol for xval in self.xvalidators]).flatten().tolist(), 
                   np.array([['{}'.format(xval.attr_name), 'Test error'] for xval in self.xvalidators]).flatten().tolist()]
        tuples = list(zip(*headers))
        index = pd.MultiIndex.from_tuples(tuples, names=['Model', 'Property'])

        # table_values = [[None for k in range(self.splitter.K)] for i in range(self.nvals*ncol)]
        table_values = [None for i in range(self.nvals*ncol)]
        for i in range(self.nvals):
            irow = 2*i
            table_values[irow] = self.best_values[i]
            table_values[irow + 1] = self.error_test[i, :].tolist()

        # transpose
        table_values = [list(x) for x in zip(*table_values)]
        self.df = pd.DataFrame(table_values, columns=index)

        print('\nResults for outer cross validation folds')
        print(self.df.T)

        print('\nDescriptives for objects outer cross validation folds')
        self.df_describe = self.df.describe(include=[np.number]).T
        print(self.df_describe)

        if self.splitter.K > 1:
            # Indices for model combinations
            alpha = 0.05
            pvals = np.empty((self.nvals, self.nvals)) + np.nan
            for idxmod1, idxmod2 in combinations(range(self.nvals), 2):
                pvals[idxmod1, idxmod2], _ = paired_ttest(self.error_test[idxmod1, :], 
                                                          self.error_test[idxmod2, :], alpha=alpha)

                # symmetry
                pvals[idxmod2, idxmod1] = pvals[idxmod1, idxmod2]

            print('\np-values from paired t-tests comparing model generalization errors.')
            names = [xval.name for xval in self.xvalidators]
            self.df_pval = pd.DataFrame(pvals, columns=names, index=names)
            print(self.df_pval)

        # Error averaged over folds 
        # self.error_test_mean = self.error_test.mean(axis=1)
        # self.error_train_mean = self.error_train.mean(axis=1)
        self.error_test_mean = np.average(self.error_test, axis=1, weights=weights)
        self.error_train_mean = np.average(self.error_train, axis=1, weights=weights)
                
        # Index corresponding to best model 
        idxmin_model = self.error_test_mean.argmin()

        # Best value is actually a cross validator
        best_xval = self.xvalidators[idxmin_model]
        best_model = best_xval.basemodel

        # For best model type find best fold 
        idxmin_fold = self.error_test[idxmin_model, :].argmin()
        best_value = self.best_values[idxmin_model][idxmin_fold]

        self.err_gen = self.error_test_mean[idxmin_model]
        self.best_model_name = best_xval.name
        self.best_attr_name = best_xval.attr_name
        self.best_attr_val = best_value
        self.best_fold = idxmin_fold + 1
        self.err_min_for_best_model = self.error_test[idxmin_model, idxmin_fold]
        fstr = '\nMinimal estimated generalization error {} (weighted fold average) for {}.'
        print(fstr.format(self.err_gen,
                          self.best_model_name))

        fstr = 'Best model complexity is "{}" = {} (from fold {} with test error {})'
        print(fstr.format(self.best_attr_name,
                          self.best_attr_val,
                          self.best_fold,
                          self.err_min_for_best_model))

        self.df = self.df.drop('strategy', axis=1, level=1)
        print(self.df.T)


    @staticmethod
    def write_result(odir, fname, value):
        with open(os.path.join(odir, fname), "w") as handle:
            handle.write(str(value))


    def write_tables(self, odir="./", fileid=''):

        if self.df_describe is not None:
            fname = "xval_describe{}.txt".format(fileid)
            fpath = os.path.join(odir, fname)
            with open(fpath, "w") as handle:
                handle.write(self.df_describe.round(3).to_latex())

        if self.df is not None:
            fname = "xval_summary{}.txt".format(fileid)
            fpath = os.path.join(odir, fname)
            with open(fpath, "w") as handle:
                handle.write(self.df.T.to_latex())


        if self.df_pval is not None:
            fname = "xval_pvalues{}.txt".format(fileid)
            fpath = os.path.join(odir, fname)
            with open(fpath, "w") as handle:
                handle.write(self.df_pval.to_latex())

        fname = "xval_err_gen{}.txt".format(fileid)
        NestedCrossValidator.write_result(odir, fname, '{:5.2e}'.format(self.err_gen))
        fname = "xval_best_model{}.txt".format(fileid)
        NestedCrossValidator.write_result(odir, fname, self.best_model_name)
        fname = "xval_best_attr{}.txt".format(fileid)
        NestedCrossValidator.write_result(odir, fname, '\\code{{{}}}'.format(self.best_attr_name))
        fname = "xval_best_attrval{}.txt".format(fileid)
        NestedCrossValidator.write_result(odir, fname, self.best_attr_val)
        fname = "xval_best_fold{}.txt".format(fileid)
        NestedCrossValidator.write_result(odir, fname, self.best_fold)
        fname = "xval_best_modelerr{}.txt".format(fileid)
        NestedCrossValidator.write_result(odir, fname, '{:5.2e}'.format(self.err_min_for_best_model))

        
        



    def plot(self, odir, fileid):

        fs_ticks = 12
        fs_axlabels = 14
        fs_legend = 14
        fs_title = 16

        rotation = 60
        xlabelstr = 'Classification models'
        vals = [xval.name for xval in self.xvalidators]
        if self.splitter.K > 1:
            fig, axs = plt.subplots(1, 2, sharey=True)
        else:
            fig, axs = plt.subplots(1, 1)

        axs = np.atleast_1d(axs)

        axs[0].plot(vals, self.error_train_mean, 'k-o', label='Train')
        axs[0].plot(vals, self.error_test_mean, 'k--o', label='Test')
        axs[0].set_xticklabels(vals, rotation=rotation, ha='right', fontsize=fs_ticks)
        for tick in axs[0].yaxis.get_major_ticks():
            tick.label.set_fontsize(fs_ticks) 


        # axs[0].set_xlabel(xlabelstr)
        axs[0].set_ylabel('Error (misclassification rate)', fontsize=fs_axlabels)
        # axs[0].set_title(self.name, fontsize=fs_title)
        axs[0].legend(fontsize=fs_legend)


        if self.splitter.K > 1:
            axs[1].boxplot(self.error_test.T)
            axs[1].set_xticklabels(vals, rotation=rotation, ha='right', fontsize=fs_ticks) # right align labels to keep them at the tick mark
            # axs[1].set_xlabel(xlabelstr)
            axs[1].set_ylabel('Test error across CV folds, (K={0})'.format(self.splitter.K), fontsize=fs_axlabels)
            # axs[1].set_title(self.name, fontsize=fs_title)
            for tick in axs[1].yaxis.get_major_ticks():
                tick.label.set_fontsize(fs_ticks) 


            # # Indices for model combinations
            # alpha = 0.05
            # pvals = np.empty((self.nvals, self.nvals)) + np.nan
            # for idxmod1, idxmod2 in combinations(range(self.nvals), 2):
            #     pvals[idxmod1, idxmod2], _ = paired_ttest(self.error_test[idxmod1, :], 
            #                                               self.error_test[idxmod2, :], alpha=alpha)

            #     # symmetry
            #     pvals[idxmod2, idxmod1] = pvals[idxmod1, idxmod2]

            # print('\np-values from paired t-tests comparing model generalization errors.')
            # names = [xval.name for xval in self.xvalidators]
            # self.df_pval = pd.DataFrame(pvals, columns=names, index=names)
            # print(df_pval)
            # print(df_pval.to_latex())

        else:
            print('Skipping boxplot and paired t-test for K=1')

        plt.tight_layout()
        fname = os.path.join(odir, "nestedcv{}.png".format(fileid))
        plt.savefig(fname, bbox_inches='tight')

        plt.show()    


class CrossValidator(object):
    """
    """
    def __init__(self, splitter, attr_name, attr_values, basemodel, name='Cross validation', attr_type=int):
        print('Initializing {}'.format(name))
        self.attr_name = attr_name
        self.attr_values = attr_values
        self.nvals = len(self.attr_values)
        self.itervals = iter(attr_values)
        self.name = name
        self.splitter = splitter
        self.basemodel = basemodel

        self.error_train = None
        self.error_test = None
        self.error_test_mean = None
        self.error_train_mean = None

        # for type-casting model parameter averaged over folds
        self.attr_type = attr_type

        self.df_describe = None
        self.df = None

    def set_itervals(self):
        """
        For (re)settimng iterator
        """
        self.itervals = iter(self.attr_values)

    def get_currentmodel(self):
        """
        Choose the model attribute of the next model
        """
        val = next(self.itervals)
        tmp = {self.attr_name : val}
        return self.basemodel.set_params(**tmp), val


    def fit(self, model, X_train, y_train):
        """
        """
        return model.fit(X_train, y_train)


    def xval_error(self, model, X_test, y_test, X_train, y_train):
        y_pred_test = model.predict(X_test)
        if X_train is not None:
            y_pred_train = model.predict(X_train)
            misclass_rate_train = sum(np.abs(y_pred_train - y_train)) / float(len(y_pred_train))
        else:
            misclass_rate_train = None

        misclass_rate_test = sum(np.abs(y_pred_test - y_test)) / float(len(y_pred_test))
        objs_misclassified = np.nonzero(y_pred_test - y_test)[0]
        confmat = confusion_matrix(y_test, y_pred_test)
        return misclass_rate_test, misclass_rate_train, objs_misclassified, confmat, y_pred_test


    def set_splitter_data(self, x, y):
        self.splitter.set_data(x, y)


    def run(self):
        """
        """
        print('\nStart cross validation "{}" using attribute "{}"'.format(self.name, self.attr_name))
        print('with values {} for each fold.'.format(self.attr_values))

        # Initialize variables
        self.error_train = np.atleast_2d(np.empty((self.nvals, self.splitter.K)))
        self.error_test = np.atleast_2d(np.empty((self.nvals, self.splitter.K)))
        weights = np.empty(self.splitter.K)

        for k in range(self.splitter.K):
            print('Fold {} of {}'.format(k + 1, self.splitter.K))

            # split data (D_par) in train = D_train, test = D_val
            X_train, X_test, y_train, y_test, weights[k] = self.splitter.split()

            self.set_itervals()
            for i in range(self.nvals):
                currentmodel, attr_value = self.get_currentmodel()
                bestmodel = self.fit(currentmodel, X_train, y_train)

                # print(currentmodel.get_params())

                (self.error_test[i, k], 
                 self.error_train[i, k],
                 _, _, _) = self.xval_error(bestmodel, X_test, y_test, X_train, y_train)

            # print('Fold {} of {}'.format(k + 1, self.splitter.K), zip(self.attr_values, self.error_test[:, k]))

        # Error averaged over folds
        self.error_test_mean = self.error_test.mean(axis=1)
        self.error_train_mean = self.error_train.mean(axis=1)
        self.error_test_mean = np.average(self.error_test, axis=1, weights=weights)
        self.error_train_mean = np.average(self.error_train, axis=1, weights=weights)


        print('Test error for each cross validation fold using model "{}" with attribute "{}"'.format(self.name, self.attr_name))
        headers = [self.attr_values]
        self.df = pd.DataFrame(self.error_test, index=headers)
        self.df.index.names = self.attr_name,
        print(self.df)

        self.df_describe = self.df.T.describe().T
        print('\nDescriptives in cross validation folds')
        print(self.df_describe)


        idxmin = self.error_test_mean.argmin()
        best_value = self.attr_values[idxmin]
        print('Minimal test error {} for {} = {}'.format(self.error_test_mean[idxmin],
                                                         self.attr_name,
                                                         best_value))


        self.best_complexity = best_value
        self.err_gen = self.error_test_mean[idxmin]

        # Set param to best value and fit to all data. 
        # Thus, the cross-validation error should be an upper limit to the true generalization error
        best_model = None
        tmp = {self.attr_name : best_value}
        best_model = self.basemodel.set_params(**tmp)

        # Train on all data (par)
        best_model = self.fit(best_model, self.splitter.X, self.splitter.y)

        return best_model, best_value


    def plot(self, odir):
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.plot(self.attr_values, self.error_train_mean, label='Train')
        ax.plot(self.attr_values, self.error_test_mean, label='Test')
        ax.set_xlabel('Model complexity ({})'.format(self.attr_name), fontsize=20)
        ax.set_ylabel('Error (misclassification rate)', fontsize=20)
        ax.set_title(self.name)
        ax.legend(fontsize=16)

        ax.set_xticks(self.attr_values)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14) 

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14) 


        fname = os.path.join(odir, "cv_error_{}.png".format(self.name.replace(' ', '_')))
        plt.savefig(fname, bbox_inches='tight')



        if self.splitter.K > 1:
            fig, ax = plt.subplots(1, 1, figsize=(6,6))
            ax.boxplot(self.error_test.T)
            ax.set_xlabel('Model complexity ({})'.format(self.attr_name), fontsize=20)
            ax.set_ylabel('Test error across {0} CV folds'.format(self.splitter.K), fontsize=20)
            ax.set_title(self.name)

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14) 


            fname = os.path.join(odir, "cv_boxplot_{}.png".format(self.name.replace(' ', '_')))
            plt.savefig(fname, bbox_inches='tight')
        else:
            print('Skipping boxplot for K=1')

        plt.show()    


    @staticmethod
    def write_result(odir, fname, value):
        with open(os.path.join(odir, fname), "w") as handle:
            handle.write(str(value))




    def write_tables(self, odir, fileid):

        if self.df_describe is not None:
            fname = "xval_describe_{}.txt".format(self.name.replace(' ', '_'))
            fpath = os.path.join(odir, fname)
            with open(fpath, "w") as handle:
                handle.write(self.df_describe.round(3).to_latex())

        if self.df is not None:
            fname = "xval_summary_{}.txt".format(self.name.replace(' ', '_'))
            fpath = os.path.join(odir, fname)
            with open(fpath, "w") as handle:
                handle.write(self.df.round(3).to_latex())

        fname = "xval_best_complexity{}.txt".format(fileid)
        CrossValidator.write_result(odir, fname, '{:5.2e}'.format(self.best_complexity))

        fname = "xval_err_gen{}.txt".format(fileid)
        CrossValidator.write_result(odir, fname, '{:5.2e}'.format(self.err_gen))


def annotate_confusionmatrix(ax, confusionmatrix):
    for indices, val in np.ndenumerate(confusionmatrix):
        ax.annotate(val,
                    xy=indices[::-1], xycoords='data',
                    xytext=(0, 0), textcoords='offset points',
                    path_effects=[path_effects.PathPatchEffect(edgecolor='white', linewidth=1., facecolor='0.25')],
                    fontsize=24,
                    horizontalalignment='center', verticalalignment='center')


def classification_plot(classnames, attributenames, X_train, X_test, y_train,
                        confmat, y_pred, objs_misclassified, title, show=True,
                        odir='./', odir_data='./', fileid=''):
    ncls = len(classnames)
    idx1 = 0
    idx2 = 1

    # Plot the training data points (color-coded) and test data points.
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    # choose qualitative color maps for max contrast
    ncol = 8
    # colors = cycle(cm.Dark2(np.linspace(0, 1, ncol)))
    for idxcls, name in enumerate(classnames):
        # c = next(colors)
        class_mask = (y_train == idxcls)

        if X_train is not None:
            axs[0].plot(X_train[class_mask, idx1], X_train[class_mask, idx2],
                    marker='.', linestyle='none',
                    label='Training {}'.format(name))

        # Plot the classfication results
        class_mask = (y_pred == idxcls)
        axs[0].plot(X_test[class_mask, idx1], X_test[class_mask, idx2],
                marker='o', linestyle='none', fillstyle='none', markersize=8,
                label='Predicted {}'.format(name))

    for tick in axs[0].xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 

    for tick in axs[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(18) 

    axs[0].set_xlabel(attributenames[idx1], fontsize=20)
    axs[0].set_ylabel(attributenames[idx2], fontsize=20)

    axs[0].plot(X_test[objs_misclassified, idx1], X_test[objs_misclassified, idx2],
                'kx', fillstyle='none', label='misclassified')

    # leg = ax.legend(loc='upper right',
    #             fancybox=True, ncol=1)
    # leg.get_frame().set_alpha(0.5)

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #             fancybox=True, shadow=True, ncol=3)

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                  fancybox=True, shadow=True, ncol=2, fontsize=14)

    # ax.set_title(title)

    fig.tight_layout()

    # fname = "../images/breast_cancer/classification_testsplit_{}.png".format(title.replace(' ', '_'))
    # plt.savefig(fname, bbox_inches='tight')


    # Compute and plot confusion matrix
    accuracy = 100*confmat.diagonal().sum()/confmat.sum()
    error_rate = 100-accuracy
    # fig, ax = plt.subplots(1)
    colornormalize = mpl.colors.Normalize(vmin=0, vmax=1)
    annotate_confusionmatrix(axs[1], confmat)
    _confmat = confmat.astype(float)/confmat.sum()
    axs[1].set_xticklabels(classnames)
    axs[1].set_yticklabels(classnames, rotation=90)
    axs[1].set_xticks(np.arange(-0.5, ncls, 1), minor=True)
    axs[1].set_yticks(np.arange(-0.5, ncls, 1), minor=True)
    axs[1].grid(which="minor", ls="-", lw=2)
    im = axs[1].imshow(_confmat, cmap='binary', interpolation='None', norm=colornormalize)

    # Axis for the colorbar
    # cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])  
    # cb = plt.colorbar(im, cax=cbaxes)
    cbaxes = fig.add_axes([0.575, 0.9, 0.4, 0.03])  
    cb = plt.colorbar(im, cax=cbaxes, orientation='horizontal')

    for tick in cbaxes.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 

    for tick in axs[1].xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 

    for tick in axs[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(18) 

    axs[1].set_xticks(range(ncls))
    axs[1].set_yticks(range(ncls))
    axs[1].set_xlabel('Predicted class', fontsize=20)
    axs[1].set_ylabel('Actual class', fontsize=20)
    axs[1].set_title('Confusion matrix ({} % error rate)'.format(error_rate), fontsize=16)

    fname = os.path.join(odir, "classification_testsplit_{}.png".format(fileid))
    plt.savefig(fname, bbox_inches='tight')


    if show:
        plt.show()