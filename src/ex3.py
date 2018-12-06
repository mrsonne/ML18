"""
Provide the basic summary statistics of your attributes preferable in a table and con-
sider if attributes are correlated, see also the functions numpy.cov() and numpy.corrcoef().
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as colors
import pandas as pd
MAXATTRLENGTH = 7

def corrcoef(X, y, classnames, attributenames):
    nattrs = X.shape[1]
    nclasses = len(classnames)
    nplots = nclasses + 1

    fig, axs = plt.subplots(nplots, 1, figsize=(5.5, 3.5*nplots), sharex=True)
    # fig.suptitle('Correlation matrix')

    corr = np.corrcoef(X.T)
    im = axs[0].imshow(corr, interpolation='none', aspect='equal',
                       cmap='bwr', norm=colors.Normalize(vmin=-1.,vmax=1.))
    idxplt = 0
    axs[idxplt].set_title('All')
    axs[idxplt].set_ylabel('Attributes')
    axs[idxplt].set_yticks(range(nattrs))
    axs[idxplt].set_yticklabels([a[:MAXATTRLENGTH] for a in attributenames], rotation=45)
    axs[idxplt].set_xticks([])

    correlations = {}
    for idxclass, classname in enumerate(classnames):
        idxplt = idxclass + 1
        class_mask = (y == idxclass) # binary mask to extract elements of class c        
        corr = np.corrcoef(X[class_mask, :].T)
        im = axs[idxplt].imshow(corr, interpolation='none', aspect='equal',
                                      cmap='bwr', norm=colors.Normalize(vmin=-1.,vmax=1.))
        axs[idxplt].set_title(classname)
        axs[idxplt].set_ylabel('Attributes')
        axs[idxplt].set_yticks(range(nattrs))
        axs[idxplt].set_yticklabels([a[:MAXATTRLENGTH] for a in attributenames], rotation=45)
        axs[idxplt].set_xticks([])
        correlations[classname] = corr

    # diff = correlations['Patient'] - correlations['Healthy control']
    # im = axs[0].imshow(diff, interpolation='none', aspect='equal',
    #                    cmap='bwr', norm=colors.Normalize(vmin=-1.,vmax=1.))
    # Leptin - resiti
    # Leptin - age

    # Axis for the colorbar
    cbaxes = fig.add_axes([0.875, 0.11, 0.03, 0.77])  
    cb = plt.colorbar(im, cax=cbaxes)

    axs[-1].set_xticks(range(nattrs))
    axs[-1].set_xticklabels([a[:MAXATTRLENGTH] for a in attributenames], rotation=45)
    axs[-1].set_xlabel('Attributes')
    
    # plt.tight_layout()
    fname = "../images/breast_cancer/correlation_matrix.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.show()



def describe(x, attributenames, odir=''):
    """
    Summary statistics
    """
    df = pd.DataFrame(x, columns=attributenames)
    # print('\nData')
    print(df)

    print('\nDescriptives')
    df_describe = df.describe()
    print(df_describe.round(2))

    fname = "data_describe.txt"
    fpath = os.path.join(odir, fname)
    with open(fpath, "w") as handle:
        handle.write(df_describe.round(2).to_latex())
