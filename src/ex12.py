"""
Include a new section in the report entitled association mining. In this part of the
report you are to investigate if there are associations among your attributes based
on association mining. In order to do so you will need to make your data binary,
see also exercise 12. (For categoric variables you can use the one-out-of-K coding
format). You will need to save the binarized data into a text le that can be analyzed
by the Apriori algorithm.

* Run the Apriori algorithm on your data and nd frequent itemsets as well as
association rules with high condence.

* Try and interpret the association rules generated.
Finally, write a very short executive summary of all your key ndings analyzing the
data throughout this course, i.e. exercise 1-12.
"""
import os
import platform
from subprocess import Popen
import re
import time
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat
import numpy as np
from mltools.similarity import binarize2
from mltools.writeapriorifile import WriteAprioriFile

# only tested on windows
EXTENSIONS = {'linux':'', 'linux2':'', 'darwin':'MAC', 'windows':'.exe'}


def get_data_12_1_5(idir='./'):
    mat_data = loadmat(os.path.join(idir, 'wine.mat'))
    X = mat_data['X']
    y = mat_data['y'].squeeze()
    C = mat_data['C'][0,0]
    M = mat_data['M'][0,0]
    N = mat_data['N'][0,0]
    attributenames = [name[0][0] for name in mat_data['attributeNames']]
    classnames = [cls[0][0] for cls in mat_data['classNames']]

    # print(attributeNames)
    # print(len(attributeNames))
    print('Binarizing the features...')
    X, attributenames = binarize2(X, attributenames)
    # print(attributeNames)
    # print(len(attributeNames))
    return X, y, attributenames, classnames



def binarize_data(X, attributenames):
    print('Binarizing the features...')
    return binarize2(X, attributenames)


def rename_attributes(attributenames):
    """
    Rename attributes to shorten names. Specific to naming created  
    by "binarize2"
    """
    _attributenames = [attr.replace("50th-100th percentile", "Q2+") for attr in attributenames]
    _attributenames = [attr.replace("0th-50th percentile", "Q2-") for attr in _attributenames]
    return _attributenames


def append_onehot_classlabels(X, y, attributenames, classnames):
    """
    One-hot encode y and append columns to X. 
    Also append class names to attribute names
    https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    """
    integer_encoded = y.reshape(len(y), 1)
    y_onehot_encoded = OneHotEncoder(sparse=False).fit_transform(integer_encoded)
    return np.hstack((X, y_onehot_encoded)), tuple(list(attributenames) + list(classnames))


def write_apriorifile(X, attributenames, odir='./'):
    fname = os.path.join(odir, "AprioriFile.txt")
    print('Writing the apriori-file (this might take a little while!)')
    WriteAprioriFile(X, titles=attributenames, filename=fname)
    print('Wrote {}.'.format(fname))



def run_mining(apriorifilepath, bindir='./', odir='./'):
    """
    Modified from Exercise 12.1.6
    Default bindir assumes that the aprioring executable is in the path
    """
    filename = os.path.normpath(apriorifilepath) #'AprioriFile.txt')
    minSup = 25
    minConf = 55
    maxRule = 4

    try:
        ext = EXTENSIONS[platform.system().lower()]
    except KeyError as err:
        raise err

    binpath = os.path.normpath(os.path.join(bindir, 'apriori{}'.format(ext)))
    fstr = '{0} -f"," -s{1} -v"[Sup. %S]" {2} apriori_temp1.txt'
    cmd = fstr.format(binpath, minSup, filename)
    print(cmd)

    status1 = Popen(cmd, shell=True)
    status1.communicate()


    if status1.returncode != 0:
        print('An error occurred while calling apriori, a likely cause is that minSup was set to high such that no '
            'frequent itemsets were generated or spaces are included in the path to the apriori files.')
        exit()
    if minConf > 0:
        print('Mining for associations by the Apriori algorithm')
        fstr = '{0} -tr -f"," -o -n{1} -c{2} -s{3} -v"[Conf. %C,Sup. %S]" {4} apriori_temp2.txt'
        cmd = fstr.format(binpath, maxRule, minConf, minSup, filename )
        status2 = Popen(cmd, shell=True)
        status2.communicate()

        if status2.returncode != 0:
            print('An error occurred while calling apriori')
            exit()
    print('Apriori analysis done, extracting results')

    # Extract information from stored files apriori_temp1.txt and apriori_temp2.txt
    with open('apriori_temp1.txt', 'r') as stream:
        lines = stream.readlines()

    # Extract Frequent Itemsets
    FrequentItemsets = [''] * len(lines)
    sup = np.zeros((len(lines), 1))
    for i, line in enumerate(lines):
        FrequentItemsets[i] = line[0:-1]
        sup[i] = re.findall(' [-+]?\d*\.\d+|\d+]', line)[0][1:-1]
    os.remove('apriori_temp1.txt')

    # Read the file
    with open('apriori_temp2.txt', 'r') as stream:
        lines = stream.readlines()

    # Extract Association rules
    AssocRules = [''] * len(lines)
    conf = np.zeros((len(lines), 1))
    for i, line in enumerate(lines):
        AssocRules[i] = line[0:-1]
        conf[i] = re.findall(' [-+]?\d*\.\d+|\d+,', line)[0][1:-1]
    os.remove('apriori_temp2.txt')

    # sort (FrequentItemsets by support value, AssocRules by confidence value)
    AssocRulesSorted = [AssocRules[item] for item in np.argsort(conf, axis=0).ravel()]
    AssocRulesSorted.reverse()
    FrequentItemsetsSorted = [FrequentItemsets[item] for item in np.argsort(sup, axis=0).ravel()]
    FrequentItemsetsSorted.reverse()

    # Print the results
    time.sleep(.5)
    print('\n')
    print('RESULTS:\n')
    print('Frequent itemsets:')
    for i, item in enumerate(FrequentItemsetsSorted):
        print('Item: {0}'.format(item))
    print('\n')
    print('Association rules:')
    for i, item in enumerate(AssocRulesSorted):
        print('Rule: {0}'.format(item))

    
    print('\nAssociation rules indicating Patient:')
    for i, item in enumerate(AssocRulesSorted):
        if "Patient" == item[:7]:
            print('Rule: {0}'.format(item))
