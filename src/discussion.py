from __future__ import print_function
import numpy as np
import ex7

# sklearn definition (and my confusion matrix plot)
#       -   +
# H    TN  FP
# S    FN  TP

# wiki
#       S   H
# +    TP  FP
# -    FN  TN

# http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Probability/BS704_Probability6.html


def get_data_synth():
    """
    https://en.wikipedia.org/wiki/False_positive_paradox
    """
    confusion_matrix_A = np.array([[400., 30.],[0., 570.]])

    # transfor to sklearn convension
    confusion_matrix_A = np.fliplr(np.rot90(confusion_matrix_A))
    n = 1000
    p_S = 0.02
    return confusion_matrix_A, p_S, n


def get_data_breastcancer():
    """
    """
    # confusion_matrix_A = np.array([[50., 10.],[10., 50.]])
    confusion_matrix_A = None
    n = 10000
    p_S = 0.001
    return confusion_matrix_A, p_S, n


def fit(x, y, attributenames, classnames, odir='./', odir_data='./', fileid=''):
    """
    Fit decision trre and manually tranfer data to "get_data_breastcancer" :-o
    I was out of time
    """
    confmat = ex7.classification(x, y, attributenames, classnames, model='tree_maxdepth3',
                                    odir=odir, odir_data=odir_data, fileid=fileid)
    return confmat


def probabilities(confusion_matrix_A, p_S, n=1):
    """
    A is the population used to train the model
    B i the wild population
    S is Sick
    H is Healthy
    """
    print(confusion_matrix_A)
    confusion_matrix_A = confusion_matrix_A.astype('float')

    # pop A
    count_H, count_S = np.sum(confusion_matrix_A, axis=1)
    # p_given_state = confusion_matrix_A/counts_S_H
    p_pos_given_H = confusion_matrix_A[0, 1] / count_H
    p_neg_given_H = confusion_matrix_A[0, 0] / count_H
    p_neg_given_S = confusion_matrix_A[1, 0] / count_S
    p_pos_given_S = confusion_matrix_A[1, 1] / count_S

    print("Population A")
    print('P(+ | H) = {:7.1e}'.format(p_pos_given_H))
    print('P(- | H) = {:7.1e}'.format(p_neg_given_H))
    print('P(+ | S) = {:7.1e}'.format(p_pos_given_S))
    print('P(- | S) = {:7.1e}'.format(p_neg_given_S))

    # pop B
    p_H =  1. - p_S 
    n_S = n*p_S
    confusion_matrix_B = np.empty_like(confusion_matrix_A)

    print("Population B")

    # Products rule (Lecture_4-5_probability_Theory.pdf p 28)
    # positive & sick (TP)
    r = p_S*p_pos_given_S
    confusion_matrix_B[1, 1] = r
    print('n true pos {} of {} (p={})'.format(r*n, n, r))

    # negative & sick (FN)
    r = p_S*p_neg_given_S
    confusion_matrix_B[1, 0] = r
    print('n false neg {} of {} (p={})'.format(r*n, n, r))

    # positive & healty (FP)
    r = p_H*p_pos_given_H
    confusion_matrix_B[0, 1] = r
    print('n false pos {} of {} (p={})'.format(r*n, n, r))

    # negative & healthy (TN)
    r = p_H*p_neg_given_H
    confusion_matrix_B[0, 0] = r
    print('n true neg {} of {} (p={})'.format(r*n, n, r))


    ntot = confusion_matrix_B.sum()
    # unconditional positive
    p_pos = confusion_matrix_B[:, 1].sum()/ntot
    p_neg = 1. - p_pos

    # sick given positive 
    p = p_pos_given_S*p_S/p_pos
    print('P(S | +) = {:7.1e}'.format(p))

    # healthy given positive
    p = p_pos_given_H*p_H/p_pos
    print('P(H | +) = {:7.1e}'.format(p))

    # sick given negative
    p = p_neg_given_S*p_S/p_neg
    print('P(S | -) = {:7.1e}'.format(p))

    # healthy given negative
    p = p_neg_given_H*p_H/p_neg
    print('P(H | -) = {:7.1e}'.format(p))
