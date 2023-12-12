import os
import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt

def SVM_score(loss, normalize=False):

    score = np.array(loss)
    return score

def Combine(loss, normalize=False):

    rgb_score = np.array(loss[0])
    opt_score = np.array(loss[1])
    if(normalize):
        rgb_score -= rgb_score.min()
        rgb_score /= rgb_score.max() if rgb_score.max() != 0 else 1
        opt_score -= opt_score.min()
        opt_score /= opt_score.max() if opt_score.max() != 0 else 1
    else:
        rgb_score /= rgb_score.max()
        opt_score /= opt_score.max()
    
    score = rgb_score + opt_score 
    score /= 2
    return score

def RGB(loss, normalize=False):

    rgb_score = np.array(loss[0])
    if(normalize):
        rgb_score -= rgb_score.min()
        rgb_score /= rgb_score.max() if rgb_score.max() != 0 else 1
    else:
        rgb_score /= rgb_score.max()
    
    score = rgb_score
    return score

def Optical(loss, normalize=False):

    opt_score = np.array(loss[1])
    if(normalize):
        opt_score -= opt_score.min()
        opt_score /= opt_score.max() if opt_score.max() != 0 else 1
    else:
        opt_score /= opt_score.max()

    score = opt_score
    return score

def ORIGIN(loss, normalize=False):
    origin = np.array(loss)
    if(normalize):
        origin -= origin.min()
        origin /= origin.max() if origin.max() != 0 else 1

    score = origin
    return score

get_score_type = \
    {
        "svm":      SVM_score,
        "combine":  Combine,
        "rgb":      RGB,
        "optical":  Optical,
        "ORIGIN":   ORIGIN
    }

if __name__ == '__main__':
    #ComputeShanghaitechLossFileAUC('result.pkl')
    None
