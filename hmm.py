import numpy as np
import pandas as pd
from hmmlearn import hmm

# POTENTIAL IMPROVEMENTS
#   - Use MOG model for hmm instead of gaussian
#   - Use full or tied covariance matrix
#   - Improve data encoding (VAE)


# for testing hmm implementation only
def getTestDf():
    nums = np.arange(20)
    nums = np.stack((nums, nums+1, nums+2, nums+3, nums, nums))
    dfNums = pd.DataFrame(data=nums)
    dfLabels = pd.DataFrame(np.array([0,1,2,3,0,0]), columns=['y'])
    df = pd.concat([dfNums,dfLabels], axis=1)
    return pd.concat([df,df])


# FOR TRAINING
# Use trained VAE to get compressed data with labels
#   the df should be formatted so that each latent feature gets a column
#   and the corresponding y label should be in column 'y' (only manually named column)
def getCompressedDataWithLabels():
    pass


# FOR INFERENCE
# Same as above but return a dataframe with no 'y' column
# to make predictions at the same frequency as the ground truth,
#   we need to return a pandas dataframe with our compressed
#   data at the desired frequency (if you want y predictions at 10Hz, give X's at 10Hz)
def getCompressedData():
    pass


# Xs is a list of X (compressed experiment) ys is a list of y (labels for experiment)
def learnHmmParams(df):
    # Learn params using all data
        # startprob_
        # transmat_
        # means_
        # covars_

    df = df.copy()

    y_vals = df['y'].unique()

    df['y_next'] = df['y'].shift(-1)

    num_states = len(y_vals)
    num_feats = len(df.drop(['y','y_next'], axis=1).columns)

    start_probs = np.zeros(num_states)
    start_probs[0] = 1 # make the first state 0
    trans_probs = np.empty((num_states, num_states))
    obs_means = np.empty((num_states, num_feats))
    obs_vars = np.empty((num_states, num_feats)) # diagonal covar matrix


    for i, y_i in enumerate(y_vals):
        # Compute transition probs
        for j, y_j in enumerate(y_vals):
            num = ((df['y'] == y_i) & (df['y_next'] == y_j)).sum()
            denom = ((df['y'] == y_i) & (df['y_next'].notna())).sum()
            trans_probs[i,j] = num/denom

        # Compute observation params for state I
        dfStateI = df.loc[df['y'] == y_i].drop(['y','y_next'], axis=1)
        obs_means[i] = dfStateI.mean().to_numpy()
        obs_vars[i] = dfStateI.var().to_numpy()

    return (start_probs, trans_probs, obs_means, obs_vars)


def getModel(start_probs, trans_probs, obs_means, obs_vars):
    model = hmm.GaussianHMM(n_components=len(start_probs), covariance_type="diag")
    model.startprob_ = start_probs
    model.transmat_ = trans_probs
    model.means_ = obs_means
    # model.covars_ = obs_vars+0.001 # if you get var can't be negative add small number
    model.covars_ = obs_vars
    return model


labeledData = getTestDf()
# labeledData = getCompressedDataWithLabels()
(start_probs, trans_probs, obs_means, obs_vars) =  learnHmmParams(labeledData)
model = getModel(start_probs, trans_probs, obs_means, obs_vars)
pdb.set_trace()
print(model.decode(labeledData.drop('y', axis=1).to_numpy()))