import numpy as np
import pandas as pd
from hmmlearn import hmm
from data_loaders import GaitData
from visualization import load_model
from pathing import get_training_dir
import pdb
import torch
from torch.utils import data
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# for testing hmm implementation only
def getTestDf():
    nums = np.arange(20)
    nums = np.stack((nums, nums+1, nums+2, nums+3, nums, nums))
    dfNums = pd.DataFrame(data=nums)
    dfLabels = pd.DataFrame(np.array([0,1,2,3,0,0]), columns=['y'])
    df = pd.concat([dfNums,dfLabels], axis=1)
    return pd.concat([df,df])

# Return encoded data from VAE from X values
def getEncodedData(model_file, ds):
    device = torch.device("cpu")
    model = load_model(model_file, device)

    kwargs = {}

    dloader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, **kwargs)

    codedData = np.empty((len(ds), 20))

    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(dloader):
            data = data.to(device)
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)
            codedData[i] = z

    return codedData

# Return the dataset corresponding to a root filename
def loadDSfromFile(filename_root):
    num_samples = 4 * 40

    x_cols = ['x1','x2','x3','x4','x5','x6']

    x_df = pd.read_csv(filename_root + "__x.csv", dtype=np.float32, names=x_cols)
    xt_df = pd.read_csv(filename_root + "__x_time.csv", dtype=np.float32, names=['time'])
    yt_df = pd.read_csv(filename_root + "__y_time.csv", dtype=np.float32, names=['time'])

    x_combined_df = pd.concat([x_df, xt_df], axis=1)

    flattened_data = np.array([x_combined_df[x_cols][i:i + num_samples].to_numpy().flatten() for i in range(int(len(x_combined_df) - num_samples))])

    flattened_df = pd.DataFrame(data=flattened_data)
    
    flattened_t_df = pd.concat([flattened_df, xt_df[int(num_samples/2):-int(num_samples/2)].reset_index()], axis=1)

    merged_df = pd.merge_asof(yt_df, flattened_t_df, on='time')

    merged_df = merged_df.fillna(method='backfill')
    
    return torch.utils.data.TensorDataset(torch.tensor(merged_df.drop(['time','index'], axis=1).values),torch.tensor(merged_df.drop(['time','index'], axis=1).values))



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


# Returns a trained HMM model given the configuration matricies
def getModel(start_probs, trans_probs, obs_means, obs_vars):
    model = hmm.GaussianHMM(n_components=len(start_probs), covariance_type="diag")
    model.startprob_ = start_probs
    model.transmat_ = trans_probs
    model.means_ = obs_means
    # model.covars_ = obs_vars+0.001 # if you get var can't be negative add small number
    model.covars_ = obs_vars
    return model

model_name = 'big3_test2.pt' # Model to use to encode the data

# Load the training and test data for the HMM
training_dir = get_training_dir()
dataset = GaitData(dirpath=training_dir)
idx = list(range(len(dataset)))
train_data = data.Subset(dataset, idx[:int(len(dataset)*0.8)])
test_data = data.Subset(dataset, idx[int(len(dataset)*0.8):])

X = pd.DataFrame(data=getEncodedData(model_name, train_data))
y = pd.DataFrame(train_data.dataset.y[train_data.indices], columns=['y'])

labeledData = pd.concat([X,y], axis=1)

# Configure the HMM based on the training data
(start_probs, trans_probs, obs_means, obs_vars) =  learnHmmParams(labeledData)
model = getModel(start_probs, trans_probs, obs_means, obs_vars)

# Evaluate the hmm on the training data
pred_y = model.decode(labeledData.drop('y', axis=1).to_numpy())[1]
y = labeledData['y'].to_numpy()

print(classification_report(y, pred_y))

# Evaluate the hmm on the test data
X = pd.DataFrame(data=getEncodedData(model_name, test_data))
y = pd.DataFrame(test_data.dataset.y[test_data.indices], columns=['y'])

labeledData = pd.concat([X,y], axis=1)

pred_y = model.decode(labeledData.drop('y', axis=1).to_numpy())[1]
y = labeledData['y'].to_numpy()

print(classification_report(y, pred_y))


# Plot the ground truth next to the predicted labels
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(y[80:160])
ax[0].set_title('Ground Truth')
ax[1].plot(pred_y[80:160])
ax[1].set_title('Predicted Labels')
plt.show()

# Make predictions on Test data
trials = ['./data/TestData/subject_009_01', './data/TestData/subject_010_01', './data/TestData/subject_011_01', './data/TestData/subject_012_01']

for t in trials:
    test_data = loadDSfromFile(t)
    X = getEncodedData('300_epoch_basic.pt', test_data)
    pred_y = model.decode(X)[1]
    pred_y_df = pd.DataFrame(data=pred_y)
    pred_y_df.to_csv(t + '__y_prediction.csv', index=False, header=False)

print('success')
