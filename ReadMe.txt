Brief Description of this project and its parts.

Overall: This repository is focused on gait detection and detecting anomolies in a persons walking.
To this end we are using a VAE to encode features from 3 different sensors and then use an hmm that is fed these
features to classify the gait into 4 different classes.


/data: This folder holds all of the training and testing data that is used in the project

/data/TrainingData: This is where all the training data that is used to train the VAE and calculate probabilities
for the HMM is stored.

/data/TestData: This is where all of the test data is stored along with our predictions that the HMM has made for the
test data.

/models: This is where all trained VAE models are stored so that they can be used for the encoding process
in the classification

/pictures: This is just a directory where images can be saved from the visualization file. This folder has
no critical use its just a convienent place to store wanted images.

.gitignore: ignore files not needed/ burdensome for the repo. aka pictures and data files

data_loaders.py: The data loaders are what take our training/testing data from their csv files and then allow them
to be easily used for training models, or evaluating metrics

hmm.py: This is the file where the hmm classifier is built. This file is also where we run our evaluations of
it and run it on our training and testing data.

main.py: This is the file that handles the training of the VAE models

pathing.py: This file contains functions that return the absolute paths of the different directories of the project
regardless of user or operating system.

requirements.txt: Most required libraries used in the project. navigate to project directory in terminal and input
"pip install -r requirements.txt" to install them. Navigate to https://pytorch.org/get-started/locally/ to
install correct version of pytorch

VAE.py: This file contains the different models of VAE used in the project.

visualization.py: This file contains functions to view outputs from the VAE as well as other visualizations such as
VAE sampling and span.