import os


# Get absolute paths of the following directories

# project directory
def get_proj_dir():
    return os.path.dirname(os.path.abspath(__file__))


# models directory
def get_model_dir():
    return os.path.join(get_proj_dir(), 'models')


# data directory
def get_data_dir():
    return os.path.join(get_proj_dir(), 'data')


# training data directory
def get_training_dir():
    return os.path.join(get_data_dir(), 'TrainingData')


# testing data directory
def get_testing_dir():
    return os.path.join(get_data_dir(), 'TestData')