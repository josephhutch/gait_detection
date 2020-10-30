import os


def get_proj_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_model_dir():
    return os.path.join(get_proj_dir(), 'models')


def get_data_dir():
    return os.path.join(get_proj_dir(), 'data')


def get_training_dir():
    return os.path.join(get_data_dir(), 'TrainingData')


def get_testing_dir():
    return os.path.join(get_data_dir(), 'TestData')