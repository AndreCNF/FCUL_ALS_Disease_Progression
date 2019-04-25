from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters and do version control
import torch                                            # PyTorch to create and apply deep learning models
from torch import nn, optim                             # nn for neural network layers and optim for training optimizers
import pandas as pd                                     # Pandas to handle the data in dataframes
from datetime import datetime                           # datetime to use proper date and time formats
import os                                               # os handles directory/workspace changes
import numpy as np                                      # NumPy to handle numeric and NaN operations
from NeuralNetwork import NeuralNetwork                 # Import the neural network model class
import utils                                            # Contains auxiliary functions
from Time_Series_Dataset import Time_Series_Dataset     # Dataset subclass which allows the creation of Dataset objects

# Set random seed to 0
np.random.seed(0)
torch.manual_seed(0)

# Change to parent directory (presumably "Documents")
os.chdir("../..")

# Path to the CSV dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/cleaned/'

print('Reading the CSV data...')

# Read the cleaned dataset dataframe
ALS_df = pd.read_csv(f'{data_path}FCUL_ALS_cleaned.csv')

# Drop the unnamed index column
ALS_df.drop(columns='Unnamed: 0', inplace=True)

# Neural network and dataset parameters
n_patients = ALS_df.subject_id.nunique()     # Total number of patients
n_inputs = len(ALS_df.columns)               # Number of input features
n_hidden = 1052                              # Number of hidden units
n_outputs = 1                                # Number of outputs
n_layers = 2                                 # Number of LSTM layers
p_dropout = 0.2                              # Probability of dropout

print('Building a dictionary containing the sequence length of each patient\'s time series...')

# Dictionary containing the sequence length (number of temporal events) of each sequence (patient)
seq_len_df = ALS_df.groupby('subject_id').ts.count().to_frame().sort_values(by='ts', ascending=False)
seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.values))])

print('Creating a padded tensor version of the dataframe...')

# Value to be used in the padding
padding_value = 999999

# Pad data (to have fixed sequence length) and convert into a PyTorch tensor
data = utils.dataframe_to_padded_tensor(ALS_df, seq_len_dict, n_patients, n_inputs, padding_value=padding_value)

print('Instantiating the model...')

# Instantiate the model (removing the two identifier columns and the labels from the input size)
model = NeuralNetwork(n_inputs-3, n_hidden, n_outputs, n_layers, p_dropout)

# Check if GPU (CUDA) is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    model = model.cuda()                        # Move the model to the GPU

# Training parameters
batch_size = 32                                 # Number of patients in a mini batch
n_epochs = 50                                   # Number of epochs
lr = 0.001                                      # Learning rate

print('Creating a dataset object...')

# Create a Dataset object from the data tensor
dataset = Time_Series_Dataset(data, ALS_df)

print('Distributing the data to train, validation and test sets and getting their data loaders...')

# Get the train, validation and test sets data loaders, which will allow loading batches
train_dataloader, val_dataloader, test_dataloader = utils.create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1, 
                                                                            batch_size=batch_size, get_indeces=False)

print('Training the model...')

# Train the model
model = utils.train(model, train_dataloader, val_dataloader, test_dataloader, seq_len_dict, batch_size, n_epochs, lr,
                    model_path='GitHub/FCUL_ALS_Disease_Progression/models/', padding_value=padding_value,
                    do_test=True, log_comet_ml=True, comet_ml_api_key='',
                    comet_ml_project_name='', comet_ml_workspace='', comet_ml_save_model=True,
                    features_list=list(ALS_df.columns).remove('niv_label'))

# [TODO] Interpret the model
