import pandas as pd              # Pandas to handle the data in dataframes
import re                        # re to do regex searches in string data
import os                        # os handles directory/workspace changes
import numpy as np               # NumPy to handle numeric and NaN operations
from tqdm import tqdm_notebook   # tqdm allows to track code execution progress
import torch                     # PyTorch to create and apply deep learning models
from torch.utils.data.sampler import SubsetRandomSampler
import shap                      # Model-agnostic interpretability package inspired on Shapley values
import pickle                    # Save python objects in files
from datetime import datetime    # datetime to use proper date and time formats
import utils                     # Contains auxiliary functions
from Time_Series_Dataset import Time_Series_Dataset # Dataset subclass which allows the creation of Dataset objects
from ModelInterpreter import ModelInterpreter # Class that enables the interpretation of models that handle variable sequence length input data
import numpy as np               # Math operations with NumPy to confirm model's behaviour
import time                      # Calculate code execution time

# +
# Change to parent directory (presumably "Documents")
os.chdir("../..")

# Path to the CSV dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/'

# Path where the models are stored
model_path = 'GitHub/FCUL_ALS_Disease_Progression/models/'

# Path where the model interpreter will be saved
interpreter_path = 'GitHub/FCUL_ALS_Disease_Progression/interpreters/'

# -

# Set random seed to the specified value
np.random.seed(utils.random_seed)
torch.manual_seed(utils.random_seed)

# ## Loading data and model

# Read the data (already processed, just like the model trained on)
ALS_df = pd.read_csv(f'{data_path}cleaned/FCUL_ALS_cleaned.csv')

# Read the original data (before normalization)
orig_ALS_df = pd.read_csv(f'{data_path}cleaned/FCUL_ALS_cleaned_denorm.csv')

# Drop the unnamed index column
ALS_df.drop(columns=['Unnamed: 0', 'niv'], inplace=True)

# Drop the unnamed index and label columns in the original dataframe
orig_ALS_df.drop(columns=['Unnamed: 0', 'niv_label', 'niv'], inplace=True)

# +
# List of used features
ALS_cols = list(ALS_df.columns)

# Remove features that aren't used by the model to predict the label
for unused_feature in ['subject_id', 'ts', 'niv_label']:
    ALS_cols.remove(unused_feature)
# -

# Load the model with the best validation performance
# model = utils.load_checkpoint('GitHub/FCUL_ALS_Disease_Progression/models/checkpoint_26_04_2019_23_36.pth')
model = utils.load_checkpoint(f'{model_path}checkpoint_no_NIV_10_05_2019_03_03.pth')

# ## Getting train and test sets, in tensor format

# Dictionary containing the sequence length (number of temporal events) of each sequence (patient)
seq_len_df = ALS_df.groupby('subject_id').ts.count().to_frame().sort_values(by='ts', ascending=False)
seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.values))])

# +
n_patients = ALS_df.subject_id.nunique()     # Total number of patients
n_inputs = len(ALS_df.columns)               # Number of input features
padding_value = 0                            # Value to be used in the padding

# Pad data (to have fixed sequence length) and convert into a PyTorch tensor
data = utils.dataframe_to_padded_tensor(ALS_df, seq_len_dict, n_patients, n_inputs, padding_value=padding_value)
# -

# Create a Dataset object from the data tensor
dataset = Time_Series_Dataset(data, ALS_df)

# Get the train, validation and test sets data loaders and indices
train_dataloader, val_dataloader, test_dataloader, \
train_indices, val_indices, test_indices            = utils.create_train_sets(dataset, test_train_ratio=0.2,
                                                                              validation_ratio=0.1,
                                                                              batch_size=1000, get_indeces=True)

# Get the tensor data of the training and test sets
train_features, train_labels = next(iter(train_dataloader))
test_features, test_labels = next(iter(test_dataloader))

# Get the original lengths of the sequences and sort the data
train_features, train_labels, x_lengths_train = utils.sort_by_seq_len(train_features, seq_len_dict, labels=train_labels)
test_features, test_labels, x_lengths_test = utils.sort_by_seq_len(test_features, seq_len_dict, labels=test_labels)

# Denormalize the feature values so that the plots are easier to understand
test_features_denorm = utils.denormalize_data(orig_ALS_df, test_features)

# ## Model Interpreter
#
# Using my custom class for model interpretability through instance and feature importance.

interpreter = ModelInterpreter(model, ALS_df, seq_len_dict, fast_calc=False, SHAP_bkgnd_samples=1000, padding_value=padding_value)
_ = interpreter.interpret_model(bkgnd_data=train_features, test_data=test_features, test_labels=test_labels, instance_importance=True, feature_importance=True)

# +
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')

# Filename and path where the model will be saved
interpreter_filename = f'{interpreter_path}checkpoint_{current_datetime}.pickle'

# Save model interpreter object, with the instance and feature importance scores, in a pickle file
with open(interpreter_filename, 'wb') as file:
    pickle.dump(interpreter, file)
# -

# Load saved model interpreter object
with open(interpreter_filename, 'rb') as file:
    interpreter_loaded = pickle.load(file)

if np.array_equal(interpreter_loaded.feat_scores, interpreter.feat_scores):
    print('The model interpreter object was correctly saved.')
    interpreter = interpreter_loaded
else:
    print('ERROR: There was a problem saving the model interpreter object.')
