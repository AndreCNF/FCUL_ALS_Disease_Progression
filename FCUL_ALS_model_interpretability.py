# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # FCUL ALS Model Interpretability
# ---
#
# Exploring the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.
#
# Using different interpretability approaches so as to understand the outputs of the models trained on FCUL's ALS dataset.

# + {"colab_type": "text", "id": "KOdmFzXqF7nq", "cell_type": "markdown"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import pandas as pd              # Pandas to handle the data in dataframes
import re                        # re to do regex searches in string data
import plotly                    # Plotly for interactive and pretty plots
import plotly.graph_objs as go
import os                        # os handles directory/workspace changes
import numpy as np               # NumPy to handle numeric and NaN operations
from tqdm import tqdm_notebook   # tqdm allows to track code execution progress
import torch                     # PyTorch to create and apply deep learning models
from torch.utils.data.sampler import SubsetRandomSampler
import utils                     # Contains auxiliary functions
from Time_Series_Dataset import Time_Series_Dataset # Dataset subclass which allows the creation of Dataset objects
import shap                      # Model-agnostic interpretability package inspired on Shapley values

# +
# Change to parent directory (presumably "Documents")
os.chdir("../..")

# Path to the CSV dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/'

# + {"colab_type": "text", "id": "bEqFkmlYCGOz", "cell_type": "markdown"}
# **Important:** Use the following two lines to be able to do plotly plots offline:

# + {"colab": {}, "colab_type": "code", "id": "fZCUmUOzCPeI"}
import plotly.offline as py
plotly.offline.init_notebook_mode(connected=True)


# + {"colab_type": "text", "id": "Yrzi8YbzDVTH", "cell_type": "markdown"}
# **Important:** The following function is needed in every Google Colab cell that contains a Plotly chart:

# + {"colab": {}, "colab_type": "code", "id": "wxyGCedgC6bX"}
def configure_plotly_browser_state():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))


# -

# Set random seed to the specified value
np.random.seed(utils.random_seed)
torch.manual_seed(utils.random_seed)

# ## Loading data and model

# Read the data (already processed, just like the model trained on)
ALS_df = pd.read_csv(f'{data_path}cleaned/FCUL_ALS_cleaned.csv')
ALS_df.head()

# Drop the unnamed index column
ALS_df.drop(columns='Unnamed: 0', inplace=True)
ALS_df.head()

# Load the model with the best validation performance
model = utils.load_checkpoint('GitHub/FCUL_ALS_Disease_Progression/models/checkpoint_26_04_2019_23_36.pth')

model

# ## Getting train and test sets, in tensor format

# Dictionary containing the sequence length (number of temporal events) of each sequence (patient)
seq_len_df = ALS_df.groupby('subject_id').ts.count().to_frame().sort_values(by='ts', ascending=False)
seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.values))])

# +
n_patients = ALS_df.subject_id.nunique()     # Total number of patients
n_inputs = len(ALS_df.columns)               # Number of input features
padding_value = 999999                       # Value to be used in the padding

# Pad data (to have fixed sequence length) and convert into a PyTorch tensor
data = utils.dataframe_to_padded_tensor(ALS_df, seq_len_dict, n_patients, n_inputs, padding_value=padding_value)
# -

# Create a Dataset object from the data tensor
dataset = Time_Series_Dataset(data, ALS_df)

# Get the train, validation and test sets data loaders and indices
train_dataloader, val_dataloader, test_dataloader, \
train_indices, val_indices, test_indices            = utils.create_train_sets(dataset, test_train_ratio=0.2, 
                                                                              validation_ratio=0.1, 
                                                                              batch_size=1, get_indeces=True)

# Get the tensor data of the training and test sets
train_data = data[train_indices]
test_data = data[test_indices]

# ## SHAP

# +
# Get the original lengths of the sequences, for the training data
x_lengths_train = [seq_len_dict[patient] for patient in list(train_data[:100, 0, 0].numpy())]

# Sorted indeces to get the data sorted by sequence length
data_sorted_idx = list(np.argsort(x_lengths_train)[::-1])

# Sort the x_lengths array by descending sequence length
x_lengths_train = [x_lengths_train[idx] for idx in data_sorted_idx]

# Sort the features by descending sequence length
train_data_exp = train_data[data_sorted_idx, :, :]

# +
# Get the original lengths of the sequences, for the test data
x_lengths_test = [seq_len_dict[patient] for patient in list(test_data[:10, 0, 0].numpy())]

# Sorted indeces to get the data sorted by sequence length
data_sorted_idx = list(np.argsort(x_lengths_test)[::-1])

# Sort the x_lengths array by descending sequence length
x_lengths_test = [x_lengths_test[idx] for idx in data_sorted_idx]

# Sort the features by descending sequence length
test_data_exp = test_data[data_sorted_idx, :, :]

# +
# Use the first 100 training examples as our background dataset to integrate over
# (Ignoring the first 2 features, as they constitute the identifiers 'subject_id' and 'ts')
explainer = shap.DeepExplainer(model, [train_data_exp[:, :, 2:], x_lengths_train])

# Explain the first 10 predictions
shap_values = explainer.shap_values([test_data_exp[:, :, 2:], x_lengths_test])

# +
# init the JS visualization code
shap.initjs()

# plot the explanation of the first prediction
# Note the model is "multi-output" because it is rank-2 but only has one column
shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test_words[0])
