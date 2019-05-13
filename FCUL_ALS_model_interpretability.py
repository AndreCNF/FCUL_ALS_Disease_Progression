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
# -

# Debugging packages
import pixiedust                 # Debugging in Jupyter Notebook cells
import numpy as np               # Math operations with NumPy to confirm model's behaviour
import time                      # Calculate code execution time

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

# Read the original data (before normalization)
orig_ALS_df = pd.read_csv(f'{data_path}cleaned/FCUL_ALS_cleaned_denorm.csv')
orig_ALS_df.head()

# Drop the unnamed index column
ALS_df.drop(columns=['Unnamed: 0', 'niv'], inplace=True)
ALS_df.head()

# Drop the unnamed index and label columns in the original dataframe
orig_ALS_df.drop(columns=['Unnamed: 0', 'niv_label', 'niv'], inplace=True)
orig_ALS_df.head()

ALS_df.describe().transpose()

# +
# List of used features
ALS_cols = list(ALS_df.columns)

# Remove features that aren't used by the model to predict the label
for unused_feature in ['subject_id', 'ts', 'niv_label']:
    ALS_cols.remove(unused_feature)
# -

ALS_cols

# Load the model with the best validation performance
# model = utils.load_checkpoint('GitHub/FCUL_ALS_Disease_Progression/models/checkpoint_26_04_2019_23_36.pth')
model = utils.load_checkpoint('GitHub/FCUL_ALS_Disease_Progression/models/checkpoint_no_NIV_10_05_2019_03_03.pth')

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
                                                                              batch_size=1000, get_indeces=True)

# Get the tensor data of the training and test sets
# train_data = data[train_indices]
# test_data = data[test_indices]
train_features, train_labels = next(iter(train_dataloader))
test_features, test_labels = next(iter(test_dataloader))

# ## Confirm performance metrics

output, metrics_vals = utils.model_inference(model, test_dataloader, seq_len_dict, 
                       metrics=['loss', 'accuracy', 'AUC', 'precision', 'recall', 'F1'], output_rounded=True)

metrics_vals

# +
# Get the original lengths of the sequences, for the test data
x_lengths_test = [seq_len_dict[patient] for patient in list(test_features[:, 0, 0].numpy())]

# Sorted indeces to get the data sorted by sequence length
data_sorted_idx = list(np.argsort(x_lengths_test)[::-1])

# Sort the x_lengths array by descending sequence length
x_lengths_test = [x_lengths_test[idx] for idx in data_sorted_idx]

# Sort the features and labels by descending sequence length
test_data_exp = test_features[data_sorted_idx, :, :]
test_labels = test_labels[data_sorted_idx, :]

# +
# Adjust the labels so that it gets the exact same shape as the predictions
# (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
labels = torch.nn.utils.rnn.pack_padded_sequence(test_labels, x_lengths_test, batch_first=True)
labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=999999)

mask = (labels <= 1).view(-1, 1).float()                    # Create a mask by filtering out all labels that are not a padding value
unpadded_labels = torch.masked_select(labels.contiguous().view(-1, 1), mask.byte()) # Completely remove the padded values from the labels using the mask
# -

[tensor.item() for tensor in list(unpadded_labels.int())]

[tensor.item() for tensor in list(output)]

list(np.diff(unpadded_labels.int().numpy()))

[i for i, x in enumerate(list(np.diff(unpadded_labels.int().numpy()))) if x==1]

[i for i, x in enumerate(list(np.diff(output.int().numpy()))) if x==1]

# **Comment:** [Before removing NIV from the features] Most times, the model only predicts NIV use after the patient already started that treatment. This means that it usely only predicts the continuation of the treatment, which isn't so useful. Need to experiment training a model without giving any information regarding current NIV usage.

# ## SHAP

# Get the original lengths of the sequences and sort the data
train_features, x_lengths_train = utils.sort_by_seq_len(train_features, seq_len_dict)
test_features, x_lengths_test = utils.sort_by_seq_len(test_features, seq_len_dict)

# + {"pixiedust": {"displayParams": {}}}
# Use the first 200 training examples as our background dataset to integrate over
# (Ignoring the first 2 features, as they constitute the identifiers 'subject_id' and 'ts')
explainer = shap.DeepExplainer(model, train_features[:, :, 2:].float(), feedforward_args=[x_lengths_train])

# + {"pixiedust": {"displayParams": {}}}
start_time = time.time()
# Explain the predictions of the first 10 patients in the test set
n_samples = 1
shap_values = explainer.shap_values(test_features[:n_samples, :, 2:].float(), 
                                    feedforward_args=[x_lengths_train, x_lengths_test[:n_samples]],
                                    var_seq_len=True)
print(f'Calculation of SHAP values took {time.time() - start_time} seconds')
# -

explainer.expected_value[0]

# +
# Init the JS visualization code
shap.initjs()

# Choosing which example to use
patient = 0
ts = 1

# Plot the explanation of one prediction
shap.force_plot(explainer.expected_value[0], shap_values[patient][ts], features=test_features[patient, ts, 2:].numpy(), feature_names=ALS_cols)
# -

# Denormalize the feature values so that the plots are easier to understand
test_features_denorm = utils.denormalize_data(orig_ALS_df, test_features)

test_features_denorm.shape

len(orig_ALS_df.columns)

# +
# Init the JS visualization code
shap.initjs()

# Choosing which example to use
patient = 0
ts = 1

# Plot the explanation of one prediction
shap.force_plot(explainer.expected_value[0], shap_values[patient][ts], features=test_features_denorm[patient, ts, 2:].numpy(), feature_names=ALS_cols)
# + {}
# Init the JS visualization code
shap.initjs()

# Choosing which example to use
patient = 0

# True sequence length of the current patient's data
seq_len = seq_len_dict[test_features_denorm[patient, 0, 0].item()]

# Plot the explanation of the predictions for one patient
shap.force_plot(explainer.expected_value[0], shap_values[patient, :seq_len], features=test_features_denorm[patient, :seq_len, 2:].numpy(), feature_names=ALS_cols)
# -
# Summarize the effects of all the features
shap.summary_plot(shap_values.reshape(-1, model.lstm.input_size), features=test_features_denorm[:n_samples, :, 2:].contiguous().view(-1, model.lstm.input_size).numpy(), feature_names=ALS_cols)

# Summarize the effects of all the features
shap.summary_plot(shap_values.reshape(-1, model.lstm.input_size), features=test_features_denorm[:, :, 2:].view(-1, model.lstm.input_size).numpy(), feature_names=ALS_cols, plot_type='bar')

# Summarize the effects of all the features
shap.summary_plot(shap_values.reshape(-1, model.lstm.input_size), features=test_features_denorm[:n_samples, :, 2:].contiguous().view(-1, model.lstm.input_size).numpy(), feature_names=ALS_cols, plot_type='violin')

# **Comments:**
#
# [Before removing padings from data]
# * The SHAP values are significantly higher than what I usually see (tends to be between -1 and 1, not between -100000 and 250000). It seems to be because of the padding (the padding value is 999999).
# * ~The output values also seem to be wrong in the patients' force plot, as it goes above 1.~ It doesn't seem to be a problem after all, it's just a SHAP indicator of whether the prediction will be 0 (if the value is negative) or 1 (if the value is positive).


