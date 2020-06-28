# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: fcul-als-python
#     language: python
#     name: fcul-als-python
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
import shap                      # Model-agnostic interpretability package inspired on Shapley values
import pickle                    # Save python objects in files
from datetime import datetime    # datetime to use proper date and time formats
import data_utils as du          # Data science and machine learning relevant methods
from ModelInterpreter import ModelInterpreter # Class that enables the interpretation of models that handle variable sequence length input data
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
du.set_random_seed(42)

# Sequence and instance identifier columns' numbers
id_column = 0
inst_column = 1

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

orig_ALS_df.niv_label.value_counts()

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
model = du.deep_learning.load_checkpoint('GitHub/FCUL_ALS_Disease_Progression/models/checkpoint_07_06_2019_23_14.pth',
                                         ModelClass)
model

# ## Getting train and test sets, in tensor format

# Dictionary containing the sequence length (number of temporal events) of each sequence (patient)
seq_len_df = ALS_df.groupby('subject_id').ts.count().to_frame().sort_values(by='ts', ascending=False)
seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.values))])

# +
n_patients = ALS_df.subject_id.nunique()     # Total number of patients
n_inputs = len(ALS_df.columns)               # Number of input features
padding_value = 0                            # Value to be used in the padding

# Pad data (to have fixed sequence length) and convert into a PyTorch tensor
data = du.padding.dataframe_to_padded_tensor(ALS_df, seq_len_dict, padding_value=padding_value)
# -

# Create a Dataset object from the data tensor
dataset = Time_Series_Dataset(data, ALS_df)

# Get the train, validation and test sets data loaders and indices
train_dataloader, val_dataloader, test_dataloader, \
train_indices, val_indices, test_indices            = du.machine_learning.create_train_sets(dataset, test_train_ratio=0.2,
                                                                                            validation_ratio=0.1,
                                                                                            batch_size=1000, get_indeces=True)

# Get the tensor data of the training and test sets
train_features, train_labels = next(iter(train_dataloader))
test_features, test_labels = next(iter(test_dataloader))

# Get the original lengths of the sequences and sort the data
train_features, train_labels, x_lengths_train = utils.sort_by_seq_len(train_features, seq_len_dict, labels=train_labels)
test_features, test_labels, x_lengths_test = utils.sort_by_seq_len(test_features, seq_len_dict, labels=test_labels)

# Create a denormalized version of the feature values so that the plots are easier to understand
test_features_denorm = utils.denormalize_data(orig_ALS_df, test_features, see_progress=False)

# ## Confirm performance metrics

output, metrics_vals = utils.model_inference(model, seq_len_dict, dataloader=test_dataloader,
                       metrics=['loss', 'accuracy', 'AUC', 'precision', 'recall', 'F1'], output_rounded=True)

metrics_vals

# ## SHAP

# ### Deep Explainer

# + {"pixiedust": {"displayParams": {}}}
# Use the first n_bkgnd_samples training examples as our background dataset to integrate over
# (Ignoring the first 2 features, as they constitute the identifiers 'subject_id' and 'ts')
n_bkgnd_samples = 200
explainer = shap.DeepExplainer(model, train_features[:n_bkgnd_samples, :, 2:].float(), feedforward_args=[x_lengths_train[:n_bkgnd_samples]])

# + {"pixiedust": {"displayParams": {}}}
start_time = time.time()
# Explain the predictions of the first n_samples patients in the test set
n_samples = 10
shap_values = explainer.shap_values(test_features[:n_samples, :, 2:].float(),
                                    feedforward_args=[x_lengths_train[:n_bkgnd_samples], x_lengths_test[:n_samples]],
                                    var_seq_len=True)
print(f'Calculation of SHAP values took {time.time() - start_time} seconds')
# -

explainer.expected_value[0]

# Choosing which example to use
subject_id = 39
patient = utils.find_subject_idx(test_features_denorm, subject_id=subject_id)
patient

# +
# Init the JS visualization code
shap.initjs()

# Choosing which example to use
ts = 9

# Plot the explanation of one prediction
shap.force_plot(explainer.expected_value[0], shap_values[patient][ts], features=test_features[patient, ts, 2:].numpy(), feature_names=ALS_cols)
# -

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
# * The output values also seem to be wrong in the patients' force plot, as it goes above 1 instead of matching the original output values.
#
# [After removing padings from data]
# * The SHAP values now seem to have normal values (between -1 and 1) and the plots also look good. However, the sum of the contributions still doesn't add up to the original output values.

# ### Kernel Explainer

# Use a single all zeroes sample as a reference value
num_id_features = sum([1 if i is not None else 0 for i in [id_column, inst_column]])
bkgnd_data = np.zeros((1, len(ALS_cols)+num_id_features))

# Convert the test data into a 2D NumPy matrix
test_data = utils.ts_tensor_to_np_matrix(test_features, list(range(2, len(ALS_df.columns)-1)), padding_value)

from ModelInterpreter import KernelFunction

# Create a function that represents the model's feedforward operation on a single instance
kf = KernelFunction(model)

# Use the background dataset to integrate over
explainer = shap.KernelExplainer(kf.f, bkgnd_data, isRNN=True, model_obj=model, max_bkgnd_samples=100,
                                 id_col_num=id_column, ts_col_num=inst_column)

# Explain the predictions of the sequences in the test set
feat_scores = explainer.shap_values(test_data, l1_reg='num_features(10)', nsamples=3000)

# Summarize the effects of all the features
shap.summary_plot(feat_scores.reshape(-1, model.lstm.input_size),
                  features=test_features_denorm[:, :, 2:].view(-1, model.lstm.input_size).numpy(),
                  feature_names=ALS_cols, plot_type='bar')

# Summarize the effects of all the features
shap.summary_plot(interpreter.feat_scores.reshape(-1, model.lstm.input_size),
                  features=test_features_denorm[:, :interpreter.feat_scores.shape[1], 2:].contiguous().view(-1, interpreter.model.lstm.input_size).numpy(),
                  feature_names=ALS_cols, plot_type='violin')

# Choosing which example to use
subject_id = 125
patient = utils.find_subject_idx(test_features_denorm, subject_id=subject_id)
patient

# +
# Init the JS visualization code
shap.initjs()

# True sequence length of the current patient's data
seq_len = seq_len_dict[test_features_denorm[patient, 0, 0].item()]

# Plot the explanation of the predictions for one patient
shap.force_plot(interpreter.explainer.expected_value[0],
                interpreter.feat_scores[patient, :seq_len],
                features=test_features_denorm[patient, :seq_len, 2:].numpy(),
                feature_names=ALS_cols)
# + {}
# Init the JS visualization code
shap.initjs()

# Choosing which timestamp to use
ts = 9

# Plot the explanation of one prediction
shap.force_plot(interpreter.explainer.expected_value[0],
                interpreter.feat_scores[patient][ts],
                features=test_features_denorm[patient, ts, 2:].numpy(),
                feature_names=ALS_cols)
# -
# ## Model Interpreter
#
# Using my custom class for model interpretability through instance and feature importance.

# + {"pixiedust": {"displayParams": {}}}
interpreter = ModelInterpreter(model, ALS_df, label_column=n_inputs-1, fast_calc=True, SHAP_bkgnd_samples=3000, padding_value=999999)

# + {"pixiedust": {"displayParams": {}}}
# Number of patients to analyse
# n_patients = 1

# _ = interpreter.interpret_model(bkgnd_data=train_features, test_data=test_features[:n_patients], test_labels=test_labels[:n_patients], instance_importance=False, feature_importance=True)
_ = interpreter.interpret_model(bkgnd_data=train_features, test_data=test_features, test_labels=test_labels, instance_importance=True, feature_importance=False)

# +
# Get the current day and time to attach to the saved model's name
current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')

# Path where the model interpreter will be saved
interpreter_path = 'GitHub/FCUL_ALS_Disease_Progression/interpreters/'

# Filename and path where the model will be saved
interpreter_filename = f'{interpreter_path}checkpoint_{current_datetime}.pickle'

# Save model interpreter object, with the instance and feature importance scores, in a pickle file
with open(interpreter_filename, 'wb') as file:
    pickle.dump(interpreter, file)
# -

# Load saved model interpreter object
# with open(interpreter_filename, 'rb') as file:
with open('GitHub/FCUL_ALS_Disease_Progression/interpreters/checkpoint_10_07_2019_05_23.pickle', 'rb') as file:
    interpreter_loaded = pickle.load(file)

if np.array_equal(interpreter_loaded.feat_scores, interpreter.feat_scores):
    print('The model interpreter object was correctly saved.')
    interpreter = interpreter_loaded
else:
    print('ERROR: There was a problem saving the model interpreter object.')

# Only to use when analysing a model interpreter, after having already been saved
interpreter = interpreter_loaded

# ### Feature importance plots

interpreter.feat_scores

# Summarize the effects of all the features
shap.summary_plot(interpreter.feat_scores.reshape(-1, interpreter.model.lstm.input_size),
                  features=test_features_denorm[:, :, 2:].view(-1, interpreter.model.lstm.input_size).numpy(),
                  feature_names=ALS_cols, plot_type='bar')

interpreter.feat_scores.reshape(-1, model.lstm.input_size).shape

test_features_denorm[:, :, 2:].contiguous().view(-1, interpreter.model.lstm.input_size).numpy().shape

# Summarize the effects of all the features
shap.summary_plot(interpreter.feat_scores.reshape(-1, model.lstm.input_size),
                  features=test_features_denorm[:, :interpreter.feat_scores.shape[1], 2:].contiguous().view(-1, interpreter.model.lstm.input_size).numpy(),
                  feature_names=ALS_cols, plot_type='violin')

# **Comments:**
#
# With the current SHAP Kernel Explainer, this plot seems to make sense. However, if not enough samples are used, there isn't much distinction between actually important features and not so relevant ones.

# Choosing which example to use
subject_id = 125
patient = utils.find_subject_idx(test_features_denorm, subject_id=subject_id)
patient

# +
# Init the JS visualization code
shap.initjs()

# True sequence length of the current patient's data
seq_len = seq_len_dict[test_features_denorm[patient, 0, 0].item()]

# Plot the explanation of the predictions for one patient
shap.force_plot(interpreter.explainer.expected_value[0],
                interpreter.feat_scores[patient, :seq_len],
                features=test_features_denorm[patient, :seq_len, 2:].numpy(),
                feature_names=ALS_cols)
# + {}
# Init the JS visualization code
shap.initjs()

# Choosing which timestamp to use
ts = 9

# Plot the explanation of one prediction
shap.force_plot(interpreter.explainer.expected_value[0],
                interpreter.feat_scores[patient][ts],
                features=test_features_denorm[patient, ts, 2:].numpy(),
                feature_names=ALS_cols)
# -
# **Comments:**
#
# With the current SHAP Kernel Explainer, the sum of the contributions match the real output values. As such, it's possible to see how each sequence's output progresses and why (which features had the biggest positive or negative impact).
#
# Although the top features can vary a bit in their ranking when using different quantities of background data and nsamples, as well as having a very small importance difference between them, it appears to increasingly resemble the real behaviour of the model (with increasing background samples). Furthermore, despite the Deep Explainer having issues in matching the original output values, it showed a similar feature importance ranking.
#
# Since this data is processed to have 0 as missing value in categorical features and as the mean in continuous features, if the only background data used is an all zeroes samples, it's possible to do more nsamples in a faster way and achieve a more truthful interpreter.
#
# [TODO] Make sure that a model interpreter trained on very big quantities of background data and nsamples closely resembles the results achieved with an all zeroes sample.

ref_output = interpreter.model(test_features[patient, :, 2:].float().unsqueeze(0), [x_lengths_test[patient]])

ref_output_s = pd.Series([float(x) for x in list(ref_output.detach().numpy())])

# Get an overview of the important features and model output for the current patient
orig_ALS_df[orig_ALS_df.subject_id == subject_id][['ts', 'p0.1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9',
                                                   'p10', '1r', '2r', '3r', 'phrenmeanampl', 'mip']] \
                                                 .reset_index().drop(columns='index').assign(output=ref_output_s)

# ### Instance importance plots

interpreter.inst_scores

interpreter.inst_scores.shape

# interpreter_loaded.inst_scores[interpreter_loaded.inst_scores == interpreter_loaded.padding_value]
interpreter.inst_scores[interpreter.inst_scores == 999999] = np.nan
interpreter.inst_scores

inst_scores_avg = np.nanmean(interpreter.inst_scores, axis=0)

list(range(1, len(inst_scores_avg)+1))

data = [go.Bar(
                x=list(range(1, len(inst_scores_avg[:20])+1)),
                y=list(inst_scores_avg[:20])
              )]
layout = go.Layout(
                    title='Average instance importance scores',
                    xaxis=dict(title='Instance'),
                    yaxis=dict(title='Importance scores')
                  )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')

# +
# True sequence length of the current patient's data
seq_len = seq_len_dict[test_features[patient, 0, 0].item()]

# Plot the instance importance of one sequence
interpreter.instance_importance_plot(test_features, interpreter.inst_scores, patient, seq_len)
# -

ref_output = interpreter.model(test_features[patient, :, 2:].float().unsqueeze(0), [x_lengths_test[patient]])
ref_output

ref_output[-1].item()

n_patients

pred_prob, _ = utils.model_inference(interpreter.model, interpreter.seq_len_dict, data=(test_features[:n_patients], test_labels[:n_patients]),
                                     metrics=[''], seq_final_outputs=True)
pred_prob

pred_prob.shape

test_features[:n_patients].shape

interpreter.inst_scores.shape

interpreter.instance_importance_plot(test_features[:n_patients], interpreter.inst_scores[:n_patients], pred_prob=pred_prob)
