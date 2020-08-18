# -*- coding: utf-8 -*-
# # FCUL ALS Model Interpretability
# ---
#
# Interpreting models trained on the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.
#
# Using different interpretability approaches so as to understand the outputs of the models trained on FCUL's ALS dataset.

# ## Importing the necessary packages

import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import torch                               # PyTorch to create and apply deep learning models
import xgboost as xgb                      # Gradient boosting trees models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib                              # Save and load scikit-learn models in disk
import pickle                              # Save python objects in files
import yaml                                # Save and load YAML files
from datetime import datetime              # datetime to use proper date and time formats
from ipywidgets import interact            # Display selectors and sliders
import shap                                # Model-agnostic interpretability package inspired on Shapley values
import plotly.graph_objs as go             # Plotly for interactive and pretty plots
from model_interpreter.model_interpreter import ModelInterpreter # Class that enables the interpretation of models that handle variable sequence length input data

import pixiedust                           # Debugging in Jupyter Notebook cells

# Path to the parquet dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/cleaned/'
# Path to the data + SHAP values dataframes
data_n_shap_path = 'Datasets/Thesis/FCUL_ALS/interpreted/'
# Path to the code files
project_path = 'GitHub/FCUL_ALS_Disease_Progression/'
# Path to the models
models_path = f'{project_path}models/'
# Path to the model interpreters
interpreters_path = f'{project_path}interpreters/'

# Change to the scripts directory
os.chdir("../scripts/")
import utils                               # Context specific (in this case, for the ALS data) methods
import Models                              # Deep learning models
# Change to parent directory (presumably "Documents")
os.chdir("../../..")
import pandas as pd                        # Pandas to load and handle the data
import data_utils as du                    # Data science and machine learning relevant methods

du.set_pandas_library(lib='pandas')

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

# Set the random seed for reproducibility:

du.set_random_seed(42)

# Allow Jupyter Lab to display all outputs:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# ## Initializing variables

# Dataset parameters:

id_column = 'subject_id'                   # Name of the sequence ID column
ts_column = 'ts'                           # Name of the timestamp column
label_column = 'niv_label'                 # Name of the label column
padding_value = 999999                     # Padding value used to fill in sequences up to the maximum sequence length

# One hot encoding columns categorization:

stream_categ_feat_ohe = open(f'{data_path}categ_feat_ohe.yml', 'r')

categ_feat_ohe = yaml.load(stream_categ_feat_ohe, Loader=yaml.FullLoader)
categ_feat_ohe

list(categ_feat_ohe.keys())

# Normalization stats (allow us to get back the original values, to display in the interpretability plots):

stream_norm_stats = open(f'{data_path}norm_stats.yml', 'r')

norm_stats = yaml.load(stream_norm_stats, Loader=yaml.FullLoader)
norm_stats

list(norm_stats.keys())

means = dict([(feat, norm_stats[feat]['mean']) for feat in norm_stats])
means

stds = dict([(feat, norm_stats[feat]['std']) for feat in norm_stats])
stds

# Dataloaders parameters:

test_train_ratio = 0.25                    # Percentage of the data which will be used as a test set
validation_ratio = 0.1                     # Percentage of the data from the training set which is used for validation purposes
batch_size = 32                            # Number of unit stays in a mini batch

# Model to interpret:

model_filename = None                      # Name of the file containing the model that will be loaded
model_class = None                         # Python class name that corresponds to the chosen model's type
model = None                               # Machine learning model object
model2 = None                              # Other machine learning model object
model3 = None                              # Other machine learning model object
dataset_mode = 'one hot encoded'           # The mode in which we'll use the data, either one hot encoded or pre-embedded
ml_core = 'deep learning'                  # The core machine learning type we'll use; either traditional ML or DL
use_delta_ts = False                       # Indicates if we'll use time variation info
time_window_days = 90                      # Number of days on which we want to predict NIV
is_custom = False                          # Indicates if the model being used is a custom built one
@interact
def get_dataset_mode(model_name=['Bidirectional LSTM with embedding layer and delta_ts',
                                 'Bidirectional LSTM with embedding layer',
                                 'Bidirectional LSTM with delta_ts',
                                 'Bidirectional LSTM',
                                 'LSTM with embedding layer and delta_ts',
                                 'LSTM with embedding layer',
                                 'LSTM with delta_ts',
                                 'LSTM',
                                 'Bidirectional RNN with embedding layer and delta_ts',
                                 'Bidirectional RNN with embedding layer',
                                 'Bidirectional RNN with delta_ts',
                                 'Bidirectional RNN',
                                 'RNN with embedding layer and delta_ts',
                                 'RNN with embedding layer',
                                 'RNN with delta_ts',
                                 'RNN',
                                 'MF1-LSTM',
                                 'MF2-LSTM',
                                 'TLSTM',
                                 'XGBoost',
                                 'Logistic regression']):
    global model_filename, model_class, model, model2, model3, dataset_mode, ml_core, use_delta_ts 
    global time_window_days, is_custom
    if model_name == 'Bidirectional LSTM with embedding layer and delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_bidir_pre_embedded_delta_ts_90dayswindow_0.3705valloss_08_07_2020_04_04.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional LSTM with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_bidir_pre_embedded_90dayswindow_0.2490valloss_06_07_2020_03_47.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional LSTM with delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3809valloss_06_07_2020_04_08.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional LSTM':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_bidir_one_hot_encoded_90dayswindow_0.4497valloss_08_07_2020_04_31.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'LSTM with embedding layer and delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_pre_embedded_delta_ts_90dayswindow_0.4771valloss_06_07_2020_03_55.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'LSTM with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_pre_embedded_90dayswindow_0.5898valloss_06_07_2020_03_21.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'LSTM with delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_one_hot_encoded_delta_ts_90dayswindow_0.5178valloss_06_07_2020_04_02.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'LSTM':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_one_hot_encoded_90dayswindow_0.4363valloss_06_07_2020_03_28.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional RNN with embedding layer and delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_bidir_pre_embedded_delta_ts_90dayswindow_0.3059valloss_06_07_2020_03_10.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional RNN with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_bidir_pre_embedded_90dayswindow_0.4005valloss_08_07_2020_03_43.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional RNN with delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3631valloss_08_07_2020_04_21.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional RNN':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_bidir_one_hot_encoded_90dayswindow_0.3713valloss_08_07_2020_04_49.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'RNN with embedding layer and delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_pre_embedded_delta_ts_30dayswindow_0.4746valloss_29_06_2020_17_30.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'RNN with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_with_embedding_90dayswindow_0.5569valloss_30_06_2020_17_04.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'RNN with delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_one_hot_encoded_delta_ts_90dayswindow_0.4275valloss_06_07_2020_02_55.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'RNN':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_one_hot_encoded_90dayswindow_0.5497valloss_30_06_2020_18_25.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'MF1-LSTM':
        # Set the model file and class names, then load the model
        model_filename = 'mf1lstm_one_hot_encoded_90dayswindow_0.6009valloss_07_07_2020_03_46.pth'
        model_class = 'MF1LSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = True
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'MF2-LSTM':
        # Set the model file and class names, then load the model
        model_filename = ''
        model_class = 'MF2LSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of delta_ts
        use_delta_ts = 'raw'
        # Set it as a custom model
        is_custom = True
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'TLSTM':
        # Set the model file and class names, then load the model
        model_filename = ''
        model_class = 'TLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = True
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'XGBoost':
        # Set the model file and class names, then load the model
        model_filename = 'xgb_0.5926valloss_09_07_2020_02_40.pth'
        model_class = 'XGBoost'
        model = xgb.XGBClassifier()
        model.load_model(f'{models_path}{model_filename}')
        # Set as a traditional ML model
        ml_core = 'machine learning'
    elif model_name == 'Logistic regression':
        # Set the model file and class names, then load the model
        model_filename = 'logreg_0.6210valloss_09_07_2020_02_54.pth'
        model_class = 'logreg'
        model = joblib.load(f'{models_path}{model_filename}')
        # Set as a traditional ML model
        ml_core = 'machine learning'
    print(model)


# ## Loading the data

# Original data:

orig_ALS_df = pd.read_csv(f'{data_path}FCUL_ALS_cleaned_denorm.csv')
orig_ALS_df.drop(columns=['Unnamed: 0'], inplace=True)
orig_ALS_df.head()

# Preprocessed data:

ALS_df = pd.read_csv(f'{data_path}FCUL_ALS_cleaned.csv')
ALS_df.head()

# Remove the `Unnamed: 0` column:

ALS_df.drop(columns=['Unnamed: 0'], inplace=True)

ALS_df.columns

len(ALS_df.columns)

# Find the maximum sequence length, so that the ML models and their related methods can handle all sequences, which have varying sequence lengths:

total_length = ALS_df.groupby(id_column)[ts_column].count().max()
total_length

# ## Preprocessing data

# Define the label column, in case we're using a time window different than 90 days:

if time_window_days is not 90:
    # Recalculate the NIV label, based on the chosen time window
    ALS_df[label_column] = utils.set_niv_label(ALS_df, time_window_days)
    display(ALS_df.head())


# Remove the `niv` column:

ALS_df.drop(columns=['niv'], inplace=True)

# Add the `delta_ts` (time variation between samples) if required:

if use_delta_ts is not False:
    # Create a time variation column
    ALS_df['delta_ts'] = ALS_df.groupby(id_column).ts.diff()
    # Fill all the delta_ts missing values (the first value in a time series) with zeros
    ALS_df['delta_ts'] = ALS_df['delta_ts'].fillna(0)
if use_delta_ts == 'normalized':
    # Add delta_ts' normalization stats to the dictionaries
    means['delta_ts'] = ALS_df['delta_ts'].mean()
    stds['delta_ts'] = ALS_df['delta_ts'].std()
    # Normalize the time variation data
    # NOTE: When using the MF2-LSTM model, since it assumes that the time
    # variation is in days, we shouldn't normalize `delta_ts` with this model.
    ALS_df['delta_ts'] = (ALS_df['delta_ts'] - means['delta_ts']) / stds['delta_ts']
else:
    # Ignore delta_ts' normalization stats to the dictionaries
    means['delta_ts'] = 0
    stds['delta_ts'] = 1
if use_delta_ts is not False:
    ALS_df.head()

# Convert into a padded tensor:

data = du.padding.dataframe_to_padded_tensor(ALS_df, padding_value=padding_value,
                                             label_column=label_column, inplace=True)
data

# Set the embedding configuration, if needed:

# Indices of the ID, timestamp and label columns
id_column_idx = du.search_explore.find_col_idx(ALS_df, id_column)
ts_column_idx = du.search_explore.find_col_idx(ALS_df, ts_column)
label_column_idx = du.search_explore.find_col_idx(ALS_df, label_column)
print(
f'''ID index: {id_column_idx}
Timestamp index: {ts_column_idx}
Label index: {label_column_idx}'''
)

if dataset_mode == 'one hot encoded':
    embed_features = None
else:
    embed_features = list()
    if len(categ_feat_ohe.keys()) == 1:
        for ohe_feature in list(categ_feat_ohe.values())[0]:
            # Find the current feature's index so as to be able to use it as a tensor
            feature_idx = du.search_explore.find_col_idx(ALS_df, ohe_feature)
            # Decrease the index number if it's larger than the label column (which will be removed)
            if feature_idx > label_column_idx:
                feature_idx = feature_idx - 1
            embed_features.append(feature_idx)
    else:
        for i in range(len(categ_feat_ohe.keys())):
            tmp_list = list()
            for ohe_feature in list(categ_feat_ohe.values())[i]:
                # Find the current feature's index so as to be able to use it as a tensor
                feature_idx = du.search_explore.find_col_idx(ALS_df, ohe_feature)
                # Decrease the index number if it's larger than the label column (which will be removed)
                if feature_idx > label_column_idx:
                    feature_idx = feature_idx - 1
                tmp_list.append(feature_idx)
            # Add the current feature's list of one hot encoded columns
            embed_features.append(tmp_list)
print(f'Embedding features: {embed_features}')

# Gather the feature names:

feature_columns = list(ALS_df.columns)
feature_columns.remove('niv_label')
if ml_core == 'machine learning':
    feature_columns.remove('subject_id')
    feature_columns.remove('ts')
    model_feature_columns = feature_columns
else:
    model_feature_columns = feature_columns.copy()
    model_feature_columns.remove('subject_id')
    model_feature_columns.remove('ts')

# ## Defining the dataset object

dataset = du.datasets.Time_Series_Dataset(ALS_df, data, padding_value=padding_value,
                                          label_name=label_column)

dataset.__len__()

# ## Separating into train and validation sets

(train_dataloader, val_dataloader, test_dataloader,
train_indeces, val_indeces, test_indeces) = du.machine_learning.create_train_sets(dataset,
                                                                                  test_train_ratio=test_train_ratio,
                                                                                  validation_ratio=validation_ratio,
                                                                                  batch_size=batch_size,
                                                                                  get_indices=True)

# Get the full arrays of each set
train_features, train_labels = dataset.X[train_indeces], dataset.y[train_indeces]
val_features, val_labels = dataset.X[val_indeces], dataset.y[val_indeces]
test_features, test_labels = dataset.X[test_indeces], dataset.y[test_indeces]
all_features, all_labels = dataset.X, dataset.y

# Ignore the dataloaders, we only care about the full arrays when using scikit-learn or XGBoost
del train_dataloader
del val_dataloader
del test_dataloader

if ml_core == 'machine learning':
    # Reshape the data into a 2D format
    train_features = train_features.reshape(-1, train_features.shape[-1])
    val_features = val_features.reshape(-1, val_features.shape[-1])
    test_features = test_features.reshape(-1, test_features.shape[-1])
    all_features = all_features.reshape(-1, all_features.shape[-1])
    train_labels = train_labels.reshape(-1)
    val_labels = val_labels.reshape(-1)
    test_labels = test_labels.reshape(-1)
    all_labels = all_labels.reshape(-1)
    # Remove padding samples from the data
    train_features = train_features[[padding_value not in row for row in train_features]]
    val_features = val_features[[padding_value not in row for row in val_features]]
    test_features = test_features[[padding_value not in row for row in test_features]]
    all_features = all_features[[padding_value not in row for row in all_features]]
    train_labels = train_labels[[padding_value not in row for row in train_labels]]
    val_labels = val_labels[[padding_value not in row for row in val_labels]]
    test_labels = test_labels[[padding_value not in row for row in test_labels]]
    all_labels = all_labels[[padding_value not in row for row in all_labels]]
    # Convert from PyTorch tensor to NumPy array
    train_features = train_features.numpy()
    val_features = val_features.numpy()
    test_features = test_features.numpy()
    all_features = all_features.numpy()
    train_labels = train_labels.numpy()
    val_labels = val_labels.numpy()
    test_labels = test_labels.numpy()
    all_labels = all_labels.numpy()

train_features

train_features.shape

# Get the original, denormalized test data:

if use_delta_ts is False:
    # Prevent the identifier columns from being denormalized
    columns_to_remove = [id_column, ts_column, 'delta_ts']
else:
    # Also prevent the time variation column from being denormalized
    columns_to_remove = [id_column, ts_column]

if ml_core == 'deep learning':
    denorm_data = du.data_processing.denormalize_data(ALS_df, data=all_features, 
                                                      id_columns=[id_column, ts_column],
                                                      feature_columns=feature_columns,
                                                      means=means, stds=stds,
                                                      see_progress=False)
else:
    denorm_data = du.data_processing.denormalize_data(ALS_df, data=all_features[:, 2:], 
                                                      id_columns=None,
                                                      feature_columns=feature_columns,
                                                      means=means, stds=stds,
                                                      see_progress=False)

# Testing the denormalization (getting the original values back):

if ml_core == 'deep learning':
    print(denorm_data[0, 0])
else:
    print(denorm_data[0])

if ml_core == 'deep learning':
    orig_ALS_df[(orig_ALS_df.subject_id == int(all_features[0, 0, 0])) & (orig_ALS_df.ts == int(all_features[0, 0, 1]))]
else:
    orig_ALS_df[(orig_ALS_df.subject_id == 2) & (orig_ALS_df.ts == 27)]

if ml_core == 'deep learning':
    print(orig_ALS_df[
            (orig_ALS_df.subject_id == int(all_features[0, 0, 0])) 
            & (orig_ALS_df.ts == int(all_features[0, 0, 1]))
        ].drop(columns=['niv', 'niv_label']).values == denorm_data[0, 0].numpy())
else:
    print(orig_ALS_df[
            (orig_ALS_df.subject_id == 2) 
            & (orig_ALS_df.ts == 27)
        ].drop(columns=['subject_id', 'ts', 'niv', 'niv_label']).values == denorm_data[0])

# ## Interpreting the model

# Define the interpreter:

if ml_core == 'deep learning':
    # Calculating the number of times to re-evaluate the model when explaining each prediction,
    # based on SHAP's formula of nsamples = 2 * n_features + 2048
    SHAP_bkgnd_samples = 2 * all_features.shape[-1] + 2048
#     SHAP_bkgnd_samples = 2 * test_features.shape[-1]
#     SHAP_bkgnd_samples = 200
    print(SHAP_bkgnd_samples)

if ml_core == 'deep learning':
    interpreter = ModelInterpreter(model, ALS_df, model_type='multivariate_rnn', id_column_name=id_column, 
                                   inst_column_name=ts_column, label_column_name=label_column, 
                                   fast_calc=True, SHAP_bkgnd_samples=SHAP_bkgnd_samples, 
                                   random_seed=du.random_seed, padding_value=padding_value, 
                                   is_custom=is_custom, total_length=total_length, occlusion_wgt=0.7)
elif model_class == 'XGBoost':
    interpreter = shap.TreeExplainer(model)
interpreter


# ### Instance importance

# Calculate instance importance scores:

if ml_core == 'deep learning':
    inst_scores = interpreter.interpret_model(test_data=all_features,
                                              test_labels=all_labels,
                                              instance_importance=True, 
                                              feature_importance=False)


inst_scores

# Visualize the instance importance:

interpreter.instance_importance_plot(orig_data=interpreter.test_data, inst_scores=inst_scores,
                                     pred_prob=None, uniform_spacing=False,
                                     show_pred_prob=False, show_title=True,
                                     show_colorbar=True, click_mode='event+select',
                                     labels=interpreter.test_labels, seq_len=None, threshold=0,
                                     get_fig_obj=False, tensor_idx=True,
                                     max_seq=10, background_color='black',
                                     font_family='Roboto', font_size=14,
                                     font_color='white')

# Check a patient's outputs to see if the instance importance scores make sense:

subjects = list(interpreter.test_data[:, :, 0].unique().int().numpy())
subjects.remove(padding_value)


@interact
def get_outputs_of_patient(patient=subjects):
    global interpreter, id_column_idx, ts_column_idx
    # Find the sequence length of the current patient
    seq_len = interpreter.seq_len_dict[patient]
    # Get the data corresponding to the current patient
    data = interpreter.test_data[interpreter.test_data[:, :, 0] == patient]
    # Add a third dimension for the data to be readable by the model
    data = data.unsqueeze(0)
    # Remove identifier columns from the data
    data = du.deep_learning.remove_tensor_column(data, [id_column_idx, ts_column_idx], inplace=True)
    # Run the model on the patient's data
    new_output = model(data[:, :seq_len, :])
    return new_output


# ### Feature importance

# Calculate the feature importance scores (through SHAP values):

if ml_core == 'deep learning':
    feat_scores = interpreter.interpret_model(test_data=all_features,
                                              test_labels=all_labels,
                                              instance_importance=False, 
                                              feature_importance='shap')
elif model_class == 'XGBoost':
    feat_scores = interpreter.shap_values(all_features)


# Get the expected value:

if ml_core == 'deep learning':
    expected_value = interpreter.explainer.expected_value[0]
else:
    expected_value = interpreter.expected_value
print(f'Expected value: {expected_value}')

# Test if the SHAP values match the model output:

feat_scores.shape

if ml_core == 'deep learning':
    feature_columns.remove('subject_id')
    feature_columns.remove('ts')

interpreter.test_data

(all_features[:2] == interpreter.test_data).all()

flat_labels = interpreter.test_labels.flatten()
no_pad_mask = flat_labels != padding_value
no_pad_mask

output_0, hidden_state_0 = model(interpreter.test_data[0, 0, 2:].unsqueeze(0).unsqueeze(0), get_hidden_state=True)
output_0
hidden_state_0

output_1, hidden_state_1 = model(interpreter.test_data[0, 1, 2:].unsqueeze(0).unsqueeze(0), get_hidden_state=True,
                                 hidden_state=hidden_state_0)
output_1
hidden_state_1

output_test, _ = model(interpreter.test_data[0, :6, 2:].unsqueeze(0), get_hidden_state=True)
output_test

seq_lengths = du.search_explore.find_seq_len(interpreter.test_labels, padding_value)
seq_lengths

ref_output = model(interpreter.test_data[:, :, 2:], seq_lengths=seq_lengths).squeeze().detach().numpy()
ref_output = ref_output[no_pad_mask]
ref_output

shap_output = np.sum(feat_scores, axis=2) + expected_value
shap_output = shap_output.flatten()
shap_output = shap_output[no_pad_mask]
shap_output

ref_output.shape

shap_output.shape

all(shap_output == ref_output)

ref_output_s = pd.Series(ref_output)

shap_output_s = pd.Series(shap_output)

subjects = list(interpreter.test_data[:, :, 0].unique().int().numpy())
subjects.remove(padding_value)
subjects

# Get an overview of the important features and model output for the current patient:

ALS_df[ALS_df.subject_id.isin(subjects)].reset_index().drop(columns='index').assign(real_output=ref_output_s) \
                                                                            .assign(shap_output=shap_output_s)

ref_output_s == shap_output_s

# Test some interpretation plots:

du.visualization.shap_summary_plot(feat_scores, model_feature_columns, max_display=15)

if ml_core == 'deep learning':
    du.visualization.shap_waterfall_plot(expected_value, feat_scores[0, 2, :], 
                                         all_features[0, 2, 2:], model_feature_columns)
else:
    du.visualization.shap_waterfall_plot(expected_value, feat_scores[2, :], 
                                         all_features[2, :], model_feature_columns)

if ml_core == 'deep learning':
    du.visualization.shap_waterfall_plot(expected_value, feat_scores[0, 2, :], 
                                         denorm_data[0, 2, 2:], model_feature_columns)
else:
    du.visualization.shap_waterfall_plot(expected_value, feat_scores[2, :], 
                                         denorm_data[2, :], model_feature_columns)

# ## Saving a dataframe with the resulting SHAP values

if ml_core == 'deep learning':
    data_n_shap_df = interpreter.shap_values_df()
else:
    # Join the original data and the features' SHAP values
    data_n_shap = np.concatenate([all_features, all_labels.reshape(-1, 1), feat_scores], axis=1)
    # Reshape into a 2D format
    data_n_shap = data_n_shap.reshape(-1, data_n_shap.shape[-1])
    # Remove padding samples
    data_n_shap = data_n_shap[[padding_value not in row for row in data_n_shap]]
    # Define the column names list
    shap_column_names = [f'{feature}_shap' for feature in feature_columns]
    column_names = ([id_column] + [ts_column] + feature_columns
                    + [label_column] + shap_column_names)
    # Create the dataframe
    data_n_shap_df = pd.DataFrame(data=data_n_shap, columns=column_names)
data_n_shap_df.head()


# Save the data:

data_n_shap_file_name = f'fcul_als_with_shap_for_{model_filename.split(".pth")[0]}'
data_n_shap_file_name

data_n_shap_df.to_csv(f'{data_n_shap_path}fcul_als_with_shap_for_{model_filename}.csv')


