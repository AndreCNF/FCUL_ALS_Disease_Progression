# -*- coding: utf-8 -*-
# # FCUL ALS Performance Testing
# ---
#
# Testing the models trained on the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.

# ## Importing the necessary packages

import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
import torch                               # PyTorch to create and apply deep learning models
import xgboost as xgb                      # Gradient boosting trees models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle                              # Save python objects in files
import yaml                                # Save and load YAML files
from ipywidgets import interact            # Display selectors and sliders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score

import pixiedust                           # Debugging in Jupyter Notebook cells

# Path to the parquet dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/cleaned/'
# Path to the code files
project_path = 'GitHub/FCUL_ALS_Disease_Progression/'
# Path to the models
models_path = f'{project_path}models/'
# Path to the metrics
metrics_path = f'{project_path}metrics/'

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

# Dataloaders parameters:

test_train_ratio = 0.25                    # Percentage of the data which will be used as a test set
validation_ratio = 0.1                     # Percentage of the data from the training set which is used for validation purposes
batch_size = 32                            # Number of unit stays in a mini batch

# Decide if we are just going to test the best model or do a comparison with all similar trained models:

test_mode = None                           # Sets if testing an individual model or multiple, similar ones
@interact
def set_test_mode(test=['one', 'aggregate']):
    global test_mode
    test_mode = test


# Model to test:

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
random_seed_1 = None                       # Model random seeds
random_seed_2 = None
random_seed_3 = None
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
                                 'MF1-LSTM with embedding layer',
                                 'MF1-LSTM',
                                 'MF2-LSTM with embedding layer',
                                 'MF2-LSTM',
                                 'TLSTM with embedding layer',
                                 'TLSTM',
                                 'XGBoost',
                                 'Logistic regression',
                                 'SVM']):
    global model_filename, model_class, model, model2, model3, dataset_mode, ml_core, use_delta_ts 
    global time_window_days, is_custom, test_mode, random_seed_1, random_seed_2, random_seed_3
    if model_name == 'Bidirectional LSTM with embedding layer and delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_bidir_pre_embedded_delta_ts_90dayswindow_0.3705valloss_08_07_2020_04_04.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'lstm_bidir_pre_embedded_delta_ts_90dayswindow_0.3674valloss_08_07_2020_03_59.pth'
            model_filename3 = 'lstm_bidir_pre_embedded_delta_ts_90dayswindow_0.3481valloss_06_07_2020_04_15.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 100
            random_seed_3 = 42
        # Set the main random seed
        random_seed_1 = 0
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional LSTM with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_bidir_pre_embedded_90dayswindow_0.2490valloss_06_07_2020_03_47.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'lstm_bidir_pre_embedded_90dayswindow_0.3652valloss_08_07_2020_03_27.pth'
            model_filename3 = 'lstm_bidir_pre_embedded_90dayswindow_0.3994valloss_08_07_2020_03_33.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional LSTM with delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3809valloss_06_07_2020_04_08.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'lstm_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3784valloss_08_07_2020_04_14.pth'
            model_filename3 = 'lstm_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3603valloss_08_07_2020_04_17.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional LSTM':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_bidir_one_hot_encoded_90dayswindow_0.4497valloss_08_07_2020_04_31.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'lstm_bidir_one_hot_encoded_90dayswindow_0.4598valloss_06_07_2020_03_39.pth'
            model_filename3 = 'lstm_bidir_one_hot_encoded_90dayswindow_0.3688valloss_08_07_2020_04_36.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 42
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 0
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'LSTM with embedding layer and delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_pre_embedded_delta_ts_90dayswindow_0.4771valloss_06_07_2020_03_55.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'lstm_pre_embedded_delta_ts_90dayswindow_0.5071valloss_21_08_2020_05_03.pth'
            model_filename3 = 'lstm_pre_embedded_delta_ts_90dayswindow_0.5712valloss_21_08_2020_05_00.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'LSTM with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_pre_embedded_90dayswindow_0.5898valloss_06_07_2020_03_21.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'lstm_pre_embedded_90dayswindow_0.5205valloss_21_08_2020_04_47.pth'
            model_filename3 = 'lstm_pre_embedded_90dayswindow_0.5186valloss_21_08_2020_04_44.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'LSTM with delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_one_hot_encoded_delta_ts_90dayswindow_0.5178valloss_06_07_2020_04_02.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'lstm_one_hot_encoded_delta_ts_90dayswindow_0.5106valloss_21_08_2020_04_53.pth'
            model_filename3 = 'lstm_one_hot_encoded_delta_ts_90dayswindow_0.5139valloss_21_08_2020_04_56.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'LSTM':
        # Set the model file and class names, then load the model
        model_filename = 'lstm_one_hot_encoded_90dayswindow_0.4363valloss_06_07_2020_03_28.pth'
        model_class = 'VanillaLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'lstm_one_hot_encoded_90dayswindow_0.5125valloss_08_07_2020_04_41.pth'
            model_filename3 = 'lstm_one_hot_encoded_90dayswindow_0.5232valloss_08_07_2020_04_44.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 100
            random_seed_3 = 0
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional RNN with embedding layer and delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_bidir_pre_embedded_delta_ts_90dayswindow_0.3059valloss_06_07_2020_03_10.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'rnn_bidir_pre_embedded_delta_ts_90dayswindow_0.3579valloss_08_07_2020_03_55.pth'
            model_filename3 = 'rnn_bidir_pre_embedded_delta_ts_90dayswindow_0.4249valloss_08_07_2020_03_49.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 100
            random_seed_3 = 0
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional RNN with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_bidir_pre_embedded_90dayswindow_0.4005valloss_08_07_2020_03_43.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'rnn_bidir_pre_embedded_90dayswindow_0.4241valloss_05_07_2020_13_05.pth'
            model_filename3 = 'rnn_bidir_pre_embedded_90dayswindow_0.4020valloss_08_07_2020_03_40.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 42
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 0
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional RNN with delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3631valloss_08_07_2020_04_21.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'rnn_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3907valloss_08_07_2020_04_24.pth'
            model_filename3 = 'rnn_bidir_one_hot_encoded_delta_ts_90dayswindow_0.3510valloss_06_07_2020_02_59.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 42
        # Set the main random seed
        random_seed_1 = 100
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'Bidirectional RNN':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_bidir_one_hot_encoded_90dayswindow_0.3713valloss_08_07_2020_04_49.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'rnn_bidir_one_hot_encoded_90dayswindow_0.3610valloss_08_07_2020_04_46.pth'
            model_filename3 = 'rnn_bidir_one_hot_encoded_90dayswindow_0.4241valloss_03_07_2020_17_40.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 42
        # Set the main random seed
        random_seed_1 = 100
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'RNN with embedding layer and delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_pre_embedded_delta_ts_90dayswindow_0.5602valloss_06_07_2020_02_50.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'rnn_pre_embedded_delta_ts_90dayswindow_0.5267valloss_21_08_2020_04_19.pth'
            model_filename3 = 'rnn_pre_embedded_delta_ts_90dayswindow_0.5393valloss_21_08_2020_04_15.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'RNN with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_with_embedding_90dayswindow_0.5569valloss_30_06_2020_17_04.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'rnn_pre_embedded_90dayswindow_0.5238valloss_21_08_2020_04_39.pth'
            model_filename3 = 'rnn_pre_embedded_90dayswindow_0.5335valloss_21_08_2020_04_41.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'RNN with delta_ts':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_one_hot_encoded_delta_ts_90dayswindow_0.4275valloss_06_07_2020_02_55.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'rnn_one_hot_encoded_delta_ts_90dayswindow_0.5354valloss_21_08_2020_04_24.pth'
            model_filename3 = 'rnn_one_hot_encoded_delta_ts_90dayswindow_0.5364valloss_21_08_2020_04_28.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'RNN':
        # Set the model file and class names, then load the model
        model_filename = 'rnn_one_hot_encoded_90dayswindow_0.5497valloss_30_06_2020_18_25.pth'
        model_class = 'VanillaRNN'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'rnn_one_hot_encoded_90dayswindow_0.5445valloss_21_08_2020_04_34.pth'
            model_filename3 = 'rnn_one_hot_encoded_90dayswindow_0.5409valloss_21_08_2020_04_30.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as not a custom model
        is_custom = False
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'MF1-LSTM with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'mf1lstm_pre_embedded_90dayswindow_0.6516valloss_07_07_2020_03_35.pth'
        model_class = 'MF1LSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'mf1lstm_pre_embedded_90dayswindow_0.6351valloss_21_08_2020_15_53.pth'
            model_filename3 = 'mf1lstm_pre_embedded_90dayswindow_0.6449valloss_21_08_2020_15_27.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = True
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'MF1-LSTM':
        # Set the model file and class names, then load the model
        model_filename = 'mf1lstm_one_hot_encoded_90dayswindow_0.6009valloss_07_07_2020_03_46.pth'
        model_class = 'MF1LSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'mf1lstm_one_hot_encoded_90dayswindow_0.6135valloss_21_08_2020_16_09.pth'
            model_filename3 = 'mf1lstm_one_hot_encoded_90dayswindow_0.6200valloss_21_08_2020_16_16.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'normalized'
        # Set it as a custom model
        is_custom = True
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'MF2-LSTM with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'mf2lstm_pre_embedded_90dayswindow_0.6388valloss_07_07_2020_03_54.pth'
        model_class = 'MF2LSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'mf2lstm_pre_embedded_90dayswindow_0.6341valloss_21_08_2020_16_40.pth'
            model_filename3 = 'mf2lstm_pre_embedded_90dayswindow_0.6405valloss_21_08_2020_17_56.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = True
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'MF2-LSTM':
        # Set the model file and class names, then load the model
        model_filename = 'mf2lstm_one_hot_encoded_90dayswindow_0.5918valloss_07_07_2020_03_58.pth'
        model_class = 'MF2LSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'mf2lstm_one_hot_encoded_90dayswindow_0.6145valloss_21_08_2020_16_33.pth'
            model_filename3 = 'mf2lstm_one_hot_encoded_90dayswindow_0.6200valloss_21_08_2020_16_25.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = 'raw'
        # Set it as a custom model
        is_custom = True
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'TLSTM with embedding layer':
        # Set the model file and class names, then load the model
        model_filename = 'tlstm_pre_embedded_90dayswindow_0.6503valloss_07_07_2020_03_03'
        model_class = 'TLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'tlstm_pre_embedded_90dayswindow_0.6173valloss_21_08_2020_15_12.pth'
            model_filename3 = 'tlstm_pre_embedded_90dayswindow_0.6402valloss_21_08_2020_15_19'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'pre-embedded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set it as a custom model
        is_custom = True
        # Set as a traditional ML model
        ml_core = 'deep learning'
    elif model_name == 'TLSTM':
        # Set the model file and class names, then load the model
        model_filename = 'tlstm_one_hot_encoded_90dayswindow_0.6153valloss_07_07_2020_03_13'
        model_class = 'TLSTM'
        model = du.deep_learning.load_checkpoint(f'{models_path}{model_filename}', getattr(Models, model_class))
        if test_mode == 'aggregate':
            model_filename2 = 'tlstm_one_hot_encoded_90dayswindow_0.6197valloss_21_08_2020_14_52.pth'
            model_filename3 = 'tlstm_one_hot_encoded_90dayswindow_0.6516valloss_21_08_2020_15_03.pth'
            model2 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename2}', getattr(Models, model_class))
            model3 = du.deep_learning.load_checkpoint(f'{models_path}{model_filename3}', getattr(Models, model_class))
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
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
        if test_mode == 'aggregate':
            model_filename2 = 'xgb_0.5694valloss_21_08_2020_18_03.pth'
            model_filename3 = 'xgb_0.5601valloss_21_08_2020_18_01.pth'
            model2 = xgb.XGBClassifier()
            model2.load_model(f'{models_path}{model_filename2}')
            model3 = xgb.XGBClassifier()
            model3.load_model(f'{models_path}{model_filename3}')
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set as a traditional ML model
        ml_core = 'machine learning'
    elif model_name == 'Logistic regression':
        # Set the model file and class names, then load the model
        model_filename = 'logreg_0.6210valloss_09_07_2020_02_54.pth'
        model_class = 'LogReg'
        model = joblib.load(f'{models_path}{model_filename}')
        if test_mode == 'aggregate':
            model_filename2 = 'logreg_0.5932valloss_21_08_2020_18_05.pth'
            model_filename3 = 'logreg_0.5644valloss_21_08_2020_18_07.pth'
            model2 = joblib.load(f'{models_path}{model_filename2}')
            model3 = joblib.load(f'{models_path}{model_filename3}')
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set as a traditional ML model
        ml_core = 'machine learning'
    elif model_name == 'SVM':
        # Set the model file and class names, then load the model
        model_filename = 'svm_0.9078valloss_09_07_2020_02_55.pth'
        model_class = 'SVM'
        model = joblib.load(f'{models_path}{model_filename}')
        if test_mode == 'aggregate':
            model_filename2 = 'svm_0.8402valloss_21_08_2020_18_06.pth'
            model_filename3 = 'svm_0.7443valloss_21_08_2020_18_07.pth'
            model2 = joblib.load(f'{models_path}{model_filename2}')
            model3 = joblib.load(f'{models_path}{model_filename3}')
            # Set the secondary random seed
            random_seed_2 = 0
            random_seed_3 = 100
        # Set the main random seed
        random_seed_1 = 42
        # Set the use of an embedding layer
        dataset_mode = 'one hot encoded'
        # Set the use of delta_ts
        use_delta_ts = False
        # Set as a traditional ML model
        ml_core = 'machine learning'
    print(model)
    if test_mode == 'aggregate':
        print(model2)
        print(model3)


# Set the random seed for reproducibility:

du.set_random_seed(random_seed_1)

# ## Loading the data

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
    # Normalize the time variation data
    # NOTE: When using the MF2-LSTM model, since it assumes that the time
    # variation is in days, we shouldn't normalize `delta_ts` with this model.
    ALS_df['delta_ts'] = (ALS_df['delta_ts'] - ALS_df['delta_ts'].mean()) / ALS_df['delta_ts'].std()
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

# ## Defining the dataset object

dataset = du.datasets.Time_Series_Dataset(ALS_df, data, padding_value=padding_value,
                                          label_name=label_column)

dataset.__len__()

# ## Separating into train and validation sets

# Make sure that we are using the right random seed
du.set_random_seed(random_seed_1)
(train_dataloader, val_dataloader, test_dataloader,
train_indeces, val_indeces, test_indeces) = du.machine_learning.create_train_sets(dataset,
                                                                                  test_train_ratio=test_train_ratio,
                                                                                  validation_ratio=validation_ratio,
                                                                                  batch_size=batch_size,
                                                                                  get_indices=True)
if test_mode == 'aggregate':
    # Temporarily change the random seed to the one used in the second model
    du.set_random_seed(random_seed_2)
    # Create a separate data division
    (train_dataloader, val_dataloader, test_dataloader,
    train_indeces2, val_indeces2, test_indeces2) = du.machine_learning.create_train_sets(dataset,
                                                                                         test_train_ratio=test_train_ratio,
                                                                                         validation_ratio=validation_ratio,
                                                                                         batch_size=batch_size,
                                                                                         get_indices=True)
    # Temporarily change the random seed to the one used in the second model
    du.set_random_seed(random_seed_3)
    # Create a separate data division
    (train_dataloader, val_dataloader, test_dataloader,
    train_indeces3, val_indeces3, test_indeces3) = du.machine_learning.create_train_sets(dataset,
                                                                                         test_train_ratio=test_train_ratio,
                                                                                         validation_ratio=validation_ratio,
                                                                                         batch_size=batch_size,
                                                                                         get_indices=True)

# Get the full arrays of each set
train_features, train_labels = dataset.X[train_indeces], dataset.y[train_indeces]
val_features, val_labels = dataset.X[val_indeces], dataset.y[val_indeces]
test_features, test_labels = dataset.X[test_indeces], dataset.y[test_indeces]
all_features, all_labels = dataset.X, dataset.y
if test_mode == 'aggregate':
    train_features2, train_labels2 = dataset.X[train_indeces2], dataset.y[train_indeces2]
    val_features2, val_labels2 = dataset.X[val_indeces2], dataset.y[val_indeces2]
    test_features2, test_labels2 = dataset.X[test_indeces2], dataset.y[test_indeces2]
    train_features3, train_labels3 = dataset.X[train_indeces3], dataset.y[train_indeces3]
    val_features3, val_labels3 = dataset.X[val_indeces3], dataset.y[val_indeces3]
    test_features3, test_labels3 = dataset.X[test_indeces3], dataset.y[test_indeces3]

# Ignore the dataloaders, we only care about the full arrays when using scikit-learn or XGBoost
del train_dataloader
del val_dataloader
del test_dataloader

if ml_core == 'machine learning':
    # Remove the ID and timestamp columns from the data arrays
    train_features = train_features[:, :, 2:]
    val_features = val_features[:, :, 2:]
    test_features = test_features[:, :, 2:]
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
    if test_mode == 'aggregate':
        # Model 2
        # Remove the ID and timestamp columns from the data arrays
        train_features2 = train_features2[:, :, 2:]
        val_features2 = val_features2[:, :, 2:]
        test_features2 = test_features2[:, :, 2:]
        # Reshape the data into a 2D format
        train_features2 = train_features2.reshape(-1, train_features2.shape[-1])
        val_features2 = val_features2.reshape(-1, val_features2.shape[-1])
        test_features2 = test_features2.reshape(-1, test_features2.shape[-1])
        train_labels2 = train_labels2.reshape(-1)
        val_labels2 = val_labels2.reshape(-1)
        test_labels2 = test_labels2.reshape(-1)
        # Remove padding samples from the data
        train_features2 = train_features2[[padding_value not in row for row in train_features2]]
        val_features2 = val_features2[[padding_value not in row for row in val_features2]]
        test_features2 = test_features2[[padding_value not in row for row in test_features2]]
        train_labels2 = train_labels2[[padding_value not in row for row in train_labels2]]
        val_labels2 = val_labels2[[padding_value not in row for row in val_labels2]]
        test_labels2 = test_labels2[[padding_value not in row for row in test_labels2]]
        # Convert from PyTorch tensor to NumPy array
        train_features2 = train_features2.numpy()
        val_features2 = val_features2.numpy()
        test_features2 = test_features2.numpy()
        train_labels2 = train_labels2.numpy()
        val_labels2 = val_labels2.numpy()
        test_labels2 = test_labels2.numpy()
        # Model 3
        # Remove the ID and timestamp columns from the data arrays
        train_features3 = train_features3[:, :, 2:]
        val_features3 = val_features3[:, :, 2:]
        test_features3 = test_features3[:, :, 2:]
        # Reshape the data into a 2D format
        train_features3 = train_features3.reshape(-1, train_features3.shape[-1])
        val_features3 = val_features3.reshape(-1, val_features3.shape[-1])
        test_features3 = test_features3.reshape(-1, test_features3.shape[-1])
        train_labels3 = train_labels3.reshape(-1)
        val_labels3 = val_labels3.reshape(-1)
        test_labels3 = test_labels3.reshape(-1)
        # Remove padding samples from the data
        train_features3 = train_features3[[padding_value not in row for row in train_features3]]
        val_features3 = val_features3[[padding_value not in row for row in val_features3]]
        test_features3 = test_features3[[padding_value not in row for row in test_features3]]
        train_labels3 = train_labels3[[padding_value not in row for row in train_labels3]]
        val_labels3 = val_labels3[[padding_value not in row for row in val_labels3]]
        test_labels3 = test_labels3[[padding_value not in row for row in test_labels3]]
        # Convert from PyTorch tensor to NumPy array
        train_features3 = train_features3.numpy()
        val_features3 = val_features3.numpy()
        test_features3 = test_features3.numpy()
        train_labels3 = train_labels3.numpy()
        val_labels3 = val_labels3.numpy()
        test_labels3 = test_labels3.numpy()

train_features

train_features.shape

# ## Testing the model

# ### Training set

if ml_core == 'deep learning':
    _, train_metrics = du.deep_learning.model_inference(model, data=(train_features, train_labels), 
                                                        metrics=['loss', 'accuracy', 'precision',
                                                                'recall', 'F1', 'AUC'], 
                                                        model_type='multivariate_rnn', is_custom=is_custom, 
                                                        padding_value=padding_value)
else:
    train_metrics = dict()
    train_pred_proba = model.predict_proba(train_features)
    train_pred = model.predict(train_features)
    train_metrics['AUC'] = roc_auc_score(train_labels, train_pred_proba[:, 1])
    train_metrics['F1'] = f1_score(train_labels, train_pred, average='weighted')
    train_metrics['accuracy'] = accuracy_score(train_labels, train_pred)
    train_metrics['loss'] = log_loss(train_labels, train_pred_proba)
    train_metrics['precision'] = precision_score(train_labels, train_pred)
    train_metrics['recall'] = recall_score(train_labels, train_pred)
train_metrics


# If doing an aggregate test, do inference on the other similar models and combine the metrics in mean and standard deviation:

if test_mode == 'aggregate':
    if ml_core == 'deep learning':
        _, train_metrics2 = du.deep_learning.model_inference(model2, data=(train_features2, train_labels2), 
                                                            metrics=['loss', 'accuracy', 'precision',
                                                                    'recall', 'F1', 'AUC'], 
                                                            model_type='multivariate_rnn', is_custom=is_custom, 
                                                            padding_value=padding_value)
        _, train_metrics3 = du.deep_learning.model_inference(model3, data=(train_features3, train_labels3), 
                                                            metrics=['loss', 'accuracy', 'precision',
                                                                    'recall', 'F1', 'AUC'], 
                                                            model_type='multivariate_rnn', is_custom=is_custom, 
                                                            padding_value=padding_value)
    else:
        # Model 2
        train_metrics2 = dict()
        train_pred_proba2 = model.predict_proba(train_features2)
        train_pred2 = model.predict(train_features2)
        train_metrics2['AUC'] = roc_auc_score(train_labels2, train_pred_proba2[:, 1])
        train_metrics2['F1'] = f1_score(train_labels2, train_pred2, average='weighted')
        train_metrics2['accuracy'] = accuracy_score(train_labels2, train_pred2)
        train_metrics2['loss'] = log_loss(train_labels2, train_pred_proba2)
        train_metrics2['precision'] = precision_score(train_labels2, train_pred2)
        train_metrics2['recall'] = recall_score(train_labels2, train_pred2)
        # Model 3
        train_metrics3 = dict()
        train_pred_proba3 = model.predict_proba(train_features3)
        train_pred3 = model.predict(train_features3)
        train_metrics3['AUC'] = roc_auc_score(train_labels3, train_pred_proba3[:, 1])
        train_metrics3['F1'] = f1_score(train_labels3, train_pred3, average='weighted')
        train_metrics3['accuracy'] = accuracy_score(train_labels3, train_pred3)
        train_metrics3['loss'] = log_loss(train_labels3, train_pred_proba3)
        train_metrics3['precision'] = precision_score(train_labels3, train_pred3)
        train_metrics3['recall'] = recall_score(train_labels3, train_pred3)
    train_metrics2
    train_metrics3


if test_mode == 'aggregate':
    train_metrics_agg = dict()
    for key in train_metrics.keys():
        # Create the current key
        train_metrics_agg[key] = dict()
        # Add the mean
        train_metrics_agg[key]['mean'] = np.mean([train_metrics[key], train_metrics2[key], train_metrics3[key]]).item()
        # Add the standard deviation
        train_metrics_agg[key]['std'] = np.std([train_metrics[key], train_metrics2[key], train_metrics3[key]]).item()
    train_metrics_agg

# ### Validation set

if ml_core == 'deep learning':
    _, val_metrics = du.deep_learning.model_inference(model, data=(val_features, val_labels), 
                                                        metrics=['loss', 'accuracy', 'precision',
                                                                'recall', 'F1', 'AUC'], 
                                                        model_type='multivariate_rnn', is_custom=is_custom, 
                                                        padding_value=padding_value)
else:
    val_metrics = dict()
    val_pred_proba = model.predict_proba(val_features)
    val_pred = model.predict(val_features)
    val_metrics['AUC'] = roc_auc_score(val_labels, val_pred_proba[:, 1])
    val_metrics['F1'] = f1_score(val_labels, val_pred, average='weighted')
    val_metrics['accuracy'] = accuracy_score(val_labels, val_pred)
    val_metrics['loss'] = log_loss(val_labels, val_pred_proba)
    val_metrics['precision'] = precision_score(val_labels, val_pred)
    val_metrics['recall'] = recall_score(val_labels, val_pred)
val_metrics

# If doing an aggregate test, do inference on the other similar models and combine the metrics in mean and standard deviation:

if test_mode == 'aggregate':
    if ml_core == 'deep learning':
        _, val_metrics2 = du.deep_learning.model_inference(model2, data=(val_features2, val_labels2), 
                                                           metrics=['loss', 'accuracy', 'precision',
                                                                   'recall', 'F1', 'AUC'], 
                                                           model_type='multivariate_rnn', is_custom=is_custom, 
                                                           padding_value=padding_value)
        _, val_metrics3 = du.deep_learning.model_inference(model3, data=(val_features3, val_labels3), 
                                                           metrics=['loss', 'accuracy', 'precision',
                                                                   'recall', 'F1', 'AUC'], 
                                                           model_type='multivariate_rnn', is_custom=is_custom, 
                                                           padding_value=padding_value)
    else:
        # Model 2
        val_metrics2 = dict()
        val_pred_proba2 = model.predict_proba(val_features2)
        val_pred2 = model.predict(val_features2)
        val_metrics2['AUC'] = roc_auc_score(val_labels2, val_pred_proba2[:, 1])
        val_metrics2['F1'] = f1_score(val_labels2, val_pred2, average='weighted')
        val_metrics2['accuracy'] = accuracy_score(val_labels2, val_pred2)
        val_metrics2['loss'] = log_loss(val_labels2, val_pred_proba2)
        val_metrics2['precision'] = precision_score(val_labels2, val_pred2)
        val_metrics2['recall'] = recall_score(val_labels2, val_pred2)
        # Model 3
        val_metrics3 = dict()
        val_pred_proba3 = model.predict_proba(val_features3)
        val_pred3 = model.predict(val_features3)
        val_metrics3['AUC'] = roc_auc_score(val_labels3, val_pred_proba3[:, 1])
        val_metrics3['F1'] = f1_score(val_labels3, val_pred3, average='weighted')
        val_metrics3['accuracy'] = accuracy_score(val_labels3, val_pred3)
        val_metrics3['loss'] = log_loss(val_labels3, val_pred_proba3)
        val_metrics3['precision'] = precision_score(val_labels3, val_pred3)
        val_metrics3['recall'] = recall_score(val_labels3, val_pred3)
    val_metrics2
    val_metrics3


if test_mode == 'aggregate':
    val_metrics_agg = dict()
    for key in val_metrics.keys():
        # Create the current key
        val_metrics_agg[key] = dict()
        # Add the mean
        val_metrics_agg[key]['mean'] = np.mean([val_metrics[key], val_metrics2[key], val_metrics3[key]]).item()
        # Add the standard deviation
        val_metrics_agg[key]['std'] = np.std([val_metrics[key], val_metrics2[key], val_metrics3[key]]).item()
    val_metrics_agg

# ### Testing set

if ml_core == 'deep learning':
    _, test_metrics = du.deep_learning.model_inference(model, data=(test_features, test_labels), 
                                                        metrics=['loss', 'accuracy', 'precision',
                                                                'recall', 'F1', 'AUC'], 
                                                        model_type='multivariate_rnn', is_custom=is_custom, 
                                                        padding_value=padding_value)
else:
    test_metrics = dict()
    test_pred_proba = model.predict_proba(test_features)
    test_pred = model.predict(test_features)
    test_metrics['AUC'] = roc_auc_score(test_labels, test_pred_proba[:, 1])
    test_metrics['F1'] = f1_score(test_labels, test_pred, average='weighted')
    test_metrics['accuracy'] = accuracy_score(test_labels, test_pred)
    test_metrics['loss'] = log_loss(test_labels, test_pred_proba)
    test_metrics['precision'] = precision_score(test_labels, test_pred)
    test_metrics['recall'] = recall_score(test_labels, test_pred)
test_metrics

# If doing an aggregate test, do inference on the other similar models and combine the metrics in mean and standard deviation:

if test_mode == 'aggregate':
    if ml_core == 'deep learning':
        _, test_metrics2 = du.deep_learning.model_inference(model2, data=(test_features2, test_labels2), 
                                                            metrics=['loss', 'accuracy', 'precision',
                                                                    'recall', 'F1', 'AUC'], 
                                                            model_type='multivariate_rnn', is_custom=is_custom, 
                                                            padding_value=padding_value)
        _, test_metrics3 = du.deep_learning.model_inference(model3, data=(test_features3, test_labels3), 
                                                            metrics=['loss', 'accuracy', 'precision',
                                                                    'recall', 'F1', 'AUC'], 
                                                            model_type='multivariate_rnn', is_custom=is_custom, 
                                                            padding_value=padding_value)
    else:
        # Model 2
        test_metrics2 = dict()
        test_pred_proba2 = model.predict_proba(test_features2)
        test_pred2 = model.predict(test_features2)
        test_metrics2['AUC'] = roc_auc_score(test_labels2, test_pred_proba2[:, 1])
        test_metrics2['F1'] = f1_score(test_labels2, test_pred2, average='weighted')
        test_metrics2['accuracy'] = accuracy_score(test_labels2, test_pred2)
        test_metrics2['loss'] = log_loss(test_labels2, test_pred_proba2)
        test_metrics2['precision'] = precision_score(test_labels2, test_pred2)
        test_metrics2['recall'] = recall_score(test_labels2, test_pred2)
        # Model 3
        test_metrics3 = dict()
        test_pred_proba3 = model.predict_proba(test_features3)
        test_pred3 = model.predict(test_features3)
        test_metrics3['AUC'] = roc_auc_score(test_labels3, test_pred_proba3[:, 1])
        test_metrics3['F1'] = f1_score(test_labels3, test_pred3, average='weighted')
        test_metrics3['accuracy'] = accuracy_score(test_labels3, test_pred3)
        test_metrics3['loss'] = log_loss(test_labels3, test_pred_proba3)
        test_metrics3['precision'] = precision_score(test_labels3, test_pred3)
        test_metrics3['recall'] = recall_score(test_labels3, test_pred3)
    test_metrics2
    test_metrics3


if test_mode == 'aggregate':
    test_metrics_agg = dict()
    for key in test_metrics.keys():
        # Create the current key
        test_metrics_agg[key] = dict()
        # Add the mean
        test_metrics_agg[key]['mean'] = np.mean([test_metrics[key], test_metrics2[key], test_metrics3[key]]).item()
        # Add the standard deviation
        test_metrics_agg[key]['std'] = np.std([test_metrics[key], test_metrics2[key], test_metrics3[key]]).item()
    test_metrics_agg

# ## Saving a dictionary with all the metrics

# Join all the sets' metrics:

metrics = dict()
if test_mode == 'aggregate':
    metrics['train'] = train_metrics_agg
    metrics['val'] = val_metrics_agg
    metrics['test'] = test_metrics_agg
else:
    metrics['train'] = train_metrics
    metrics['val'] = val_metrics
    metrics['test'] = test_metrics
metrics

# Save in a YAML file:

model_file_name_no_ext = model_filename.split('.pth')[0]

if test_mode == 'aggregate':
    file_path = f'{metrics_path}aggregate/'
    model_file_name_yml = model_file_name_no_ext.split('.')[0][:-2]
else:
    file_path = f'{metrics_path}individual_models/'
    model_file_name_yml = model_file_name_no_ext
model_file_name_yml

metrics_stream = open(f'{file_path}{model_file_name_yml}.yml', 'w')
yaml.dump(metrics, metrics_stream, default_flow_style=False)


