# -*- coding: utf-8 -*-
# # FCUL ALS LaTeX Metrics
# ---
#
# Joining the metrics of the models trained on the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.

# ## Importing the necessary packages

import os                                  # os handles directory/workspace changes
import yaml                                # Save and load YAML files

import pixiedust                           # Debugging in Jupyter Notebook cells

# Path to the metrics
metrics_path = 'GitHub/FCUL_ALS_Disease_Progression/metrics/aggregate/'

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

# ## Loading the data

metrics_files = os.listdir(metrics_path)
try:
    metrics_files.remove('.DS_Store')
except:
    pass
metrics_files

# Create a dictionary with all the metrics:

metrics = dict()
for file_name in metrics_files:
    # Load the current metrics file
    stream = open(f'{metrics_path}{file_name}', 'r')
    model_metrics = yaml.load(stream, Loader=yaml.FullLoader)
    # Remove the extension from the name
    file_name = file_name.split('.yml')[0]
    # Define the model name which will appear in the table
    model_name = ''
    if 'bidir' in file_name:
        model_name = 'Bidirectional '
    if 'tlstm' in file_name:
        model_name += 'TLSTM'
    elif 'mf1lstm' in file_name:
        model_name += 'MF1-LSTM'
    elif 'mf2lstm' in file_name:
        model_name += 'MF2-LSTM'
    elif 'lstm' in file_name:
        model_name += 'LSTM'
    elif 'rnn' in file_name:
        model_name += 'RNN'
    elif 'xgb' in file_name:
        model_name += 'XGBoost'
    elif 'logreg' in file_name:
        model_name += 'Logistic Regression'
    elif 'svm' in file_name:
        model_name += 'SVM'
    if 'embed' in file_name:
        model_name += ', embedded'
    if 'delta_ts' in file_name:
        model_name += ', time aware'
    # Create a dictionary entry for the current model
    metrics[model_name] = dict()
    metrics[model_name]['Avg. Test AUC'] = model_metrics['test']['AUC']['mean']
    metrics[model_name]['Std. Test AUC'] = model_metrics['test']['AUC']['std']

metrics

# Convert to a dataframe:

metrics_df = pd.DataFrame(metrics)
metrics_df

# Transpose to have a row per model:

metrics_df = metrics_df.transpose()
metrics_df

metrics_df = metrics_df.sort_values('Avg. Test AUC', ascending=False)
metrics_df

# Convert to a LaTeX table:

metrics_df.to_latex()


