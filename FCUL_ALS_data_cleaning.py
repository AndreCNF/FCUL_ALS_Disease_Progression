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

# # FCUL ALS Data Cleaning
#
# Exploring the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.
#
# The main goal of this notebook is to prepare a single CSV document that contains all the relevant data to be used when training a machine learning model that predicts disease progression, filtering useless columns and performing imputation.

# + {"colab_type": "text", "id": "KOdmFzXqF7nq", "cell_type": "markdown"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import pandas as pd              # Pandas to handle the data in dataframes
import re                        # re to do regex searches in string data
import plotly                    # Plotly for interactive and pretty plots
import plotly.graph_objs as go
from datetime import datetime    # datetime to use proper date and time formats
import os                        # os handles directory/workspace changes
import numpy as np               # NumPy to handle numeric and NaN operations
from tqdm import tqdm_notebook   # tqdm allows to track code execution progress
import numbers                   # numbers allows to check if data is numeric
import torch                     # PyTorch to create and apply deep learning models
from torch.utils.data.sampler import SubsetRandomSampler
import utils                     # Contains auxiliary functions

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

# ## Reading the data

ALS_proc_df = pd.read_csv(f'{data_path}dataWithoutDunnoNIV.csv')
ALS_proc_df.head()

# ## Renaming columns

ALS_proc_df.rename(columns={'REF': 'subject_id'}, inplace=True)
ALS_proc_df.head()

# ## Deleting unused columns

ALS_proc_df.columns

ALS_proc_df.drop(columns=['NIV_DATE', 'firstDate', 'lastDate', 'medianDate', 
                          'SNIP', 'CervicalFlex', 'CervicalExt'], inplace=True)
ALS_proc_df.head()

# ## Getting discrete timestamps
#
# Creating a index for each patient that serves as a discrete timestamp, starting at 0 in their first clinical visit and ending at the length of their time series (-1).

ALS_proc_df['ts'] = ALS_proc_df.groupby('subject_id').cumcount()
ALS_proc_df.head(10)

# ## Cleaning categorical columns
#
# Combining redundant values and one hot encoding categorical features.

# Fixing a bug in the "1R" column
ALS_proc_df['1R'] = ALS_proc_df['1R'].replace(to_replace='\\1', value=1).astype('float64')

ALS_proc_df = utils.one_hot_encoding_dataframe(ALS_proc_df, columns=['El Escorial reviewed criteria',
                                                                     'Onset form',
                                                                     'UMN vs LMN',
                                                                     'C9orf72'])
ALS_proc_df.head()

# Reduxing the UMN vs LMN columns into just 2 clear columns:

ALS_proc_df.rename(columns={'UMN vs LMN_lmn': 'LMN',
                            'UMN vs LMN_umn': 'UMN',
                            'UMN vs LMN_nan': 'UMN_vs_LMN_unknown'}, inplace=True)
ALS_proc_df.head()

# Activate both UMN and LMN features if the "both" value is 1
ALS_proc_df.LMN = ALS_proc_df.apply(lambda df: 1 if df['UMN vs LMN_both'] == 1 or df['LMN'] == 1 else 0, axis=1)
ALS_proc_df.UMN = ALS_proc_df.apply(lambda df: 1 if df['UMN vs LMN_both'] == 1 or df['UMN'] == 1 else 0, axis=1)

# Drop the "both" column as it's redundant
ALS_proc_df.drop(columns='UMN vs LMN_both', inplace=True)

ALS_proc_df.head()

len(ALS_proc_df[(ALS_proc_df.UMN == 1) & (ALS_proc_df.LMN == 1)])

# **Comment:** The previous length matches the number found on the value counts of the original dataframe, corresponding to the value "both".

# ## Standardize all column names to be lower case and without spaces

ALS_proc_df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in ALS_proc_df.columns]

ALS_proc_df.head()

# ## Imputation and removal of incomplete data

# +
# [TODO] Forward fill the values of each column and remove rows that still have missing values 
# (or replace those remaining missing values with the feature's mean or do a backward fill)
# -

# ## Normalizing continuous values

# +
# [TODO] Normalize continuous features to be within a range from 0 to 1 or from -1 to 1
