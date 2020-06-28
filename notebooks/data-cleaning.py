# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: fcul_als_disease_progression
#     language: python
#     name: fcul_als_disease_progression
# ---

# # FCUL ALS Data Cleaning
# ---
#
# Exploring the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.
#
# The main goal of this notebook is to prepare a single CSV document that contains all the relevant data to be used when training a machine learning model that predicts disease progression, filtering useless columns and performing imputation.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq"}
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
import data_utils as du          # Data science and machine learning relevant methods
# -

import pixiedust                 # Debugging in Jupyter Notebook cells

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the CSV dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/'

du.set_pandas_library(lib='pandas')

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

# Set the random seed for reproducibility:

du.set_random_seed(42)

# ## Setting the initial parameters

time_window_days = 90            # How many days into the future will we predict the use of NIV

# ## Reading the data

ALS_proc_df = pd.read_csv(f'{data_path}dataWithoutDunnoNIV.csv')
ALS_proc_df.head()

# ## Renaming columns

ALS_proc_df.rename(columns={'REF': 'subject_id'}, inplace=True)
ALS_proc_df.head()

# ## Creating a timestamp column
#
# Using `medianDate`, we can define a column that serves as the timestamp, which indicates how many days have gone by since the patient's first data sample.

# Convert column `medianDate` to a datetime format:

ALS_proc_df.medianDate = pd.to_datetime(ALS_proc_df.medianDate, format='%d/%m/%Y')
ALS_proc_df.medianDate

# Get the difference in days between the samples:

ALS_proc_df.medianDate = ALS_proc_df.groupby('subject_id').medianDate.diff()
ALS_proc_df.medianDate

# Convert to a numeric format and replace the missing values (which are the first sample in each time series) with 0:

ALS_proc_df.medianDate = ALS_proc_df.medianDate / np.timedelta64(1, 'D')
ALS_proc_df.medianDate = ALS_proc_df.medianDate.fillna(0)
ALS_proc_df.medianDate

# Rename to `ts`:

ALS_proc_df.rename(columns={'medianDate': 'ts'}, inplace=True)
ALS_proc_df.head()

ALS_proc_df.ts.describe()

# ## Deleting unused columns
#
# Removing kind of useless columns ('NIV_DATE', 'firstDate', 'lastDate'), ones with too many missing values ('SNIP', 'CervicalFlex', 'CervicalExt') and ones that would give away the labels ('ALS-FRS', 'ALS-FRS-R', 'ALS-FRSb', 'ALS-FRSsUL', 'ALS-FRSsLL', 'ALS-FRSr').

ALS_proc_df.columns

ALS_proc_df.drop(columns=['NIV_DATE', 'firstDate', 'lastDate', 'SNIP', 
                          'CervicalFlex', 'CervicalExt', 'ALS-FRS',
                          'ALS-FRS-R', 'ALS-FRSb', 'ALS-FRSsUL', 
                          'ALS-FRSsLL', 'ALS-FRSr'], inplace=True)
ALS_proc_df.head()

# ## Removing patients without enough samples to predict one time window
#
# Since we want to predict the use of NIV in the next 90 days (time window), it doesn't make any sense to include patients that don't have samples that represent at least 90 days.

ALS_proc_df.subject_id.nunique()

ALS_proc_df.groupby('subject_id').ts.count().min()

for patient in ALS_proc_df.subject_id.unique():
    subject_data = ALS_proc_df[ALS_proc_df.subject_id == patient]
    # Check if the current patient only has one clinical visit
    if subject_data.ts.max() - subject_data.ts.min() < time_window_days:
        # Remove patient's data from the dataframe
        ALS_proc_df = ALS_proc_df[ALS_proc_df.subject_id != patient]

ALS_proc_df.subject_id.nunique()

ALS_proc_df.groupby('subject_id').ts.count().min()

ALS_proc_df.groupby('subject_id').ts.count().describe()

# ## Cleaning categorical columns
#
# Combining redundant values and one hot encoding categorical features.

# Making "Gender" a proper one hot encoded column:

ALS_proc_df['Gender'] = ALS_proc_df['Gender'] - 1

# Fixing a bug in the `1R` column:

ALS_proc_df['1R'] = ALS_proc_df['1R'].replace(to_replace='\\1', value=1).astype('float64')

du.search_explore.dataframe_missing_values(ALS_proc_df)

# One hot encode the remaining categorical columns:

# + {"pixiedust": {"displayParams": {}}}
ALS_proc_df = du.data_processing.one_hot_encoding_dataframe(ALS_proc_df,
                                                            columns=['El Escorial reviewed criteria',
                                                                     'Onset form',
                                                                     'UMN vs LMN',
                                                                     'C9orf72'],
                                                            join_rows=True,
                                                            join_by=['subject_id', 'ts'],
                                                            lower_case=True, 
                                                            has_nan=True,
                                                            inplace=True)
ALS_proc_df.head()
# -

# Reduxing the UMN vs LMN columns into just 2 clear columns:

ALS_proc_df.rename(columns={'UMN vs LMN_lmn': 'LMN',
                            'UMN vs LMN_umn': 'UMN'}, inplace=True)
ALS_proc_df.head()

# Activate both UMN and LMN features if the "both" value is 1
ALS_proc_df.LMN = ALS_proc_df.apply(lambda df: 1 if df['UMN vs LMN_both'] == 1 or df['LMN'] == 1 else 0, axis=1)
ALS_proc_df.UMN = ALS_proc_df.apply(lambda df: 1 if df['UMN vs LMN_both'] == 1 or df['UMN'] == 1 else 0, axis=1)

# Drop the "both" column as it's redundant
ALS_proc_df.drop(columns='UMN vs LMN_both', inplace=True)

ALS_proc_df.head()

len(ALS_proc_df[(ALS_proc_df.UMN == 1) & (ALS_proc_df.LMN == 1)])

# **Comment:** The previous length matches the number found on the value counts of the original dataframe, corresponding to the value "both".

# Remove the redundant `C9orf72_no` column:

ALS_proc_df.columns

ALS_proc_df.drop(columns='C9orf72_no', inplace=True)
ALS_proc_df.head()

ALS_proc_df.rename(columns={'C9orf72_yes': 'C9orf72'}, inplace=True)
ALS_proc_df.head()

# ## Standardize all column names to be lower case and without spaces

ALS_proc_df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in ALS_proc_df.columns]

ALS_proc_df.head()

# ## NIV label
#
# In order to predict the use of NIV in the next 3 months, we need to create a shifted version of the "niv" column.

ALS_proc_df['niv_label'] = ALS_proc_df['niv']

ALS_proc_df.head().niv.max()


def set_niv_label_in_row(df, time_window_days=90):
    global ALS_proc_df
    # Get a list of all the timestamps in the current patient's time series
    subject_ts_list = ALS_proc_df[ALS_proc_df.subject_id == df.subject_id].ts
    try:
        # Try to find the timestamp of a sample that is equal or bigger than 
        # the current one + the desired time window
        closest_ts = subject_ts_list[subject_ts_list >= df.ts+time_window_days].iloc[0]
    except IndexError:
        # Just use the data from the subject's last sample if there are no 
        # samples in the desired time window for this subject
        closest_ts = subject_ts_list.iloc[-1]
    # Check if the patient has been on NIV anytime during the defined time window
    if closest_ts > df.ts+time_window_days:
        time_window_data = ALS_proc_df[(ALS_proc_df.subject_id == df.subject_id) 
                                       & (ALS_proc_df.ts < closest_ts)
                                       & (ALS_proc_df.ts > df.ts)]
    else:
        time_window_data = ALS_proc_df[(ALS_proc_df.subject_id == df.subject_id) 
                                       & (ALS_proc_df.ts <= closest_ts)
                                       & (ALS_proc_df.ts > df.ts)]
    if time_window_data.empty:
        # Just use the last NIV indication when it's the last sample in the subject's
        # time series or there are no other samples in the specified time window
        time_window_data = ALS_proc_df[(ALS_proc_df.subject_id == df.subject_id) 
                                       & (ALS_proc_df.ts == df.ts)]
    return time_window_data.niv.max() == 1


ALS_proc_df[['subject_id', 'ts', 'niv', 'niv_label']].head(20)

# + {"pixiedust": {"displayParams": {}}}
ALS_proc_df['niv_label'] = ALS_proc_df.apply(set_niv_label_in_row, axis=1)
# -

ALS_proc_df[['subject_id', 'ts', 'niv', 'niv_label']].head(200)

# Save a version of the dataframe without normalization
ALS_proc_df.to_csv(f'{data_path}cleaned/FCUL_ALS_cleaned_denorm.csv')

ALS_proc_df.describe().transpose()

# ## Normalizing continuous values
#
# Continuous data is normalized into z-scores, where 0 represents the mean and an absolute value of 1 corresponds to the standard deviation.

ALS_proc_df = du.data_processing.normalize_data(ALS_proc_df, id_columns=['subject_id', 'ts'])
ALS_proc_df.head()


ALS_proc_df.describe().transpose()

# ## Imputation and removal of incomplete data
#
# Starting from a last information carried forward technique, the data is initially forward filled. Next, a backward fill is done, as current data of the patient should still be a good indicator of the recent past. Finally, the remaining missing values are filled with zeroes, as it represents the average value of each given feature.

ALS_proc_df[['subject_id', 'ts', 'r', 'p1', 'p2', 'bmi', 'fvc', 'vc', 'mip', 'niv_label']].head(20)

ALS_proc_df = du.data_processing.missing_values_imputation(ALS_proc_df, method='zigzag', id_column='subject_id')
ALS_proc_df.head()

ALS_proc_df[['subject_id', 'ts', 'r', 'p1', 'p2', 'bmi', 'fvc', 'vc', 'mip', 'niv_label']].head(20)

# ## Saving the data

ALS_proc_df.to_csv(f'{data_path}cleaned/FCUL_ALS_cleaned.csv')

ALS_proc_df.head()

ALS_proc_df.columns
