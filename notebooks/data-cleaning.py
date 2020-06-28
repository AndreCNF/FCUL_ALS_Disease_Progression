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
import os                        # os handles directory/workspace changes
import yaml                      # Save and load YAML files
import pandas as pd              # Pandas to handle the data in dataframes
import numpy as np               # NumPy to handle numeric and NaN operations
import data_utils as du          # Data science and machine learning relevant methods
# -

import pixiedust                 # Debugging in Jupyter Notebook cells

# Change to the scripts directory
os.chdir("../scripts/")
import utils                     # Context specific (in this case, for the ALS data) methods
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

ALS_df = pd.read_csv(f'{data_path}dataWithoutDunnoNIV.csv')
ALS_df.head()

# ## Renaming columns

ALS_df.rename(columns={'REF': 'subject_id'}, inplace=True)
ALS_df.head()

# ## Creating a timestamp column
#
# Using `medianDate`, we can define a column that serves as the timestamp, which indicates how many days have gone by since the patient's first data sample.

# Convert column `medianDate` to a datetime format:

ALS_df.medianDate = pd.to_datetime(ALS_df.medianDate, format='%d/%m/%Y')
ALS_df.medianDate

# Get the difference in days between the samples:

ALS_df.medianDate = ALS_df.groupby('subject_id').medianDate.diff()
ALS_df.medianDate

# Convert to a numeric format and replace the missing values (which are the first sample in each time series) with 0:

ALS_df.medianDate = ALS_df.medianDate / np.timedelta64(1, 'D')
ALS_df.medianDate = ALS_df.medianDate.fillna(0)
ALS_df.medianDate

# Rename to `ts`:

ALS_df.rename(columns={'medianDate': 'ts'}, inplace=True)
ALS_df.head()

ALS_df.ts.describe()

# ## Deleting unused columns
#
# Removing kind of useless columns ('NIV_DATE', 'firstDate', 'lastDate'), ones with too many missing values ('SNIP', 'CervicalFlex', 'CervicalExt') and ones that would give away the labels ('ALS-FRS', 'ALS-FRS-R', 'ALS-FRSb', 'ALS-FRSsUL', 'ALS-FRSsLL', 'ALS-FRSr').

ALS_df.columns

ALS_df.drop(columns=['NIV_DATE', 'firstDate', 'lastDate', 'SNIP',
                          'CervicalFlex', 'CervicalExt', 'ALS-FRS',
                          'ALS-FRS-R', 'ALS-FRSb', 'ALS-FRSsUL',
                          'ALS-FRSsLL', 'ALS-FRSr'], inplace=True)
ALS_df.head()

# ## Removing patients without enough samples to predict one time window
#
# Since we want to predict the use of NIV in the next 90 days (time window), it doesn't make any sense to include patients that don't have samples that represent at least 90 days.

ALS_df.subject_id.nunique()

ALS_df.groupby('subject_id').ts.count().min()

for patient in ALS_df.subject_id.unique():
    subject_data = ALS_df[ALS_df.subject_id == patient]
    # Check if the current patient only has one clinical visit
    if subject_data.ts.max() - subject_data.ts.min() < time_window_days:
        # Remove patient's data from the dataframe
        ALS_df = ALS_df[ALS_df.subject_id != patient]

ALS_df.subject_id.nunique()

ALS_df.groupby('subject_id').ts.count().min()

ALS_df.groupby('subject_id').ts.count().describe()

# ## Cleaning categorical columns
#
# Combining redundant values and one hot encoding categorical features.

# Making "Gender" a proper one hot encoded column:

ALS_df['Gender'] = ALS_df['Gender'] - 1

# Fixing a bug in the `1R` column:

ALS_df['1R'] = ALS_df['1R'].replace(to_replace='\\1', value=1).astype('float64')

du.search_explore.dataframe_missing_values(ALS_df)

# One hot encode the remaining categorical columns:

# + {"pixiedust": {"displayParams": {}}}
ALS_df, new_columns = du.data_processing.one_hot_encoding_dataframe(ALS_df,
                                                                         columns=['El Escorial reviewed criteria',
                                                                                  'Onset form',
                                                                                  'UMN vs LMN',
                                                                                  'C9orf72'],
                                                                         join_rows=True,
                                                                         join_by=['subject_id', 'ts'],
                                                                         lower_case=True,
                                                                         has_nan=True,
                                                                         get_new_column_names=True,
                                                                         inplace=True)
ALS_df.head()
# -

# Save the association between the original categorical features and the new one hot encoded columns:

categ_feat_ohe = dict()
categ_feat_ohe['El Escorial reviewed criteria'] = [ohe_col for ohe_col in new_columns
                                                   if ohe_col.startswith('El Escorial reviewed criteria')]
categ_feat_ohe

stream = open(f'{data_path}/cleaned/categ_feat_ohe.yml', 'w')
yaml.dump(categ_feat_ohe, stream, default_flow_style=False)

# Reduxing the UMN vs LMN columns into just 2 clear columns:

ALS_df.rename(columns={'UMN vs LMN_lmn': 'LMN',
                            'UMN vs LMN_umn': 'UMN'}, inplace=True)
ALS_df.head()

# Activate both UMN and LMN features if the "both" value is 1
ALS_df.LMN = ALS_df.apply(lambda df: 1 if df['UMN vs LMN_both'] == 1 or df['LMN'] == 1 else 0, axis=1)
ALS_df.UMN = ALS_df.apply(lambda df: 1 if df['UMN vs LMN_both'] == 1 or df['UMN'] == 1 else 0, axis=1)

# Drop the "both" column as it's redundant
ALS_df.drop(columns='UMN vs LMN_both', inplace=True)

ALS_df.head()

len(ALS_df[(ALS_df.UMN == 1) & (ALS_df.LMN == 1)])

# **Comment:** The previous length matches the number found on the value counts of the original dataframe, corresponding to the value "both".

# Remove the redundant `C9orf72_no` column:

ALS_df.columns

ALS_df.drop(columns='C9orf72_no', inplace=True)
ALS_df.head()

ALS_df.rename(columns={'C9orf72_yes': 'C9orf72'}, inplace=True)
ALS_df.head()

# ## Standardize all column names to be lower case and without spaces

ALS_df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in ALS_df.columns]

ALS_df.head()

# ## NIV label
#
# In order to predict the use of NIV in the next 3 months, we need to create a shifted version of the "niv" column.

ALS_df[['subject_id', 'ts', 'niv']].head(20)

# + {"pixiedust": {"displayParams": {}}}
ALS_df['niv_label'] = utils.set_niv_label(ALS_df, time_window_days=90)
# -

ALS_df[['subject_id', 'ts', 'niv', 'niv_label']].head(200)

# Save a version of the dataframe without normalization
ALS_df.to_csv(f'{data_path}cleaned/FCUL_ALS_cleaned_denorm.csv')

ALS_df.describe().transpose()

# ## Normalizing continuous values
#
# Continuous data is normalized into z-scores, where 0 represents the mean and an absolute value of 1 corresponds to the standard deviation.

ALS_df = du.data_processing.normalize_data(ALS_df, id_columns=['subject_id', 'ts'])
ALS_df.head()


ALS_df.describe().transpose()

# ## Imputation and removal of incomplete data
#
# Starting from a last information carried forward technique, the data is initially forward filled. Next, a backward fill is done, as current data of the patient should still be a good indicator of the recent past. Finally, the remaining missing values are filled with zeroes, as it represents the average value of each given feature.

ALS_df[['subject_id', 'ts', 'r', 'p1', 'p2', 'bmi', 'fvc', 'vc', 'mip', 'niv_label']].head(20)

ALS_df = du.data_processing.missing_values_imputation(ALS_df, method='zigzag', id_column='subject_id')
ALS_df.head()

ALS_df[['subject_id', 'ts', 'r', 'p1', 'p2', 'bmi', 'fvc', 'vc', 'mip', 'niv_label']].head(20)

# ## Saving the data

ALS_df.to_csv(f'{data_path}cleaned/FCUL_ALS_cleaned.csv')

ALS_df.head()

ALS_df.columns
