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

# # FCUL ALS Data Exploration
# ---
#
# Exploring the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.
#
# Amyotrophic lateral sclerosis, or ALS (also known in the US as Lou Gehrig’s Disease and as Motor Neuron Disease in the UK) is a disease that involves the degeneration and death of the nerve cells in the brain and spinal cord that control voluntary muscle movement. Death typically occurs within 3 - 5 years of diagnosis. Only about 25% of patients survive for more than 5 years after diagnosis.

# + [markdown] {"colab_type": "text", "id": "KOdmFzXqF7nq"}
# ## Importing the necessary packages

# + {"colab": {}, "colab_type": "code", "id": "G5RrWE9R_Nkl"}
import pandas as pd                        # Pandas to handle the data in dataframes
import re                                  # re to do regex searches in string data
import plotly                              # Plotly for interactive and pretty plots
import plotly.graph_objs as go
from datetime import datetime              # datetime to use proper date and time formats
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
from tqdm import tqdm_notebook             # tqdm allows to track code execution progress
import numbers                             # numbers allows to check if data is numeric
import torch                               # PyTorch to create and apply deep learning models
from torch.utils.data.sampler import SubsetRandomSampler
import data_utils as du                    # Data science and machine learning relevant methods
# -

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the CSV dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/'
# ## Exploring the preprocessed dataset

# ### Basic stats

ALS_proc_df = pd.read_csv(f'{data_path}dataWithoutDunnoNIV.csv')
ALS_proc_df.head()

ALS_proc_df.dtypes

ALS_proc_df.nunique()

du.search_explore.dataframe_missing_values(ALS_proc_df)

# **Comment:** Many relevant features (timestamps, NIV, age, ALSFRS, etc) have zero or low missing values percentage (bellow 10%), much better than in the PRO-ACT dataset. However, there are other interesting ones with more than half missing values (FVC, VC, etc).

ALS_proc_df.describe().transpose()

ALS_proc_df['El Escorial reviewed criteria'].value_counts()

ALS_proc_df['Onset form'].value_counts()

ALS_proc_df['UMN vs LMN'].value_counts()

ALS_proc_df['C9orf72'].value_counts()

ALS_proc_df['SNIP'].value_counts()

ALS_proc_df['1R'].value_counts()

# ### Plots

ALS_proc_gender_count = ALS_proc_df.groupby('REF').first().Gender.value_counts().to_frame()
data = [go.Pie(labels=ALS_proc_gender_count.index, values=ALS_proc_gender_count.Gender)]
layout = go.Layout(title='Patients Gender Demographics')
fig = go.Figure(data, layout)
fig.show()

ALS_proc_niv_count = ALS_proc_df.NIV.value_counts().to_frame()
data = [go.Pie(labels=ALS_proc_niv_count.index, values=ALS_proc_niv_count.NIV)]
layout = go.Layout(title='Visits where the patient is using NIV')
fig = go.Figure(data, layout)
fig.show()

data = [go.Histogram(x = ALS_proc_df.NIV)]
layout = go.Layout(title='Number of visits where the patient is using NIV.')
fig = go.Figure(data, layout)
fig.show()

data = [go.Scatter(
                    x = ALS_proc_df.FVC,
                    y = ALS_proc_df.NIV,
                    mode = 'markers'
                  )]
layout = go.Layout(
                    title='Relation between NIV use and FVC values',
                    xaxis=dict(title='FVC'),
                    yaxis=dict(title='NIV')
                  )
fig = go.Figure(data, layout)
fig.show()

# Average FVC value when NIV is used:
ALS_proc_df[ALS_proc_df.NIV == 1].FVC.mean()

# **Comments:** The average FVC when NIV is 1 is lower than average, but the scatter plot doesn't show a very clear dependence between the variables.

data = [go.Scatter(
                    x = ALS_proc_df['Disease duration'],
                    y = ALS_proc_df.NIV,
                    mode = 'markers'
                  )]
layout = go.Layout(
                    title='Relation between NIV use and disease duration',
                    xaxis=dict(title='Disease duration'),
                    yaxis=dict(title='NIV')
                  )
fig = go.Figure(data, layout)
fig.show()

# Average disease duration when NIV is used:
ALS_proc_df[ALS_proc_df.NIV == 1]['Disease duration'].mean()

data = [go.Scatter(
                    x = ALS_proc_df['Age at onset'],
                    y = ALS_proc_df.NIV,
                    mode = 'markers'
                  )]
layout = go.Layout(
                    title='Relation between NIV use and age',
                    xaxis=dict(title='Age at onset'),
                    yaxis=dict(title='NIV')
                  )
fig = go.Figure(data, layout)
fig.show()

# Average age at onset when NIV is used:
ALS_proc_df[ALS_proc_df.NIV == 1]['Age at onset'].mean()

ALS_proc_NIV_3R = ALS_proc_df.groupby(['3R', 'NIV']).REF.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_proc_NIV_3R[ALS_proc_NIV_3R.NIV == 0]['3R'],
                    y=ALS_proc_NIV_3R[ALS_proc_NIV_3R.NIV == 0]['REF'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_proc_NIV_3R[ALS_proc_NIV_3R.NIV == 1]['3R'],
                    y=ALS_proc_NIV_3R[ALS_proc_NIV_3R.NIV == 1]['REF'],
                    name='Using NIV'
        )]
layout = go.Layout(barmode='group')
fig = go.Figure(data=data, layout=layout)
fig.show()

# Average 3R value when NIV is used:
ALS_proc_df[ALS_proc_df.NIV == 1]['3R'].mean()

# **Comments:** Clearly, there's a big dependence of the use of NIV with the respiratory symptoms indicated by 3R, as expected.

# ## Exploring the raw dataset

ALS_raw_df = pd.read_excel(f'{data_path}TabelaGeralnew_21012019_sem.xlsx')
ALS_raw_df.head()
