# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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

# # FCUL ALS Pre Training Exploration
# ---
#
# Exploring the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.
#
# Just playing around with the cleaned dataframe before inputing it to the machine learning pipeline.

# + [markdown] colab_type="text" id="KOdmFzXqF7nq"
# ## Importing the necessary packages

# + colab={} colab_type="code" id="G5RrWE9R_Nkl"
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

import plotly.io as pio
pio.templates

# Use Plotly in dark mode:

pio.templates.default = 'plotly_dark'

# Change to parent directory (presumably "Documents")
os.chdir("../../..")
# Path to the CSV dataset files
data_path = 'Datasets/Thesis/FCUL_ALS/cleaned/'

du.set_pandas_library(lib='pandas')

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

# Set the random seed for reproducibility:

du.set_random_seed(42)

# ## Exploring the cleaned dataset

# ### Loading the data

# Original data:

orig_ALS_df = pd.read_csv(f'{data_path}FCUL_ALS_cleaned_denorm.csv')
orig_ALS_df.drop(columns=['Unnamed: 0'], inplace=True)
orig_ALS_df.head()

# Preprocessed data:

ALS_proc_df = pd.read_csv(f'{data_path}FCUL_ALS_cleaned.csv')
ALS_proc_df.drop(columns=['Unnamed: 0'], inplace=True)
ALS_proc_df.head()

# ### Basic stuff

ALS_proc_df.dtypes

ALS_proc_df.nunique()

du.search_explore.dataframe_missing_values(ALS_proc_df)

ALS_proc_df.describe().transpose()

orig_ALS_df.describe().transpose()

# Number of model features:

len(ALS_proc_df.columns) - 4

# Number of data points (# features x # rows):

len(ALS_proc_df.columns) * len(ALS_proc_df)

# ### Label analysis

# Counting the samples with positive label:

label_count = ALS_proc_df.niv_label.value_counts()
label_count

all(label_count == orig_ALS_df.niv_label.value_counts())

print(f'{(label_count[True] / (label_count[True] + label_count[False])) * 100}%')

label_per_subject_count = ALS_proc_df.groupby('subject_id').niv_label.max().value_counts()
label_per_subject_count

print(f'{(label_per_subject_count[True] / (label_per_subject_count[True] + label_per_subject_count[False])) * 100}%')

# How many subjects always have the same label in their time series:

const_label_subj = list()
for subject in ALS_proc_df.subject_id.unique():
    subject_data = ALS_proc_df[ALS_proc_df.subject_id == subject]
    if subject_data.niv_label.min() == subject_data.niv_label.max():
        const_label_subj.append(subject)
const_label_subj

len(const_label_subj)

percent_const_label_subj = (len(const_label_subj) / ALS_proc_df.subject_id.nunique()) * 100
print(f'{percent_const_label_subj}%')

# **Comment:** It's a real bummer that over 60% of the subjects have a static / constant label value. But it is what it is. I think that I would have more to lose if I would remove all of these subjects' data, which still give the model an idea of what a patient needing NIV looks like, and vice versa.

# ### Time / sampling variation

ALS_proc_df['delta_ts'] = ALS_proc_df.groupby('subject_id').ts.diff()
ALS_proc_df.head()

ALS_proc_df.delta_ts.describe()

# ### Sequence length analysis

seq_len = ALS_proc_df.groupby('subject_id').ts.count()
seq_len.head()

seq_len.describe()

# ## Random exploratory stuff

labels = torch.Tensor([0, 0, 0, 1, 1, 1])
pred = torch.Tensor([1, 0, 0, 0, 1, 1])
correct_pred = pred == labels
correct_pred

torch.masked_select(pred, labels.byte())

true_pos = int(sum(torch.masked_select(pred, labels.byte())))
true_pos

false_neg = int(sum(torch.masked_select(pred == 0, labels.byte())))
false_neg

true_neg = int(sum(torch.masked_select(pred == 0, (labels == 0).byte())))
true_neg

false_pos = int(sum(torch.masked_select(pred, (labels == 0).byte())))
false_pos

any(metric in ['a', 'b', 'c'] for metric in ['precision', 'recall', 'F1'])

x = 1

'x' in locals()

# ### Plots

ALS_proc_gender_count = ALS_proc_df.groupby('subject_id').first().gender.value_counts().to_frame()
data = [go.Pie(labels=ALS_proc_gender_count.index, values=ALS_proc_gender_count.gender)]
layout = go.Layout(title='Patients Gender Demographics')
fig = go.Figure(data, layout)
fig.show()

data = [go.Histogram(x = orig_ALS_df.groupby('subject_id').first().age_at_onset)]
layout = go.Layout(title='Patient age distribution')
fig = go.Figure(data, layout)
fig.show()

ALS_proc_niv_count = ALS_proc_df.niv.value_counts().to_frame()
data = [go.Pie(labels=ALS_proc_niv_count.index, values=ALS_proc_niv_count.niv)]
layout = go.Layout(title='Visits where the patient is using NIV')
fig = go.Figure(data, layout)
fig.show()

data = [go.Histogram(x = ALS_proc_df.niv)]
layout = go.Layout(title='Number of visits where the patient is using NIV')
fig = go.Figure(data, layout)
fig.show()

ALS_proc_patient_niv_count = ALS_proc_df.groupby('subject_id').niv.max().value_counts().to_frame()
data = [go.Pie(labels=ALS_proc_patient_niv_count.index, values=ALS_proc_patient_niv_count.niv)]
layout = go.Layout(title='Patients which eventually use NIV')
fig = go.Figure(data, layout)
fig.show()

data = [go.Scatter(
                    x = ALS_proc_df.fvc,
                    y = ALS_proc_df.niv,
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
ALS_proc_df[ALS_proc_df.niv == 1].fvc.mean()

# **Comments:** The average FVC when NIV is 1 is lower than average, but the scatter plot doesn't show a very clear dependence between the variables.

data = [go.Scatter(
                    x = ALS_proc_df['disease_duration'],
                    y = ALS_proc_df.niv,
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
ALS_proc_df[ALS_proc_df.niv == 1]['disease_duration'].mean()

data = [go.Scatter(
                    x = ALS_proc_df['age_at_onset'],
                    y = ALS_proc_df.niv,
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
ALS_proc_df[ALS_proc_df.niv == 1]['age_at_onset'].mean()

ALS_proc_NIV_3R = ALS_proc_df.groupby(['3r', 'niv']).subject_id.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_proc_NIV_3R[ALS_proc_NIV_3R.niv == 0]['3r'],
                    y=ALS_proc_NIV_3R[ALS_proc_NIV_3R.niv == 0]['subject_id'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_proc_NIV_3R[ALS_proc_NIV_3R.niv == 1]['3r'],
                    y=ALS_proc_NIV_3R[ALS_proc_NIV_3R.niv == 1]['subject_id'],
                    name='Using NIV'
        )]
layout = go.Layout(title='Relation between NIV use and normalized 3R', barmode='group')
fig = go.Figure(data=data, layout=layout)
fig.show()

ALS_NIV_3R = orig_ALS_df.groupby(['3r', 'niv']).subject_id.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_NIV_3R[ALS_NIV_3R.niv == 0]['3r'],
                    y=ALS_NIV_3R[ALS_NIV_3R.niv == 0]['subject_id'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_NIV_3R[ALS_NIV_3R.niv == 1]['3r'],
                    y=ALS_NIV_3R[ALS_NIV_3R.niv == 1]['subject_id'],
                    name='Using NIV'
        )]
layout = go.Layout(title='Relation between NIV use and 3R', barmode='group')
fig = go.Figure(data=data, layout=layout)
fig.show()

# Average 3R value when NIV is used:
ALS_proc_df[ALS_proc_df.niv == 1]['3r'].mean()

# **Comments:** Clearly, there's a big dependence of the use of NIV with the respiratory symptoms indicated by 3R, as expected.

data = [go.Histogram(x = ALS_proc_df[ALS_proc_df.niv == 0].p10, name='Not used'),
        go.Histogram(x = ALS_proc_df[ALS_proc_df.niv == 1].p10, name='Using NIV')]
layout = go.Layout(title='Relation between NIV use and normalized P10.')
fig = go.Figure(data, layout)
fig.show()

data = [go.Histogram(x = orig_ALS_df[orig_ALS_df.niv == 0].p10, name='Not used'),
        go.Histogram(x = orig_ALS_df[orig_ALS_df.niv == 1].p10, name='Using NIV')]
layout = go.Layout(title='Relation between NIV use and P10.')
fig = go.Figure(data, layout)
fig.show()

# Average P10 value when NIV is used:
ALS_proc_df[ALS_proc_df.niv == 1]['p10'].mean()

# **Comments:** Clearly, there's a big dependence of the use of NIV with the respiratory symptoms indicated by P10, as expected.

ALS_proc_NIV_R = ALS_proc_df.groupby(['r', 'niv']).subject_id.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_proc_NIV_R[ALS_proc_NIV_R.niv == 0]['r'],
                    y=ALS_proc_NIV_R[ALS_proc_NIV_R.niv == 0]['subject_id'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_proc_NIV_R[ALS_proc_NIV_R.niv == 1]['r'],
                    y=ALS_proc_NIV_R[ALS_proc_NIV_R.niv == 1]['subject_id'],
                    name='Using NIV'
        )]
layout = go.Layout(title='Relation between NIV use and normalized R', barmode='group')
fig = go.Figure(data=data, layout=layout)
fig.show()

ALS_NIV_R = orig_ALS_df.groupby(['r', 'niv']).subject_id.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_NIV_R[ALS_NIV_R.niv == 0]['r'],
                    y=ALS_NIV_R[ALS_NIV_R.niv == 0]['subject_id'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_NIV_R[ALS_NIV_R.niv == 1]['r'],
                    y=ALS_NIV_R[ALS_NIV_R.niv == 1]['subject_id'],
                    name='Using NIV'
        )]
layout = go.Layout(title='Relation between NIV use and R', barmode='group')
fig = go.Figure(data=data, layout=layout)
fig.show()

# Average R value when NIV is used:
ALS_proc_df[ALS_proc_df.niv == 1]['r'].mean()

# **Comments:** There seems to be a relationship between the use of NIV and the respiratory symptoms indicated by R, as expected.

data = [go.Histogram(x = ALS_proc_df[ALS_proc_df.niv == 0].bmi, name='Not used'),
        go.Histogram(x = ALS_proc_df[ALS_proc_df.niv == 1].bmi, name='Using NIV')]
layout = go.Layout(title='Relation between NIV use and normalized BMI.')
fig = go.Figure(data, layout)
fig.show()

data = [go.Histogram(x = orig_ALS_df[orig_ALS_df.niv == 0].bmi, name='Not used'),
        go.Histogram(x = orig_ALS_df[orig_ALS_df.niv == 1].bmi, name='Using NIV')]
layout = go.Layout(title='Relation between NIV use and BMI.')
fig = go.Figure(data, layout)
fig.show()

# Average BMI value when NIV is used:
ALS_proc_df[ALS_proc_df.niv == 1]['bmi'].mean()

# **Comments:** There is no clear, universal relationship between the use of NIV and BMI.

ALS_proc_NIV_p5 = ALS_proc_df.groupby(['p5', 'niv']).subject_id.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_proc_NIV_p5[ALS_proc_NIV_p5.niv == 0]['p5'],
                    y=ALS_proc_NIV_p5[ALS_proc_NIV_p5.niv == 0]['subject_id'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_proc_NIV_p5[ALS_proc_NIV_p5.niv == 1]['p5'],
                    y=ALS_proc_NIV_p5[ALS_proc_NIV_p5.niv == 1]['subject_id'],
                    name='Using NIV'
        )]
layout = go.Layout(title='Relation between NIV use and normalized P5', barmode='group')
fig = go.Figure(data=data, layout=layout)
fig.show()

ALS_NIV_p5 = orig_ALS_df.groupby(['p5', 'niv']).subject_id.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_NIV_p5[ALS_NIV_p5.niv == 0]['p5'],
                    y=ALS_NIV_p5[ALS_NIV_p5.niv == 0]['subject_id'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_NIV_p5[ALS_NIV_p5.niv == 1]['p5'],
                    y=ALS_NIV_p5[ALS_NIV_p5.niv == 1]['subject_id'],
                    name='Using NIV'
        )]
layout = go.Layout(title='Relation between NIV use and P5', barmode='group')
fig = go.Figure(data=data, layout=layout)
fig.show()

# Average P5 value when NIV is used:
ALS_proc_df[ALS_proc_df.niv == 1]['p5'].mean()

# **Comments:** There seems to be a relationship between the use of NIV and the strength symptoms indicated by P5.

ALS_proc_NIV_P4 = ALS_proc_df.groupby(['p4', 'niv']).subject_id.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_proc_NIV_P4[ALS_proc_NIV_P4.niv == 0]['p4'],
                    y=ALS_proc_NIV_P4[ALS_proc_NIV_P4.niv == 0]['subject_id'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_proc_NIV_P4[ALS_proc_NIV_P4.niv == 1]['p4'],
                    y=ALS_proc_NIV_P4[ALS_proc_NIV_P4.niv == 1]['subject_id'],
                    name='Using NIV'
        )]
layout = go.Layout(title='Relation between NIV use and normalized P4', barmode='group')
fig = go.Figure(data=data, layout=layout)
fig.show()

ALS_NIV_P4 = orig_ALS_df.groupby(['p4', 'niv']).subject_id.count().to_frame().reset_index()
data = [go.Bar(
                    x=ALS_NIV_P4[ALS_NIV_P4.niv == 0]['p4'],
                    y=ALS_NIV_P4[ALS_NIV_P4.niv == 0]['subject_id'],
                    name='Not used'
              ),
        go.Bar(
                    x=ALS_NIV_P4[ALS_NIV_P4.niv == 1]['p4'],
                    y=ALS_NIV_P4[ALS_NIV_P4.niv == 1]['subject_id'],
                    name='Using NIV'
        )]
layout = go.Layout(title='Relation between NIV use and P4', barmode='group')
fig = go.Figure(data=data, layout=layout)
fig.show()

# Average P4 value when NIV is used:
ALS_proc_df[ALS_proc_df.niv == 1]['p4'].mean()

# **Comments:** There seems to be a relationship between the use of NIV and the handwriting symptoms indicated by P4.

data = [go.Histogram(x = ALS_proc_df[ALS_proc_df.niv == 0]['p0.1'], name='Not used'),
        go.Histogram(x = ALS_proc_df[ALS_proc_df.niv == 1]['p0.1'], name='Using NIV')]
layout = go.Layout(title='Relation between NIV use and normalized P0.1.')
fig = go.Figure(data, layout)
fig.show()

data = [go.Histogram(x = orig_ALS_df[orig_ALS_df.niv == 0]['p0.1'], name='Not used'),
        go.Histogram(x = orig_ALS_df[orig_ALS_df.niv == 1]['p0.1'], name='Using NIV')]
layout = go.Layout(title='Relation between NIV use and P0.1.')
fig = go.Figure(data, layout)
fig.show()

# Average P0.1 value when NIV is used:
ALS_proc_df[ALS_proc_df.niv == 1]['p0.1'].mean()

# **Comments:** There is no clear, universal relationship between the use of NIV and P0.1.


