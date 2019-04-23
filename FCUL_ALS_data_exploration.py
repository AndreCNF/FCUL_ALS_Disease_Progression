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

# # FCUL ALS Data Exploration
#
# Exploring the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.
#
# Amyotrophic lateral sclerosis, or ALS (also known in the US as Lou Gehrig’s Disease and as Motor Neuron Disease in the UK) is a disease that involves the degeneration and death of the nerve cells in the brain and spinal cord that control voluntary muscle movement. Death typically occurs within 3 - 5 years of diagnosis. Only about 25% of patients survive for more than 5 years after diagnosis.

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

# ## Exploring the preprocessed dataset

ALS_proc_df = pd.read_csv(f'{data_path}dataWithoutDunnoNIV.csv')
ALS_proc_df.head()

ALS_proc_df.dtypes

ALS_proc_df.nunique()

utils.dataframe_missing_values(ALS_proc_df)

# **Comment:** Many relevant features (timestamps, NIV, age, ALSFRS, etc) have zero or low missing values percentage (bellow 10%), much better than in the PRO-ACT dataset. However, there other interesting ones with more than half missing values (FVC, VC, etc).

ALS_proc_df.describe().transpose()

ALS_proc_df['El Escorial reviewed criteria'].value_counts()

ALS_proc_df['Onset form'].value_counts()

ALS_proc_df['UMN vs LMN'].value_counts()

ALS_proc_df['C9orf72'].value_counts()

ALS_proc_df['SNIP'].value_counts()

ALS_proc_df['1R'].value_counts()

# ## Exploring the raw dataset

ALS_raw_df = pd.read_excel(f'{data_path}TabelaGeralnew_21012019_sem.xlsx')
ALS_raw_df.head()
