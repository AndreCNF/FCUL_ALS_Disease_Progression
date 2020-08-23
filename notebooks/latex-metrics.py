# -*- coding: utf-8 -*-
# # FCUL ALS LaTeX Metrics
# ---
#
# Joining the metrics of the models trained on the ALS dataset from Faculdade de Ciências da Universidade de Lisboa (FCUL) with the data from over 1000 patients collected in Portugal.

# ## Importing the necessary packages

import os                                  # os handles directory/workspace changes
import yaml                                # Save and load YAML files
import plotly.graph_objs as go             # Plotly for interactive and pretty plots

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

# ## Creating the tables

# ### Performance metrics

# Convert to a dataframe:

metrics_df = pd.DataFrame(metrics)
metrics_df

# Transpose to have a row per model:

metrics_df = metrics_df.transpose()
metrics_df

# Sort by a descending order of performance:

metrics_df = metrics_df.sort_values('Avg. Test AUC', ascending=False)
metrics_df

# Convert to a LaTeX table:

metrics_df.to_latex()

# ### Component impact
#
# Measuring the average gain in performance that we get from the components of bidirectionality, embedding layer and time awareness.

model_names = list(metrics_df.index)
model_names

component_gains = dict()
components_str = dict(bidirectionality='Bidirectional ', 
                      embedding=', embedded', 
                      time_awareness=', time aware')
for component in components_str.keys():
    # Find and match the names of the models with and without the component
    models_without_comp = [model_name.replace(components_str[component], '') 
                           for model_name in model_names 
                           if components_str[component] in model_name]
    models_with_comp = [model_name 
                        for model_name in model_names 
                        if components_str[component] in model_name]
    model_comp_names_match = dict(zip(models_without_comp, models_with_comp))
    curr_component_gains = list()
    for model_name in models_without_comp:
        # Calculate the difference in model performance with and without the component
        component_gain = (metrics_df.loc[model_comp_names_match[model_name], 'Avg. Test AUC'] 
                          - metrics_df.loc[model_name, 'Avg. Test AUC'])
        curr_component_gains.append(component_gain)
    # Average the component's effect
    component_gains[component] = sum(curr_component_gains) / len(curr_component_gains)
component_gains

# Find and match the names of the models with LSTM and with RNN
models_with_lstm = [model_name.replace('RNN', 'LSTM')
                    for model_name in model_names 
                    if 'RNN' in model_name]
models_with_rnn = [model_name 
                   for model_name in model_names 
                   if 'RNN' in model_name]
model_comp_names_match = dict(zip(models_with_rnn, models_with_lstm))
curr_component_gains = list()
for model_name in models_with_rnn:
    # Calculate the difference in model performance with LSTM and with RNN
    component_gain = (metrics_df.loc[model_comp_names_match[model_name], 'Avg. Test AUC'] 
                      - metrics_df.loc[model_name, 'Avg. Test AUC'])
    curr_component_gains.append(component_gain)
# Average LSTM's effect
component_gains['LSTM'] = sum(curr_component_gains) / len(curr_component_gains)
component_gains

# Convert to a dataframe:

gain_df = pd.Series(component_gains, name='Avg. Impact on Test AUC')
gain_df

gain_df.index = ['Bidirectionality', 'Embedding', 'Time Awareness', 'LSTM']
gain_df

gain_df.index.rename('Component')
gain_df

# Sort by a descending order of performance gain:

gain_df = gain_df.sort_values(ascending=False)
gain_df

# Convert to a LaTeX table:

gain_df.to_latex()

# Make a bar plot, similar to SHAP's summary plot:

gain_plot_df = gain_df.copy()
gain_plot_df = gain_plot_df.sort_values(ascending=True)
# Define the colors based on the value
marker_color = ['rgba(255,13,87,1)' 
                if gain_plot_df[comp] > 0
                else 'rgba(30,136,229,1)'
                for comp in gain_plot_df.index ]
# Create the figure
figure=dict(
    data=[dict(
        type='bar',
        x=gain_plot_df,
        y=gain_plot_df.index,
        orientation='h',
        marker=dict(color=marker_color)
    )],
    layout=dict(
        paper_bgcolor='white',
        plot_bgcolor='white',
        title='Average impact on model\'s test AUC',
        yaxis_title=gain_plot_df.index.name,
        font=dict(
            family='Roboto',
            size=14,
            color='black'
        )
    )
)
go.Figure(figure)


