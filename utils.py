from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters and do version control
import torch                                            # PyTorch to create and apply deep learning models
from torch import nn, optim                             # nn for neural network layers and optim for training optimizers
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd                                     # Pandas to handle the data in dataframes
from datetime import datetime                           # datetime to use proper date and time formats
import os                                               # os handles directory/workspace changes
import numpy as np                                      # NumPy to handle numeric and NaN operations
from tqdm import tqdm                                   # tqdm allows to track code execution progress
from tqdm import tqdm_notebook                          # tqdm allows to track code execution progress
import numbers                                          # numbers allows to check if data is numeric
from NeuralNetwork import NeuralNetwork                 # Import the neural network model class

# Exceptions

class ColumnNotFoundError(Exception):
   """Raised when the column name is not found in the dataframe."""
   pass


# Auxiliary functions

def dataframe_missing_values(df, column=None):
    '''Returns a dataframe with the percentages of missing values of every column of the original dataframe.

    Parameters
    ----------
    df : pandas.Dataframe
        Original dataframe which the user wants to analyze for missing values.
    column : string, default None
        Optional argument which, if provided, makes the function only return
        the percentage of missing values in the specified column.
    
    Returns
    -------
    missing_value_df : pandas.Dataframe
        Dataframe containing the percentages of missing values for each column.
    col_percent_missing : float
        If the "column" argument is provided, the function only returns a float
        corresponfing to the percentage of missing values in the specified column.
    '''
    if column is None:
        columns = df.columns
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame({'column_name': columns,
                                       'percent_missing': percent_missing})
        missing_value_df.sort_values('percent_missing', inplace=True)
        return missing_value_df
    else:
        col_percent_missing = df[column].isnull().sum() * 100 / len(df)
        return col_percent_missing


def get_clean_label(orig_label, clean_labels, column_name=None):
    '''Gets the clean version of a given label.

    Parameters
    ----------
    orig_label : string
        Original label name that needs to be converted to the new format.
    clean_labels : dict
        Dictionary that converts each original label into a new, cleaner designation.
    column_name : string, default None
        Optional parameter to indicate a column name, which is used to specify better the
        missing values.

    Returns
    -------
    key : string
        Returns the dictionary key from clean_labels that corresponds to the translation 
        given to the input label orig_label.
    '''
    for key in clean_labels:
        if orig_label in clean_labels[key]:
            return key

    # Remaining labels (or lack of one) are considered as missing data
    if column_name is not None:
        return f'{column_name}_missing_value'
    else:
        return 'missing_value'


def is_one_hot_encoded_column(df, column):
    '''Checks if a given column is one hot encoded.

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe that will be used, which contains the specified column.
    column : string
        Name of the column that will be checked for one hot encoding.

    Returns
    -------
    bool
        Returns true if the column is in one hot encoding format.
        Otherwise, returns false.
    '''
    # Check if it only has 2 possible values
    if df[column].nunique() == 2:
        # Check if the possible values are all numeric
        if all([isinstance(x, numbers.Number) for x in df[column].unique()]):
            # Check if the only possible values are 0 and 1 (and ignore NaN's)
            if (np.sort(list(set(np.nan_to_num(df[column].unique())))) == [0, 1]).all():
                return True
    return False


def list_one_hot_encoded_columns(df):
    '''Lists the columns in a dataframe which are in a one hot encoding format.

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe that will be used checked for one hot encoded columns.

    Returns
    -------
    list of strings
        Returns a list of the column names which correspond to one hot encoded columns.
    '''
    return [col for col in df.columns if is_one_hot_encoded_column(df, col)]


def one_hot_encoding_dataframe(df, columns, std_name=True, has_nan=False, join_rows=True, join_by=['subject_id', 'ts']):
    '''Transforms a specified column from a dataframe into a one hot encoding representation.

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe that will be used, which contains the specified column.
    columns : list of strings
        Name of the column(s) that will be conveted to one hot encoding. Even if it's just one
        column, please provide inside a list.
    std_name : bool, default True
        If set to true, changes the name of the categorical values into lower case, with words
        separated by an underscore instead of space.
    has_nan : bool, default False
        If set to true, will first fill the missing values (NaN) with the string 
        f'{column}_missing_value'.
    join_rows : bool, default True
        If set to true, will group the rows created by the one hot encoding by summing the 
        boolean values in the rows that have the same identifiers.
    join_by : string or list, default ['subject_id', 'ts'])
        Name of the column (or columns) which serves as a unique identifier of the dataframe's
        rows, which will be used in the groupby operation if the parameter join_rows is set to
        true. Can be a string (single column) or a list of strings (multiple columns).
        
    Raises
    ------
    ColumnNotFoundError
        Column name not found in the dataframe.

    Returns
    -------
    ohe_df : pandas.Dataframe
        Returns a new dataframe with the specified column in a one hot encoding representation.
    '''
    for col in columns:
        if has_nan:
            # Fill NaN with "missing_value" name
            df[col].fillna(value='missing_value', inplace=True)

        # Check if the column exists
        if col not in df:
            raise ColumnNotFoundError('Column name not found in the dataframe.')

        if std_name:
            # Change categorical values to only have lower case letters and underscores
            df[col] = df[col].apply(lambda x: str(x).lower().replace(' ', '_').replace(',', '_and'))

        # Cast the variable into the built in pandas Categorical data type
        df[col] = pd.Categorical(df[col])

    # Apply the one hot encoding to the specified columns
    ohe_df = pd.get_dummies(df, columns=columns)
    
    if join_rows:
        # Columns which are one hot encoded
        ohe_columns = list_one_hot_encoded_columns(ohe_df)
        
        # Group the rows that have the same identifiers
        ohe_df = ohe_df.groupby(join_by).sum(min_count=1).reset_index()
        
        # Clip the one hot encoded columns to a maximum value of 1
        # (there might be duplicates which cause values bigger than 1)
        ohe_df.loc[:, ohe_columns] = ohe_df[ohe_columns].clip(upper=1)

    return ohe_df


def is_definitely_string(x):
    '''Reports if a value is actually a real string or if it has some number in it.

    Parameters
    ----------
    x
        Any value which will be judged to be either a real string or numeric.
        
    Returns
    -------
    boolean
        Returns a boolean, being it True if it really is a string or False if it's
        either numeric data or a string with a number inside.
    '''
    if isinstance(x, int) or isinstance(x, float):
        return False
    
    try:
        float(x)
        return False
    
    except:
        return isinstance(x, str)


def remove_rows_unmatched_key(df, key, columns):
    '''Remove rows corresponding to the keys that weren't in the dataframe merged at the right.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe resulting from a asof merge which will be searched for missing values.
    key : string
        Name of the column which was used as the "by" key in the asof merge. Typically 
        represents a temporal feature from a time series, such as days or timestamps.
    columns : list of strings
        Name of the column(s), originating from the dataframe which was merged at the 
        right, which should not have any missing values. If it has, it means that
        the corresponding key wasn't present in the original dataframe. Even if there's
        just one column to analyze, it should be received in list format.
        
    Returns
    -------
    df : pandas.DataFrame
        Returns the input dataframe but without the rows which didn't have any values
        in the right dataframe's features.
    '''
    for k in tqdm_notebook(df[key].unique()):
        # Variable that count the number of columns which don't have any value 
        # (i.e. all rows are missing values) for a given identifier 'k'
        num_empty_columns = 0
        
        for col in columns:
            if df[df[key] == k][col].isnull().sum() == len(df[df[key] == k]):
                # Found one more column which is full of missing values for identifier 'k'
                num_empty_columns += 1
                
        if num_empty_columns == len(columns):
            # Eliminate all rows corresponding to the analysed key if all the columns 
            # are empty for the identifier 'k'
            df = df[~(df[key] == k)]   
                
    return df


def dataframe_to_padded_tensor(df, seq_len_dict, n_ids, n_inputs, id_column='subject_id', data_type='PyTorch', padding_value=999999):
    '''Converts a Pandas dataframe into a padded NumPy array or PyTorch Tensor.

    Parameters
    ----------
    df : pandas.Dataframe
        Data in a Pandas dataframe format which will be padded and converted 
        to the requested data type.
    seq_len_dict : dictionary
        Dictionary containing the original sequence lengths of the dataframe.
    n_ids : integer
        Total number of subject identifiers in a dataframe.
        Example: Total number of patients in a health dataset.
    n_inputs : integer
        Total number of input features present in the dataframe.
    id_column : string, default 'subject_id'
        Name of the column which corresponds to the subject identifier in the
        dataframe.
    data_type : string, default 'PyTorch'
        Indication of what kind of output data type is desired. In case it's 
        set as 'NumPy', the function outputs a NumPy array. If it's 'PyTorch',
        the function outputs a PyTorch tensor.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    arr : torch.Tensor or numpy.array
        PyTorch tensor or NumPy array version of the dataframe, after being
        padded with the specified padding value to have a fixed sequence 
        length.
    '''
    # Max sequence length (e.g. patient with the most temporal events)
    max_seq_len = seq_len_dict[max(seq_len_dict, key=seq_len_dict.get)]

    # Making a padded numpy array version of the dataframe (all index has the same sequence length as the one with the max)
    arr = np.ones((n_ids, max_seq_len, n_inputs)) * padding_value

    # Iterator that gives each unique identifier (e.g. each patient in the dataset)
    id_iter = iter(df[id_column].unique())

    # Count the iterations of ids
    count = 0

    # Assign each value from the dataframe to the numpy array
    for idt in id_iter:
        arr[count, :seq_len_dict[idt], :] = df[df[id_column] == idt].to_numpy()
        arr[count, seq_len_dict[idt]:, :] = padding_value
        count += 1

    # Make sure that the data type asked for is a string
    if not isinstance(data_type, str):
        raise Exception('ERROR: Please provide the desirable data type in a string format.')

    if data_type.lower() == 'numpy':
        return arr
    elif data_type.lower() == 'pytorch':
        return torch.from_numpy(arr)
    else:
        raise Exception('ERROR: Unavailable data type. Please choose either NumPy or PyTorch.')


def normalize_data(df, data=None, id_columns=['subject_id', 'ts'], normalization_method='z-score', columns_to_normalize=None):
    '''Performs data normalization to a continuous valued tensor, to scale to a range between -1 and 1.

    Parameters
    ----------
    df : pandas.Dataframe
        Original pandas dataframe which is used to correctly calculate the  
        necessary statistical values used in the normalization. These values
        can't be calculated from the tensor as it might have been padded. If
        the data tensor isn't specified, the normalization is applied directly
        on the dataframe.
    data : torch.Tensor, default None
        PyTorch tensor corresponding to the data which will be normalized
        by the specified normalization method. If the data tensor isn't 
        specified, the normalization is applied directly on the dataframe.
    id_columns : list of strings, default ['subject_id', 'ts']
        List of columns names which represent identifier columns. These are not
        supposed to be normalized.
    normalization_method : string, default 'z-score'
        Specifies the normalization method used. It can be a z-score
        normalization, where the data is subtracted of it's mean and divided
        by the standard deviation, which makes it have zero average and unit
        variance, much like a standard normal distribution; it can be a 
        min-max normalization, where the data is subtracted by its minimum
        value and then divided by the difference between the minimum and the
        maximum value, getting to a fixed range from 0 to 1.
    columns_to_normalize : list of strings, default None
        If specified, the columns provided in the list are the only ones that
        will be normalized. Otherwise, all non identifier continuous columns
        will be normalized.

    Returns
    -------
    data : pandas.Dataframe or torch.Tensor
        Normalized Pandas dataframe or PyTorch tensor.
    '''
    # Check if specific columns have been specified for normalization
    if columns_to_normalize is None: 
        # Normalize all non identifier continuous columns, ignore one hot encoded ones
        columns_to_normalize = [col for col in df.columns if col not in list_one_hot_encoded_columns(df) and col not in id_columns]
    
    if type(normalization_method) is not str:
        raise ValueError('Argument normalization_method should be a string. Available options \
                         are \'z-score\' and \'min-max\'.')
    
    if normalization_method.lower() == 'z-score':
        column_means = dict(df[columns_to_normalize].mean())
        column_stds = dict(df[columns_to_normalize].std())

        # Check if the data being normalized is directly the dataframe
        if data is None:
            # Treat the dataframe as the data being normalized
            data = df
            
            # Normalize the right columns
            for col in columns_to_normalize:
                data[col] = (data[col] - column_means[col]) / column_stds[col]
        
        # Otherwise, the tensor is normalized
        else:
            # Dictionary to convert the the tensor's column indeces into the dataframe's column names
            idx_to_name = dict(enumerate(df.columns))

            # Dictionary to convert the dataframe's column names into the tensor's column indeces
            name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

            # List of indeces of the tensor's columns which are needing normalization
            tensor_columns_to_normalize = [name_to_idx[name] for name in columns_to_normalize]

            # Normalize the right columns
            for col in tensor_columns_to_normalize:
                data[:, :, col] = (data[:, :, col] - column_means[idx_to_name[col]]) / column_stds[idx_to_name[col]]
                         
    elif normalization_method.lower() == 'min-max':
        column_mins = dict(df[columns_to_normalize].min())
        column_maxs = dict(df[columns_to_normalize].max())

        # Check if the data being normalized is directly the dataframe
        if data is None:
            # Treat the dataframe as the data being normalized
            data = df
            
            # Normalize the right columns
            for col in columns_to_normalize:
                data[col] = (data[col] - column_mins[col]) / (column_maxs[col] - column_mins[col])
        
        # Otherwise, the tensor is normalized
        else:
            # Dictionary to convert the the tensor's column indeces into the dataframe's column names
            idx_to_name = dict(enumerate(df.columns))

            # Dictionary to convert the dataframe's column names into the tensor's column indeces
            name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

            # List of indeces of the tensor's columns which are needing normalization
            tensor_columns_to_normalize = [name_to_idx[name] for name in columns_to_normalize]

            # Normalize the right columns
            for col in tensor_columns_to_normalize:
                data[:, :, col] = (data[:, :, col] - column_mins[idx_to_name[col]]) / \
                                  (column_maxs[idx_to_name[col]] - column_mins[idx_to_name[col]])
                         
    else:
        raise ValueError(f'{normalization_method} isn\'t a valid normalization method. Available options \
                         are \'z-score\' and \'min-max\'.')

    return data


def missing_values_imputation(tensor):
    '''Performs missing values imputation to a tensor corresponding to a single column.

    Parameters
    ----------
    tensor : torch.Tensor
        PyTorch tensor corresponding to a single column which will be imputed.

    Returns
    -------
    tensor : torch.Tensor
        Imputed PyTorch tensor.
    '''
    # Replace NaN's with zeros
    tensor = torch.where(tensor != tensor, torch.zeros_like(tensor), tensor)

    return tensor


def create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1, batch_size=32, get_indeces=True, 
                      random_seed=42, shuffle_dataset=True):
    '''Distributes the data into train, validation and test sets and returns the respective data loaders.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset object which will be used to train, validate and test the model.
    test_train_ratio : float, default 0.8
        Number from 0 to 1 which indicates the percentage of the data 
        which will be used as a test set. The remaining percentage
        is used in the training and validation sets.
    validation_ratio : float, default 0.1
        Number from 0 to 1 which indicates the percentage of the data
        from the training set which is used for validation purposes.
        A value of 0.0 corresponds to not using validation.
    batch_size : integer, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    get_indeces : bool, default True
        If set to True, the function returns the dataloader objects of 
        the train, validation and test sets and also the indices of the
        sets' data. Otherwise, it only returns the data loaders.
    random_seed : integer, default 42
        Seed used when shuffling the data.
    shuffle_dataset : bool, default True
        If set to True, the data of which set is shuffled.

    Returns
    -------
    train_data : torch.Tensor
        Data which will be used during training.
    val_data : torch.Tensor
        Data which will be used to evaluate the model's performance 
        on a validation set during training.
    test_data : torch.Tensor
        Data which will be used to evaluate the model's performance
        on a test set, after finishing the training process.
    '''
    # Create data indices for training and test splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_train_ratio * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[test_split:], indices[:test_split]

    # Create data indices for training and validation splits
    train_dataset_size = len(train_indices)
    val_split = int(np.floor(validation_ratio * train_dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)
    train_indices, val_indices = train_indices[val_split:], train_indices[:val_split]

    # Create data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create dataloaders for each set, which will allow loading batches
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    if get_indeces:
        # Return the data loaders and the indices of the sets
        return train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices
    else:
        # Just return the data loaders of each set
        return train_dataloader, val_dataloader, test_dataloader


def load_checkpoint(filepath):
    '''Load a model from a specified path and name.

    Parameters
    ----------
    filepath : str
        Path to the model being loaded, including it's own file name.

    Returns
    -------
    model : nn.Module
        The loaded model with saved weight values.
    '''
    checkpoint = torch.load(filepath)
    model = NeuralNetwork(checkpoint['n_inputs'],
                          checkpoint['n_hidden'],
                          checkpoint['n_outputs'],
                          checkpoint['n_layers'],
                          checkpoint['p_dropout'])
    model.load_state_dict(checkpoint['state_dict'])

    return model


def train(model, train_dataloader, val_dataloader, test_dataloader, seq_len_dict, batch_size=32, n_epochs=50, lr=0.001,
          model_path='models/', padding_value=999999, do_test=True, log_comet_ml=False, comet_ml_api_key=None, 
          comet_ml_project_name=None, comet_ml_workspace=None, comet_ml_save_model=False):
    '''Trains a given model on the provided data.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which is trained on the data to perform a 
        classification task.
    train_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches during training.
    val_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches when evaluating  
        the model's performance on a validation set during training.
    test_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches whe evaluating 
        the model's performance on a test set, after finishing the 
        training process.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
   batch_size : integer, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    n_epochs : integer, default 50
        Number of epochs, i.e. the number of times the training loop 
        iterates through all of the training data.
    lr : float, default 0.001
        Learning rate used in the optimization algorithm.
    model_path : string, default 'models/'
        Path where the model will be saved. By default, it saves in
        the directory named "models".
    padding_value : numeric
        Value to use in the padding, to fill the sequences.
    do_test : bool, default True
        If true, evaluates the model on the test set, after completing
        the training.
    log_comet_ml : bool, default False
        If true, makes the code upload a training report and metrics
        to comet.ml, a online platform which allows for a detailed 
        version control for machine learning models.
    comet_ml_api_key : string, default None
        Comet.ml API key used when logging data to the platform.
    comet_ml_project_name : string, default None
        Name of the comet.ml project used when logging data to the 
        platform.
    comet_ml_workspace : string, default None
        Name of the comet.ml workspace used when logging data to the 
        platform.
    comet_ml_save_model : bool, default False
        If set to true, uploads the model with the lowest validation loss
        to comet.ml when logging data to the platform.

    Returns
    -------
    model : nn.Module
        The same input model but with optimized weight values.
    '''
    if log_comet_ml:
        # Create a Comet.ml experiment
        experiment = Experiment(api_key=comet_ml_api_key, project_name=comet_ml_project_name, workspace=comet_ml_workspace)
        experiment.log_other("completed", False)

        # Report hyperparameters to Comet.ml
        hyper_params = {"batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "n_hidden": model.n_hidden,
                        "n_layers": model.n_layers,
                        "learning_rate": lr}
        experiment.log_parameters(hyper_params)

    optimizer = optim.Adam(model.parameters(), lr=lr)                       # Adam optimization algorithm
    step = 0                                                                # Number of iteration steps done so far
    print_every = 10                                                        # Steps interval where the metrics are printed
    val_loss_min = np.inf                                                   # Minimum validation loss
    train_on_gpu = torch.cuda.is_available()                                # Check if GPU is available

    for epoch in range(1, n_epochs+1):
        # Initialize the training metrics
        train_loss = 0
        train_acc = 0

        # Loop through the training data
        for features, labels in train_dataloader:
            model.train()                                                   # Activate dropout to train the model
            optimizer.zero_grad()                                           # Clear the gradients of all optimized variables

            if train_on_gpu:
                features, labels = features.cuda(), labels.cuda()           # Move data to GPU

            features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
            x_lengths = [seq_len_dict[patient] for patient in list(features[:, 0, 0].numpy())]  # Get the original lengths of the sequences
            data_sorted_idx = list(np.argsort(x_lengths)[::-1])             # Sorted indeces to get the data sorted by sequence length
            x_lengths = [x_lengths[idx] for idx in data_sorted_idx]         # Sort the x_lengths array by descending sequence length
            features = features[data_sorted_idx, :, :]                      # Sort the features by descending sequence length
            labels = labels[data_sorted_idx, :]                             # Sort the labels by descending sequence length
            scores, _ = model.forward(features[:, :, 2:], x_lengths)        # Feedforward the data through the model

            # Adjust the labels so that it gets the exact same shape as the predictions
            # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
            labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)  
            labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)

            loss = model.loss(scores, labels, x_lengths)                    # Calculate the cross entropy loss
            loss.backward()                                                 # Backpropagate the loss
            optimizer.step()                                                # Update the model's weights
            train_loss += loss                                              # Add the training loss of the current batch
            pred = torch.round(scores)                                      # Get the predictions
            correct_pred = pred == labels.contiguous().view_as(pred)        # Get the correct predictions
            mask = (labels <= 1).float()                                    # Create a mask by filtering out all labels that are not a padding value
            n_pred = int(torch.sum(mask).item())                            # Count how many predictions we have
            train_acc += torch.sum(correct_pred.type(torch.FloatTensor)) / n_pred  # Add the training accuracy of the current batch, ignoring all padding values
            step += 1                                                       # Count one more iteration step
            model.eval()                                                    # Deactivate dropout to test the model

            # Initialize the validation metrics
            val_loss = 0
            val_acc = 0

            # Loop through the validation data
            for features, labels in val_dataloader:
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
                    x_lengths = [seq_len_dict[patient] for patient in list(features[:, 0, 0].numpy())]  # Get the original lengths of the sequences
                    data_sorted_idx = list(np.argsort(x_lengths)[::-1])             # Sorted indeces to get the data sorted by sequence length
                    x_lengths = [x_lengths[idx] for idx in data_sorted_idx]         # Sort the x_lengths array by descending sequence length
                    features = features[data_sorted_idx, :, :]                      # Sort the features by descending sequence length
                    labels = labels[data_sorted_idx, :]                             # Sort the labels by descending sequence length
                    scores, _ = model.forward(features[:, :, 2:], x_lengths)        # Feedforward the data through the model

                    # Adjust the labels so that it gets the exact same shape as the predictions
                    # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
                    labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)  
                    labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)
            
                    val_loss += model.loss(scores, labels, x_lengths)               # Calculate and add the validation loss of the current batch
                    pred = torch.round(scores)                                      # Get the predictions
                    correct_pred = pred == labels.contiguous().view_as(pred)        # Get the correct predictions
                    mask = (labels <= 1).float()                                    # Create a mask by filtering out all labels that are not a padding value
                    n_pred = int(torch.sum(mask).item())                            # Count how many predictions we have
                    val_acc += torch.sum(correct_pred.type(torch.FloatTensor)) / n_pred  # Add the validation accuracy of the current batch, ignoring all padding values

            # Calculate the average of the metrics over the batches
            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)

            # [TODO] Also calculate the AUC metric

            # Display validation loss
            if step%print_every == 0:
                print(f'Epoch {epoch} step {step}: Validation loss: {val_loss}; Validation Accuracy: {val_acc}')

            # Check if the performance obtained in the validation set is the best so far (lowest loss value)
            if val_loss < val_loss_min:
                print(f'New minimum validation loss: {val_loss_min} -> {val_loss}.')

                # Update the minimum validation loss
                val_loss_min = val_loss

                # Get the current day and time to attach to the saved model's name
                current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')

                # Filename and path where the model will be saved
                model_filename = f'{model_path}checkpoint_{current_datetime}.pth'

                print(f'Saving model in {model_filename}')

                # Save the best performing model so far, a long with additional information to implement it
                checkpoint = {'n_inputs': model.n_inputs,
                              'n_hidden': model.n_hidden,
                              'n_outputs': model.n_outputs,
                              'n_layers': model.n_layers,
                              'p_dropout': model.p_dropout,
                              'state_dict': model.state_dict()}
                torch.save(checkpoint, model_filename)

                if log_comet_ml and comet_ml_save_model:
                    # Upload the model to Comet.ml
                    experiment.log_asset(file_path=model_filename, overwrite=True)

        # Calculate the average of the metrics over the epoch
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        if log_comet_ml:
            # Log metrics to Comet.ml
            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("train_acc", train_acc, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)
            experiment.log_metric("val_acc", val_acc, step=epoch)
            experiment.log_metric("epoch", epoch)
        
        # Print a report of the epoch
        print(f'Epoch {epoch}: Training loss: {train_loss}; Training Accuracy: {train_acc}; \
                Validation loss: {val_loss}; Validation Accuracy: {val_acc}')
        print('----------------------')

    if do_test:
        # [TODO] Make this inference part into a function

        # Load the model with the best validation performance
        model = load_checkpoint(model_filename)

        # Initialize the test metrics
        test_loss = 0
        test_acc = 0

        # Evaluate the model on the test set
        for features, labels in test_dataloader:
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                features, labels = features.float(), labels.float()             # Make the data have type float instead of double, as it would cause problems
                x_lengths = [seq_len_dict[patient] for patient in list(features[:, 0, 0].numpy())]  # Get the original lengths of the sequences
                data_sorted_idx = list(np.argsort(x_lengths)[::-1])             # Sorted indeces to get the data sorted by sequence length
                x_lengths = [x_lengths[idx] for idx in data_sorted_idx]         # Sort the x_lengths array by descending sequence length
                features = features[data_sorted_idx, :, :]                      # Sort the features by descending sequence length
                labels = labels[data_sorted_idx, :]                             # Sort the labels by descending sequence length
                scores, _ = model.forward(features[:, :, 2:], x_lengths)        # Feedforward the data through the model

                # Adjust the labels so that it gets the exact same shape as the predictions
                # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
                labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)  
                labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)
        
                test_loss += model.loss(scores, labels, x_lengths)              # Calculate and add the validation loss of the current batch
                pred = torch.round(scores)                                      # Get the predictions
                correct_pred = pred == labels.contiguous().view_as(pred)        # Get the correct predictions
                mask = (labels <= 1).float()                                    # Create a mask by filtering out all labels that are not a padding value
                n_pred = int(torch.sum(mask).item())                            # Count how many predictions we have
                test_acc += torch.sum(correct_pred.type(torch.FloatTensor)) / n_pred  # Add the test accuracy of the current batch, ignoring all padding values

        # Calculate the average of the metrics over the batches
        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)

        # [TODO] Also calculate the AUC metric

        if log_comet_ml:
            # Log metrics to Comet.ml
            experiment.log_metric("test_loss", test_loss, step=step)
            experiment.log_metric("test_acc", test_acc, step=step)
    
    if log_comet_ml:
        # Only report that the experiment completed successfully if it finished the training without errors
        experiment.log_other("completed", True)

    return model
