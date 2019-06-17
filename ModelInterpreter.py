import torch                            # PyTorch to create and apply deep learning models
from torch.autograd import Variable     # Create optimizable PyTorch variables
import pandas as pd                     # Pandas to handle the data in dataframes
import numpy as np                      # NumPy to handle numeric and NaN operations
import shap                             # Module used for the calculation of approximate Shapley values
import warnings                         # Print warnings for bad practices
from tqdm import tqdm                   # tqdm allows to track code execution progress
from tqdm import tqdm_notebook          # tqdm allows to track code execution progress
import time                             # Calculate code execution time
import utils                            # Contains auxiliary functions
from Time_Series_Dataset import Time_Series_Dataset     # Dataset subclass which allows the creation of Dataset objects
import plotly                           # Plotly for interactive and pretty plots
import plotly.graph_objs as go
import plotly.offline as py
import colorlover as cl                 # Get colors from colorscales

if utils.in_ipynb:
    plotly.offline.init_notebook_mode(connected=True)

# Constants
POS_COLOR = 'rgba(255,13,87,1)'
NEG_COLOR = 'rgba(30,136,229,1)'

class ModelInterpreter:
    def __init__(self, model, data, labels, seq_len_dict=None, id_column=0, inst_column=1,
                 fast_calc=True, SHAP_bkgnd_samples=1000, random_seed=42,
                 feat_names=None, padding_value=999999, occlusion_wgt=0.7):
        '''A machine learning model interpreter which calculates instance and
        feature importance.

        The current focus of the class is to analyze neural networks built in
        the PyTorch framework, which classify sequential data with potentially
        variable sequence length.

        Parameters
        ----------
        model : nn.Module
            Machine learning model which will be interpreted.
        data : torch.Tensor or pandas.DataFrame
            Data used in the interpretation, either directly by analyzing the
            outputs obtained with each sample or indireclty by using as
            background data in methods such as SHAP explainers. The data will be
            used in PyTorch tensor format, but the user can submit it as a
            pandas dataframe, which is then automatically padded and converted.
        labels : torch.Tensor, default None
            Labels corresponding to the data used, either specified in the input
            or all the data that the interpreter has.
        seq_len_dict : dict, default None
            Dictionary containing the sequence lengths for each index of the
            original dataframe. This allows to ignore the padding done in
            the fixed sequence length tensor.
        id_column : int, default 0
            Number of the column which corresponds to the subject identifier in
            the data tensor.
        inst_column : int, default 1
            Number of the column which corresponds to the instance or timestamp
            identifier in the data tensor.
        fast_calc : bool, default True
            If set to True, the algorithm uses simple mask filters, occluding
            instances and replacing features with reference values, in order
            to do a fast interpretation of the model. If set to False, SHAP
            values are used for a more precise and truthful interpretation of
            the model's behavior, requiring longer computation times.
        SHAP_bkgnd_samples : int, default 1000
            Number of samples to use as background data, in case a SHAP
            explainer is applied (fast_calc must be set to False).
        random_seed : integer, default 42
            Seed used when shuffling the data.
        feat_names : list of string, default None
            Column names of the dataframe associated to the data. If no list is
            provided, the dataframe should be given in the data argument, so as
            to fetch the names of the columns.
        padding_value : numeric
            Value to use in the padding, to fill the sequences.
        occlusion_wgt : float, default 0.75
            Weight given to the occlusion part of the instance importance score.
            This scores is calculated as a weighted average of the instance's
            influence on the final output and the variation of the output
            probability, between the current instance and the previous one. As
            such, this weight should have a value between 0 and 1, with the
            output variation receiving the remaining weight (1 - occlusion_wgt),
            where 0 corresponds to not using the occlusion component at all, 0.5
            is a normal, unweighted average and 1 deactivates the use of the
            output variation part.
        '''
        # Initialize parameters according to user input
        self.model = model
        self.seq_len_dict = seq_len_dict
        self.id_column = id_column
        self.inst_column = inst_column
        self.fast_calc = fast_calc
        self.SHAP_bkgnd_samples = SHAP_bkgnd_samples
        self.random_seed = random_seed
        self.feat_names = feat_names
        self.padding_value = padding_value
        self.occlusion_wgt = occlusion_wgt

        # Put the model in evaluation mode to deactivate dropout
        self.model.eval()

        if type(data) is torch.Tensor:
            self.data = data
            self.labels = labels
        elif type(data) is pd.DataFrame:
            n_ids = data.iloc[:, self.id_column].nunique()      # Total number of unique sequence identifiers
            n_features = len(data.columns)                      # Number of input features

            # Find the sequence lengths of the data
            self.seq_len_dict = self.calc_seq_len_dict(data)

            # Pad data (to have fixed sequence length) and convert into a PyTorch tensor
            data_tensor = utils.dataframe_to_padded_tensor(data, self.seq_len_dict, n_ids,
                                                           n_features, padding_value=padding_value)

            # Separate labels from features
            dataset = Time_Series_Dataset(data_tensor, data)
            self.data = dataset.X
            self.labels = dataset.y

            if feat_names is None:
                # Fetch the column names, ignoring the id and instance id ones
                self.feat_names = list(data.columns)
                self.feat_names.remove(self.feat_names[self.id_column])
                self.feat_names.remove(self.feat_names[self.inst_column])
        else:
            raise Exception('ERROR: Invalid data type. Please provide data in a Pandas DataFrame or PyTorch Tensor format.')

        # Declare explainer attribute which will store the SHAP DEEP Explainer object
        self.explainer = None

        # Declare attributes that will store importance scores (instance and feature importance)
        self.inst_scores = None
        self.feat_scores = None

    def find_subject_idx(subject_id, data=None, subject_id_col=0):
        '''Find the index that corresponds to a given subject in a data tensor.

        Parameters
        ----------
        subject_id : int or string
            Unique identifier of the subject whose index on the data tensor one
            wants to find out.
        data : torch.Tensor, default None
            PyTorch tensor containing the data on which the subject's index will be
            searched for. If not specified, all data known to the model
            interpreter will be used.
        subject_id_col : int, default 0
            The number of the column in the data tensor that stores the subject
            identifiers. If not specified, subject id column number previously
            defined in the model interpreter will be used.

        Returns
        -------
        idx : int
            Index where the specified subject appears in the data tensor.'''
        if data is None:
            # If a subset of data to interpret isn't specified, the interpreter will use all the data
            data = self.data

        if subject_id_col is None:
            # If the subject id column number isn't specified, the interpreter
            # will use its previously defined one
            subject_id_col = self.id_column

        return (data[:, 0, subject_id_col] == subject_id).nonzero().item()

    def create_bkgnd_test_sets(self, shuffle_dataset=True):
        '''Distributes the data into background and test sets and returns
        the respective data tensors.

        Parameters
        ----------
        random_seed : integer, default 42
            Seed used when shuffling the data.
        shuffle_dataset : bool, default True
            If set to True, the data of which set is shuffled.

        Returns
        -------
        bkgnd_data : torch.Tensor, default None
            Background data used in the SHAP explainer to estimate conditional
            expectations.
        test_data : torch.Tensor, default None
            A subset of data on which model interpretation will be made (i.e.
            calculating feature and/or instance importance).
        '''
        # Create data indices for training and test splits
        dataset_size = self.data.shape[0]
        indices = list(range(dataset_size))
        if shuffle_dataset:
            # Shuffle data
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        bkgnd_indices, test_indices = indices[:self.SHAP_bkgnd_samples], indices[self.SHAP_bkgnd_samples:]

        # Get separate tensors for the background data and the test data
        bkgnd_data = self.data[bkgnd_indices]
        test_data = self.data[test_indices]
        return bkgnd_data, test_data

    def calc_seq_len_dict(self, df):
        '''Create a dictionary that contains the sequence length for each
        sequence identifier in the input data.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe that will be used to identify sequence lengths.

        Returns
        -------
        seq_len_dict : dict
            Dictionary containing the original sequence lengths of the dataframe.
        '''
        # Dictionary containing the sequence length (number of temporal events) of each sequence (e.g. patient)
        seq_len_df = df.groupby(df.columns[self.id_column])[df.columns[self.inst_column]].count() \
                                                                                          .to_frame() \
                                                                                          .sort_values(by=df.columns[self.inst_column], ascending=False)
        seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.values))])
        return seq_len_dict

    def mask_filter_step(self, mask, optimizer, data, l1_coeff=0, hidden_state=None):
        '''Perform a single optimization step to calculate a new version of the
        mask filter.

        Parameters
        ----------
        mask : numpy.Array
            Current mask filter, either the initial one or from the previous
            optimization iteration.
        optimizer : torch.optim
            Optimizer used in the calculation of the mask filter.
        data : torch.Tensor
            Data sample which will be used to determine the most relevant
            features. In case of multivariate sequential data, this must be a
            single instance of a sequence.
        l1_coeff : int, default 1
            Weight given in the loss function to the L1 norm of the mask filter.
        hidden_state : torch.Tensor or tuple of two torch.Tensor, default None
            Hidden state coming from the previous recurrent cell. If none is
            specified, the hidden state is initialized as zero.

        Returns
        -------
        mask : numpy.Array
            Current mask filter, after the performed optimization step.
        optimizer : torch.optim
            Optimizer used in the calculation of the mask filter, updated with
            the performed step.
        '''
        # # Clear the gradients of all optimized variables
        # optimizer.zero_grad()

        # Get the model's output for the input data
        output = self.model((mask * data).unsqueeze(0).unsqueeze(0), hidden_state=hidden_state)

        # Calculate the loss function
        loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + output

        # Backpropagate the loss function and run an optimization step (update the mask filter)
        loss.backward()
        mask.grad = utils.change_grad((-1) * mask.grad, mask.data)
        mask.data = mask.data + mask.grad
        # optimizer.step()

        # Make sure that the mask has values either 0 or 1
        mask.data.clamp_(0, 1)
        mask.data.round_()

        return mask, optimizer

    # [TODO] Confirm that the mask filter is working in every scenario
    def mask_filter(self, data=None, x_lengths=None, max_iter=500, l1_coeff=0,
                    lr=0.001, recur_layer=None, see_progress=True):
        '''Calculate a mask filter for the given data samples, through an
        appropriate optimization.

        Parameters
        ----------
        data : torch.Tensor, default None
            Data sample(s) which will be used to determine the most relevant
            features. In case of multivariate sequential data, each instance will
            be analyzed seperately. If None, all data known to the model
            interpreter will be used.
        x_lengths : list of int
            Sorted list of sequence lengths, relative to the input data.
        max_iter : int, default 500
            Maximum number of iterations of the mask filter optimization, for
            each instance.
        l1_coeff : int, default 1
            Weight given in the loss function to the L1 norm of the mask filter.
        lr : float, default 0.001
            Learning rate used in the optimization algorithm.
        recur_layer : torch.nn.LSTM or torch.nn.GRU or torch.nn.RNN, default None
            Pointer to the recurrent layer in the model, if it exists. It should
            either be a LSTM, GRU or RNN network. If none is specified, the
            method will automatically search for a recurrent layer in the model.
        see_progress : bool, default True
            If set to True, a progress bar will show up indicating the execution
            of the feature importance scores calculations.

        Returns
        -------
        mask : numpy.Array
            Output mask, after finishing the optimization for every specified
            sample. It will be inverted before returning, so as to be an array
            filled with zeros, except in the indeces corresponding to the most
            relevant features, where it will be one.
        '''
        # [TODO] Work on an option to use input data different from multivariate sequential

        if data is None:
            # If a subset of data to interpret isn't specified, the interpreter will use all the data
            data = self.data

        if x_lengths is None:
            # Sort the data by sequence length
            data, x_lengths = utils.sort_by_seq_len(data, self.seq_len_dict)

        if len(data.shape) > 1 and recur_layer is None:
            # Search for a recurrent layer
            if hasattr(self.model, 'lstm'):
                recur_layer = self.model.lstm
            elif hasattr(self.model, 'gru'):
                recur_layer = self.model.gru
            elif hasattr(self.model, 'rnn'):
                recur_layer = self.model.rnn
            else:
                raise Exception('ERROR: No recurrent layer found. Please specify it in the recur_layer argument.')

        # Confirm that the model is in evaluation mode to deactivate dropout
        self.model.eval()

        # Create a mask filter variable, initialized as an all ones tensor
        mask = torch.ones(data.shape)

        if len(data.shape) == 3:
            # Loop to go through each sequence in the input data
            for seq in utils.iterations_loop(range(data.shape[0]), see_progress=see_progress):
                # Get the true length of the current sequence
                seq_len = x_lengths[seq]

                # Loop to go through each instance in the input sequence
                for inst in utils.iterations_loop(range(seq_len), see_progress=see_progress):
                    hidden_state = None
                    # Get the hidden state that the model receives as an input
                    if inst > 0:
                        # Get the hidden state outputed from the previous recurrent cell
                        _, hidden_state = recur_layer(data[:inst-1])

                    # Temporary mask filter for he current instance
                    tmp_mask = Variable(mask[seq, inst, :], requires_grad=True)

                    # # Instantiate an optimizer that will calculate the optimal mask filter for the current instance
                    # optimizer = torch.optim.Adam([tmp_mask], lr=lr)

                    # Mask filter optimization loop
                    for iter in utils.iterations_loop(range(max_iter), see_progress=see_progress):
                        # Perform a single optimization step
                        tmp_mask, optimizer = self.mask_filter_step(tmp_mask, optimizer, data[seq, inst, :], l1_coeff, hidden_state)

                    # Save the optimized mask filter of the current instance
                    mask[seq, inst, :] = tmp_mask

        elif len(data.shape) == 2:
            # Loop to go through each instance in the input sequence
            for inst in utils.iterations_loop(range(data.shape[0]), see_progress=see_progress):
                hidden_state = None
                # Get the hidden state that the model receives as an input
                if inst > 0:
                    # Get the hidden state outputed from the previous recurrent cell
                    _, hidden_state = recur_layer(data[:inst-1])

                # Temporary mask filter for he current instance
                tmp_mask = Variable(mask[inst], requires_grad=True)

                # # Instantiate an optimizer that will calculate the optimal mask filter for the current instance
                # optimizer = torch.optim.Adam([tmp_mask.requires_grad_()], lr=lr)

                # Mask filter optimization loop
                for iter in utils.iterations_loop(range(max_iter), see_progress=see_progress):
                    # Perform a single optimization step
                    tmp_mask, optimizer = self.mask_filter_step(tmp_mask, optimizer, data[inst], l1_coeff, hidden_state)

                # Save the optimized mask filter of the current instance
                mask[inst] = tmp_mask

        elif len(data.shape) == 1:
            # Make sure that the mask can be optimized properly
            mask.requires_grad_()

            # # Instantiate an optimizer that will calculate the optimal mask filter
            # optimizer = torch.optim.Adam([mask], lr=lr)

            # Mask filter optimization loop
            for iter in utils.iterations_loop(range(max_iter), see_progress=see_progress):
                # Perform a single optimization step
                mask, optimizer = self.mask_filter_step(mask, optimizer, data, l1_coeff)

        else:
            raise Exception(f'ERROR: Can\'t handle data with more than 3 dimensions. \
                              Submitted data with {len(data.shape)} dimensions.')

        # Return the inverted version of the mask, to atrribute 1 (one) to the most relevant features
        return 1 - mask

    def instance_importance(self, data=None, labels=None, x_lengths=None,
                            see_progress=True, occlusion_wgt=None):
        '''Calculate the instance importance scores to interpret the impact of
        each instance of a sequence on the final output.

        Parameters
        ----------
        data : torch.Tensor, default None
            Optionally, the user can specify a subset of data on which model
            interpretation will be made (i.e. calculating feature and/or
            instance importance). Otherwise, all the data is used.
        labels : torch.Tensor, default None
            Labels corresponding to the data used, either specified in the input
            or all the data that the interpreter has.
        x_lengths : list of int
            Sorted list of sequence lengths, relative to the input data.
        see_progress : bool, default True
            If set to True, a progress bar will show up indicating the execution
            of the instance importance scores calculations.
        occlusion_wgt : float, default None
            Weight given to the occlusion part of the instance importance score.
            This scores is calculated as a weighted average of the instance's
            influence on the final output and the variation of the output
            probability, between the current instance and the previous one. As
            such, this weight should have a value between 0 and 1, with the
            output variation receiving the remaining weight (1 - occlusion_wgt),
            where 0 corresponds to not using the occlusion component at all, 0.5
            is a normal, unweighted average and 1 deactivates the use of the
            output variation part. If the value wasn't specified in the
            intepreter's initialization nor in the method argument, it will
            default to 0.7

        Returns
        -------
        inst_scores : numpy.Array
            Array containing the importance scores of each instance in the
            given input sequences.
        '''
        if occlusion_wgt is None:
            if self.occlusion_wgt is not None:
                # Set to the class's occlusion weight value
                occlusion_wgt = self.occlusion_wgt
            else:
                # Set the occlusion weight value to 0.7 (default)
                occlusion_wgt = 0.7
                self.occlusion_wgt = occlusion_wgt

        # Confirm that the occlusion weight has a valid value (between 0 and 1)
        if occlusion_wgt > 1 or occlusion_wgt < 0:
            raise Exception(f'ERROR: Inserted invalid occlusion weight value {occlusion_wgt}. Please replace with a value between 0 and 1.')

        if occlusion_wgt < 1:
            # If the output variation is used in the calculation of the score,
            # get the reference outputs for all the instances of the sequences
            seq_final_outputs = False
        else:
            # Otherwise, only the final outputs of the sequences are retrieved
            seq_final_outputs = True

        if data is None:
            # If a subset of data to interpret isn't specified, the interpreter will use all the data
            data = self.data
            labels = self.labels

        # Make sure that the data is in type float
        data = data.float()

        # Model output when using all the original instances in the input sequences
        ref_output, _ = utils.model_inference(self.model, self.seq_len_dict,
                                              data=(data, labels), metrics=[''],
                                              seq_final_outputs=seq_final_outputs,
                                              cols_to_remove=[self.id_column, self.inst_column])

        if not seq_final_outputs:
            # Organize the stacked outputs to become a list of outputs for each sequence
            x_lengths_arr = np.array(x_lengths)
            x_lengths_cum = np.cumsum(x_lengths_arr)
            start_idx = np.roll(x_lengths_cum, 1)
            start_idx[0] = 0
            end_idx = x_lengths_cum
            ref_output = [ref_output[start_idx[i]:end_idx[i]] for i in range(len(start_idx))]

        def calc_instance_score(sequence_data, instance, ref_output, x_length, occlusion_wgt):
            # Remove identifier columns from the data
            features_idx = list(range(sequence_data.shape[1]))
            features_idx.remove(self.id_column)
            features_idx.remove(self.inst_column)
            sequence_data = sequence_data[:, features_idx]

            # Indeces without the instance that is being analyzed
            instances_idx = list(range(sequence_data.shape[0]))
            instances_idx.remove(instance)

            # Sequence data without the instance that is being analyzed
            sequence_data = sequence_data[instances_idx, :]

            # Add a third dimension for the data to be readable by the model
            sequence_data = sequence_data.unsqueeze(0)

            # Calculate the output without the instance that is being analyzed
            new_output = self.model(sequence_data, [x_length-1])

            # Only use the last output (i.e. the one from the last instance of the sequence)
            new_output = new_output[-1].item()

            # Flag that indicates whether the output variation component will be used in the instance importance score
            # (in a weighted average)
            use_outvar_score = ref_output.shape[0] > 1

            if use_outvar_score:
                # Get the output from the previous instance
                prev_output = ref_output[instance-1].item()

                # Get the output from the current instance
                curr_output = ref_output[instance].item()

                # Get the last output
                ref_output = ref_output[x_length-1].item()
            else:
                ref_output = ref_output.item()

            # The instance importance score is then the difference between the output probability with the instance
            # and the probability without the instance
            inst_score = ref_output - new_output

            if instance > 0 and use_outvar_score:
                # If it's not the first instance, add the output variation characteristic in a weighted average
                inst_score = occlusion_wgt * inst_score + (1 - occlusion_wgt) * (curr_output - prev_output)

            # Apply a tanh function to make even the smaller scores (which are the most frequent) more salient
            inst_score = np.tanh(4 * inst_score)

            return inst_score

        inst_scores = [[calc_instance_score(data[seq_num, :, :], inst, ref_output[seq_num], x_lengths[seq_num], occlusion_wgt)
                        for inst in range(x_lengths[seq_num])] for seq_num in utils.iterations_loop(range(data.shape[0]), see_progress=see_progress)]
        # DEBUG
        # inst_scores = []
        # for seq_num in utils.iterations_loop(range(data.shape[0]), see_progress=see_progress):
        #     tmp_list = []
        #     for inst in range(x_lengths[seq_num]):
        #         tmp_list.append(calc_instance_score(data[seq_num, :, :], inst, ref_output[seq_num], x_lengths[seq_num], occlusion_wgt))
        #     inst_scores.append(tmp_list)

        # Pad the instance scores lists so that all have the same length
        inst_scores = [utils.pad_list(scores_list, data.shape[1], padding_value=self.padding_value) for scores_list in inst_scores]

        # Convert to a NumPy array
        inst_scores = np.array(inst_scores)
        return inst_scores

    def feature_importance(self, test_data=None, fast_calc=None,
                           see_progress=True, bkgnd_data=None, max_iter=500,
                           l1_coeff=0, lr=0.001, recur_layer=None):
        '''Calculate the feature importance scores to interpret the impact
        of each feature in each instance's output.

        Parameters
        ----------
        test_data : torch.Tensor, default None
            Optionally, the user can specify a subset of data on which model
            interpretation will be made (i.e. calculating feature and/or
            instance importance). Otherwise, all the data is used.
        fast_calc : bool, default None
            If set to True, the algorithm uses simple mask filters, occluding
            instances and replacing features with reference values, in order
            to do a fast interpretation of the model. If set to False, SHAP
            values are used for a more precise and truthful interpretation of
            the model's behavior, requiring longer computation times.
        see_progress : bool, default True
            If set to True, a progress bar will show up indicating the execution
            of the feature importance scores calculations.

        if fast_calc is False:

        bkgnd_data : torch.Tensor, default None
            In case of setting fast_calc to True, which makes the algorithm use
            SHAP in the feature importance, the background data used in the
            explainer can be set through this parameter.

        if fast_calc is True:

        max_iter : int, default 500
            Maximum number of iterations of the mask filter optimization, for
            each instance.
        l1_coeff : int, default 1
            Weight given in the loss function to the L1 norm of the mask filter.
        lr : float, default 0.001
            Learning rate used in the optimization algorithm of the mask filter.
        recur_layer : torch.nn.LSTM or torch.nn.GRU or torch.nn.RNN, default None
            Pointer to the recurrent layer in the model, if it exists. It should
            either be a LSTM, GRU or RNN network. If none is specified, the
            method will automatically search for a recurrent layer in the model.

        Returns
        -------
        feat_scores : numpy.Array
            Array containing the importance scores of each feature, of each
            instance, in the given input sequences.
        '''
        if fast_calc is None:
            # Use the predefined option if fast_calc isn't set in the function call
            fast_calc = self.fast_calc

        # Sort the test data by sequence length
        test_data, x_lengths_test = utils.sort_by_seq_len(test_data, self.seq_len_dict)

        # Remove identifier columns from the test data
        features_idx = list(range(test_data.shape[2]))
        features_idx.remove(self.id_column)
        features_idx.remove(self.inst_column)
        test_data = test_data[:, :, features_idx]

        # Make sure that the test data is in type float
        test_data = test_data.float()

        if not fast_calc:
            print(f'Attention: you have chosen to interpret the model using SHAP, with {self.SHAP_bkgnd_samples} background samples applied to {test_data.shape[0]} test samples. This might take a while. Depending on your computer\'s processing power, you should do a coffee break or even go to sleep!')

            # Sort the background data by sequence length
            bkgnd_data, x_lengths_bkgnd = utils.sort_by_seq_len(bkgnd_data, self.seq_len_dict)

            # Remove identifier columns from the background data
            bkgnd_data = bkgnd_data[:, :, features_idx]

            # Make sure that the background data is in type float
            bkgnd_data = bkgnd_data.float()

            # Use the background dataset to integrate over
            self.explainer = shap.DeepExplainer(self.model, bkgnd_data, feedforward_args=[x_lengths_bkgnd])

            # Count the time that takes to calculate the SHAP values
            start_time = time.time()

            # Explain the predictions of the sequences in the test set
            feat_scores = self.explainer.shap_values(test_data,
                                                     feedforward_args=[x_lengths_bkgnd, x_lengths_test],
                                                     var_seq_len=True, see_progress=see_progress)
            print(f'Calculation of SHAP values took {time.time() - start_time} seconds')
            return feat_scores

        else:
            # Count the time that takes to calculate the SHAP values
            start_time = time.time()

            # Apply mask filter
            feat_scores = self.mask_filter(test_data, x_lengths_test, max_iter,
                                           l1_coeff, lr, recur_layer, see_progress=see_progress)
            print(f'Calculation of mask filter values took {time.time() - start_time} seconds')

            return feat_scores

    # [Bonus TODO] Upload model explainer and interpretability plots to Comet.ml
    def interpret_model(self, bkgnd_data=None, test_data=None, test_labels=None, new_data=False,
                        df=None, instance_importance=True, feature_importance=False,
                        fast_calc=None, see_progress=True, save_data=True):
        '''Method to calculate scores of feature and/or instance importance, in
        order to be able to interpret a model on a given data.

        Parameters
        ----------
        bkgnd_data : torch.Tensor, default None
            In case of setting fast_calc to True, which makes the algorithm use
            SHAP in the feature importance, the background data used in the
            explainer can be set through this parameter as a PyTorch tensor.
        test_data : torch.Tensor, default None
            Optionally, the user can specify a subset of data on which model
            interpretation will be made (i.e. calculating feature and/or
            instance importance) as a PyTorch tensor. Otherwise, all the data is
            used.
        test_labels : torch.Tensor, default None
            Labels corresponding to the data used, either specified in the input
            or all the data that the interpreter has.
        new_data : bool, default False
            If set to True, it indicates that the data that will be interpreted
            hasn't been seen before, i.e. it has different ids than those in the
            original dataset defined in the object initialization. This implies
            that a dataframe of the new data is provided (parameter df) so that
            the sequence lengths are calculated. Otherwise, the original
            sequence lengths known by the model interpreter are used.
        df : pandas.DataFrame, default None
            Dataframe containing the new data so as to calculate the sequence
            lengths of the new ids. Only used if new_data is set to True.
        instance_importance : bool, default True
            If set to True, instance importance is made on the data. In other
            words, the algorithm will analyze the impact that each instance of
            an input sequence had on the output.
        feature_importance : bool, default False
            If set to True, feature importance is made on the data. In other
            words, the algorithm will analyze the impact that each feature of
            an instance had on the output. This is analyzed instance by instance,
            not in the entire sequence at once. For example, from the feature
            importance alone, it's not straightforward how a value in a previous
            instance impacted the current output.
        fast_calc : bool, default None
            If set to True, the algorithm uses simple mask filters, occluding
            instances and replacing features with reference values, in order
            to do a fast interpretation of the model. If set to False, SHAP
            values are used for a more precise and truthful interpretation of
            the model's behavior, requiring longer computation times.
        see_progress : bool, default True
            If set to True, a progress bar will show up indicating the execution
            of the instance importance scores calculations.
        save_data : bool, default True
            If set to True, the possible background data (used in the SHAP
            explainer) and the test data (on which importance scores are
            calculated) are saved as object attributes.

        Returns
        -------
        inst_scores : numpy.Array
            Array containing the importance scores of each instance in the
            given input sequences. Only calculated if instance_importance is set
            to True.
        feat_scores : numpy.Array
            Array containing the importance scores of each feature, of each
            instance, in the given input sequences. Only calculated if
            feature_importance is set to True.
        '''
        # Confirm that the model is in evaluation mode to deactivate dropout
        self.model.eval()

        if fast_calc is None:
            # Use the predefined option if fast_calc isn't set in the function call
            fast_calc = self.fast_calc

        if test_data is None:
            if fast_calc:
                # If a subset of data to interpret isn't specified, the interpreter will use all the data
                test_data = self.data
                test_labels = self.labels
            else:
                if bkgnd_data is None:
                    # Get the background and test sets from the dataset
                    bkgnd_data, test_data = self.create_bkgnd_test_sets()
                else:
                    # Get the test set from the dataset
                    _, test_data = self.create_bkgnd_test_sets()

        if new_data:
            if df is None:
                raise Exception('ERROR: A dataframe must be provided in order to \
                                 work with the new data.')

            # Find the sequence lengths of the new data
            seq_len_dict = self.calc_seq_len_dict(df)

            # Sort the data by sequence length
            test_data, test_labels, x_lengths_test = utils.sort_by_seq_len(test_data, seq_len_dict, test_labels)
        else:
            # Sort the data by sequence length
            test_data, test_labels, x_lengths_test = utils.sort_by_seq_len(test_data, self.seq_len_dict, test_labels)

        if not fast_calc and bkgnd_data is None:
            # Get the background set from the dataset
            bkgnd_data, _ = self.create_bkgnd_test_sets()

        if save_data:
            # Save the data used in the model interpretation
            self.bkgnd_data = bkgnd_data
            self.bkgnd_data = test_data

        if instance_importance:
            print('Calculating instance importance scores...')
            # Calculate the scores of importance of each instance
            self.inst_scores = self.instance_importance(test_data, test_labels, x_lengths_test, see_progress)

        if feature_importance:
            print('Calculating feature importance scores...')
            # Calculate the scores of importance of each feature in each instance
            self.feat_scores = self.feature_importance(test_data, fast_calc, see_progress, bkgnd_data)

        print('Done!')

        if instance_importance and feature_importance:
            return self.inst_scores, self.feat_scores
        elif instance_importance and not feature_importance:
            return self.inst_scores
        elif not instance_importance and feature_importance:
            return self.feat_scores
        else:
            warnings.warn('Without setting instance_importance nor feature_importance \
                           to True, the interpret_model function won\'t do anything relevant.')
            return


    def instance_importance_plot(self, orig_data=None, inst_scores=None, id=None,
                                 pred_prob=None, show_pred_prob=True, labels=None,
                                 seq_len=None, threshold=0, get_fig_obj=False,
                                 tensor_idx=True):
        '''Create a bar chart that allows visualizing instance importance scores.

        Parameters
        ----------
        orig_data : torch.Tensor or numpy.Array, default None
            Original data used in the machine learning model. Used here to fetch
            the true ID corresponding to the plotted sequence.
        inst_scores : numpy.Array, default None
            Array containing the instance importance scores to be plotted.
        id : int, default None
            ID or sequence index that select which time series / sequences to
            use in the plot. If it's a single value, the method plots a single
        pred_prob : numpy.Array or torch.Tensor or list of floats, default None
            Array containing the prediction probabilities for each sequence in
            the input data (orig_data). Only relevant if show_pred_prob is True.
        show_pred_prob : bool, default True
            If set to true, a percentage bar chart will be shown to the right of
            the standard instance importance plot.
        labels : torch.Tensor, default None
            Labels corresponding to the data used, either specified in the input
            or all the data that the interpreter has.
        seq_len : int, default None
            Sequence lengths which represent the true, unpadded size of the
            input sequences.
        threshold : int or float, default 0
            Value to use as a threshold in the plot's color selection. In other
            words, values that exceed this threshold will have one color while the
            remaining have a different one, as specified in the parameters.
        get_fig_obj : bool, default False
            If set to True, the function returns the object that contains the
            displayed plotly figure.
        tensor_idx : bool, default True
            If set to True, the ID specified in the respective parameter
            constitutes the index where the desired sequence resides. Otherwise,
            it's the actual unique identifier that appears in the original data.

        Returns
        -------
        fig : plotly.graph_objs.Figure or None
            If argument get_fig_obj is set to True, the figure object is returned.
            Otherwise, nothing is returned, only the plot is showned.'''
        if orig_data is None:
            # Use all the data if none was specified
            orig_data = self.data

            if labels is None:
                labels = self.labels

        if inst_scores is None:
            if self.inst_scores is None:
                raise Exception('ERROR: No instance importance scores found. If the scores aren\'t specified, then they must have already been calculated through the interpret_model method.')

            # Use all the previously calculated scores if none were specified
            inst_scores = self.inst_scores

        # Plot the instance importance of multiple sequences
        # Convert the instance scores data into a NumPy array
        if type(inst_scores) is torch.Tensor:
            inst_scores = inst_scores.detach().numpy()
        elif type(inst_scores) is list:
            inst_scores = np.array(inst_scores)

        if pred_prob is None and show_pred_prob is True:
            if labels is None:
                raise Exception('ERROR: By setting show_pred_prob to True, either the prediction probabilities (pred_prob) or the labels must be provided.')

            # Calculate the prediction probabilities for the provided data
            pred_prob, _ = utils.model_inference(self.model, self.seq_len_dict,
                                                 data=(orig_data, labels),
                                                 metrics=[''], seq_final_outputs=True,
                                                 cols_to_remove=[self.id_column, self.inst_column])

        # Convert the prediction probability data into a NumPy array
        if type(pred_prob) is torch.Tensor:
            pred_prob = pred_prob.detach().numpy()
        elif type(pred_prob) is list:
            pred_prob = np.array(pred_prob)

        # if is not tensor_idx:
        # [TODO] Search for the index associated to the specific ID asked for by the user
        # [TODO] Allow to search for multiple indeces and generate a multiple patients time series plot from it

        if len(inst_scores.shape) == 1 or (id is not None and type(id) is not list):
            # True sequence length of the current id's data
            if seq_len is None:
                seq_len = self.seq_len_dict[orig_data[id, 0, self.id_column].item()]

            # [TODO] Add a prediction probability bar plot like in the multiple sequences case

            # Plot the instance importance of one sequence
            plot_data = [go.Bar(
                            x = list(range(seq_len)),
                            y = inst_scores[id, :seq_len],
                            marker=dict(color=utils.set_bar_color(inst_scores, id, seq_len,
                                                                  threshold=threshold,
                                                                  pos_color=POS_COLOR,
                                                                  neg_color=NEG_COLOR))
                          )]
            layout = go.Layout(
                                title=f'Instance importance scores for ID {int(orig_data[id, 0, self.id_column])}',
                                xaxis=dict(title='Instance'),
                                yaxis=dict(title='Importance scores')
                              )
        else:
            if id is None:
                # Use all the sequences data if a subset isn't specified
                id = list(range(inst_scores.shape[0]))

            # Select the desired data according to the specified IDs
            inst_scores = inst_scores[id, :]
            orig_data = orig_data[id, :, :]
            pred_prob = pred_prob[id]

            # Unique patient ids in string format
            patients = [str(int(item)) for item in [tensor.item()
                        for tensor in list(orig_data[:, 0, self.id_column])]]

            # Sequence instances count, used as X in the plot
            seq_insts_x = [list(range(inst_scores.shape[1]))
                           for patient in range(len(patients))]

            # Patients ids repeated max sequence length times, used as Y in the plot
            patients_y = [[patient]*inst_scores.shape[1] for patient in list(patients)]

            # Flatten seq_insts and patients_y
            seq_insts_x = list(np.array(seq_insts_x).flatten())
            patients_y = list(np.array(patients_y).flatten())

            # Define colors for the data points based on their normalized scores (from 0 to 1 instead of -1 to 1)
            colors = [val for val in inst_scores.flatten()]

            # Count the number of already deleted paddings
            count = 0

            for i in range(inst_scores.shape[0]):
                for j in range(inst_scores.shape[1]):
                    if inst_scores[i, j] == self.padding_value:
                        # Delete elements that represent paddings, not real instances
                        del seq_insts_x[i*inst_scores.shape[1]+j-count]
                        del patients_y[i*inst_scores.shape[1]+j-count]
                        del colors[i*inst_scores.shape[1]+j-count]

                        # Increment the counting of already deleted items
                        count += 1

            # Colors to use in the prediction probability bar plots
            pred_colors = cl.scales['8']['div']['RdYlGn']

            # Create "percentage bar" plots through pairs of unfilled and filled rectangles
            shapes_list = []

            # Starting y coordinate of the first shape
            y0 = -0.25

            # Height of the shapes (y length)
            step = 0.5

            # Maximum width of the shapes
            max_width = 1

            for i in range(len(patients)):
                # Set the starting x coordinate to after the last data point
                x0 = inst_scores.shape[1]

                # Set the filling length of the shape
                x1_fill = x0 + pred_prob[i] * max_width

                shape_unfilled = {
                                    'type': 'rect',
                                    'x0': x0,
                                    'y0': y0,
                                    'x1': x0 + max_width,
                                    'y1': y0 + step,
                                    'line': {
                                                'color': 'rgba(0, 0, 0, 1)',
                                                'width': 2,
                                            },
                                 }

                shape_filled = {
                                    'type': 'rect',
                                    'x0': x0,
                                    'y0': y0,
                                    'x1': x1_fill,
                                    'y1': y0 + step,
                                    'fillcolor': pred_colors[int(len(pred_colors)-1-(max(pred_prob[i]*len(pred_colors)-1, 0)))]
                                 }
                shapes_list.append(shape_unfilled)
                shapes_list.append(shape_filled)

                # Set the starting y coordinate for the next shapes
                y0 = y0 + 2 * step

            # Getting points along the percentage bar plots
            x_range = [list(np.array(range(0, 10, 1))*0.1+inst_scores.shape[1]) for idx in range(len(patients))]

            # Flatten the list
            text_x = [item for sublist in x_range for item in sublist]

            # Y coordinates of the prediction probability text
            text_y = [patient for patient in patients for idx in range(10)]

            # Prediction probabilities in text form, to appear in the plot
            text_content = [pred_prob[idx] for idx in range(len(pred_prob)) for i in range(10)]

            # [TODO] Ajdust the zoom so that the initial plot doens't block part of the first and last sequences that show up

            # Create plotly chart
            plot_data = [{"x": seq_insts_x,
                          "y": patients_y,
                          "marker": dict(color=colors, size=12,
                                         line = dict(
                                                      color = 'black',
                                                      width = 1
                                                    ),
                                         colorbar=dict(title='Scores'),
                                         colorscale=[[0, 'rgba(30,136,229,1)'], [0.5, 'white'], [1, 'rgba(255,13,87,1)']],
                                         cmax=1,
                                         cmin=-1),
                          "mode": "markers",
                          "type": "scatter",
                          "hoverinfo": 'x+y'
                         },
                         go.Scatter(
                                     x=text_x,
                                     y=text_y,
                                     text=text_content,
                                     mode='text',
                                     textfont=dict(size = 1, color='#ffffff'),
                                     hoverinfo='y+text'
                         )]
            layout = go.Layout(
                                title="Patients time series",
                                xaxis=dict(
                                            title="Instance",
                                            showgrid=False,
                                            zeroline=False
                                          ),
                                yaxis=dict(
                                            title="Patient ID",
                                            showgrid=False,
                                            zeroline=False,
                                            type='category'
                                          ),
                                hovermode="closest",
                                shapes=shapes_list,
                                showlegend=False
            )

            if len(patients) > 10:
                # Prevent cramming too many sequences into the plot
                layout.yaxis.range = [patients[-10], patients[-1]]

        # Show the plot
        fig = go.Figure(plot_data, layout)
        py.iplot(fig)

        if get_fig_obj:
            # Only return the figure object if specified by the user
            return fig
        else:
            return

    # [TODO] Develop function to explain, in text form, why a given input data has a certain output.
    # The results gather with instance and feature importance, as well as counter-examples, should
    # be used.
    # def explain_output(self, data, detailed_explanation=True):
        # if detailed_explanation:
        #     inst_scores, feat_scores = self.interpret_model(test_data=data, instance_importance=True,
        #                                                     feature_importance=True, fast_calc=False)
        #
            # [TODO] Explain the most important instances and most important features on those instances
            # [TODO] Compare with counter-examples, i.e. cases where the classification was different
        # else:
        #     inst_scores, feat_scores = self.interpret_model(test_data=data, instance_importance=True,
        #                                                     feature_importance=True, fast_calc=True)
        #
            # [TODO] Explain the most important instances and most important features on those instances
