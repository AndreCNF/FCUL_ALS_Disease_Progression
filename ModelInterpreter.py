import torch                            # PyTorch to create and apply deep learning models
import pandas as pd                     # Pandas to handle the data in dataframes
import shap                             # Module used for the calculation of approximate Shapley values
import warnings                         # Print warnings for bad practices

class ModelInterpreter:
    def __init__(self, model, data, seq_len_dict=None, id_column=0, inst_column=1
                 fast_calc=True, SHAP_bkgnd_samples=1000, random_seed=42):
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
        fast_calc : bool, default None
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
        '''
        # Initialize parameters according to user input
        self.model = model
        self.seq_len_dict = seq_len_dict
        self.id_column = id_column
        self.inst_column = inst_column
        self.fast_calc = fast_calc
        self.SHAP_bkgnd_samples = SHAP_bkgnd_samples
        self.random_seed = random_seed

        if type(data) is torch.Tensor:
            self.data = data
        elif type(data) is pd.DataFrame:
            n_patients = data[self.id_column].nunique()       # Total number of patients
            n_inputs = len(data.columns)                          # Number of input features
            padding_value = 999999                                # Value to be used in the padding

            # Pad data (to have fixed sequence length) and convert into a PyTorch tensor
            self.data = utils.dataframe_to_padded_tensor(data, self.seq_len_dict, n_subjects,
                                                         n_inputs, padding_value=padding_value)
        else:
            raise Exception('ERROR: Invalid data type. Please provide data in a Pandas DataFrame or PyTorch Tensor format.')

        # Declare attributes that will store importance scores (instance and feature importance)
        self.inst_scores = None
        self.feat_scores = None

    def dataframe_to_padded_tensor(df, seq_len_dict, n_ids, n_inputs, id_column='subject_id',
                                   data_type='PyTorch', padding_value=999999):
        '''Converts a Pandas dataframe into a padded NumPy array or PyTorch Tensor.

        Parameters
        ----------
        df : pandas.Dataframe
            Data in a Pandas dataframe format which will be padded and converted
            to the requested data type.
        seq_len_dict : dict
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
        seq_len_df = df.groupby(df.columns[self.id_column]).df.columns[self.inst_column].count() \
                                                                                        .to_frame() \
                                                                                        .sort_values(by=df.columns[self.inst_column], ascending=False)
        seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.values))])
        return seq_len_dict

    def sort_by_seq_len(self, data, seq_len_dict=None):
        '''Sort the data by sequence length in order to correctly apply it to a
        PyTorch neural network.

        Parameters
        ----------
        data : torch.Tensor, default None
            Data tensor on which sorting by sequence length will be applied.

        Returns
        -------
        sorted_data : torch.Tensor, default None
            Data tensor already sorted by sequence length.
        x_lengths : list of int
            Sorted list of sequence lengths, relative to the input data.
        '''
        if seq_len_dict is None:
            # Use the same sequence length dictionary as the one from the original dataset
            seq_len_dict = self.seq_len_dict

        # Get the original lengths of the sequences, for the input data
        x_lengths = [self.seq_len_dict[id] for id in list(data[:, 0, self.id_column].numpy())]

        # Sorted indeces to get the data sorted by sequence length
        data_sorted_idx = list(np.argsort(x_lengths)[::-1])

        # Sort the x_lengths array by descending sequence length
        x_lengths = [x_lengths[idx] for idx in data_sorted_idx]

        # Sort the data by descending sequence length
        sorted_data = data[data_sorted_idx, :, :]
        return sorted_data, x_lengths

    def instance_importance(self, data=None, x_lengths=None):
        '''Calculate the instance importance scores to interpret the impact of
        each instance of a sequence on the final output.

        Parameters
        ----------
        data : torch.Tensor, default None
            Optionally, the user can specify a subset of data on which model
            interpretation will be made (i.e. calculating feature and/or
            instance importance). Otherwise, all the data is used.
        x_lengths : list of int
            Sorted list of sequence lengths, relative to the input data.

        Returns
        -------
        inst_scores : torch.Tensor
            Tensor containing the importance scores of each instance in the
            given input sequences. Only calculated if instance_importance is set
            to True.
        '''
        if data is None:
            # If a subset of data to interpret isn't specified, the interpreter will use all the data
            data = self.data

        # Model output when using all the original instances in the input sequences
        ref_output = self.model(data)

        def calc_instance_score(sequence_data, instance, ref_output):
            # Indeces without the instance that is being analyzed
            new_idx = [i for i in range(0, instance)] + [i for i in range(instance, sequence_data.shape[0])]

            # Sequence data without the instance that is being analyzed
            sequence_data = sequence_data[new_idx, :]

            # Add a third dimension for the data to be readable by the model
            sequence_data = sequence_data.unsqueeze(0)

            # Calculate the output without the instance that is being analyzed
            new_output = self.model(sequence_data)

            # The instance importance score is then the difference between the output probability without the instance
            # and the probability with the instance
            inst_score = new_output - ref_output
            return inst_score

        inst_scores = [[calc_instance_score(data[seq_id, :, :], inst, ref_output[seq_id]) for inst in x_lengths[seq_id]]
                       for seq_id in range(data.shape[0])]
        inst_scores = torch.Tensor(inst_scores)
        return inst_scores

    def feature_importance(self, bkgnd_data=None, test_data=None, x_lengths=None, fast_calc=None):
        '''Calculate the feature importance scores to interpret the impact
        of each feature in each instance's output.

        Parameters
        ----------
        bkgnd_data : torch.Tensor, default None
            In case of setting fast_calc to True, which makes the algorithm use
            SHAP in the feature importance, the background data used in the
            explainer can be set through this parameter.
        test_data : torch.Tensor, default None
            Optionally, the user can specify a subset of data on which model
            interpretation will be made (i.e. calculating feature and/or
            instance importance). Otherwise, all the data is used.
        x_lengths : list of int
            Sorted list of sequence lengths, relative to the input data.
        fast_calc : bool, default None
            If set to True, the algorithm uses simple mask filters, occluding
            instances and replacing features with reference values, in order
            to do a fast interpretation of the model. If set to False, SHAP
            values are used for a more precise and truthful interpretation of
            the model's behavior, requiring longer computation times.

        Returns
        -------
        feat_scores : torch.Tensor
            Tensor containing the importance scores of each feature, of each
            instance, in the given input sequences. Only calculated if
            feature_importance is set to True.
        '''
        if fast_calc is None:
            # Use the predefined option if fast_calc isn't set in the function call
            fast_calc = self.fast_calc

        if not fast_calc:
            print(f'Attention: you have chosen to interpret the model using SHAP, \
                    with {self.SHAP_bkgnd_samples} background samples applied to \
                    {test_data.shape[0]*test_data.shape[1]} test samples. \
                    This might take a while. Depending on your computer\'s \
                    processing power, you should do a coffee break or even go \
                    to sleep!')

            # Sort the background data by sequence length
            bkgnd_data, x_lengths_bkgnd = self.sort_by_seq_len(bkgnd_data)

            # Remove identifier columns from the data
            bkgnd_data.drop(columns=[bkgnd_data.columns[self.id_column],
                                     bkgnd_data.columns[self.inst_column]], inplace=True)
            test_data.drop(columns=[test_data.columns[self.id_column],
                                    test_data.columns[self.inst_column]], inplace=True)

            # Make sure that the data is in type float
            bkgnd_data = bkgnd_data.float()
            test_data = test_data.float()

            # Use the background dataset to integrate over
            explainer = shap.DeepExplainer(model, bkgnd_data, feedforward_args=[x_lengths_bkgnd])

            # Count the time that takes to calculate the SHAP values
            start_time = time.time()

            # Explain the predictions of the sequences in the test set
            feat_scores = explainer.shap_values(test_data_exp,
                                                feedforward_args=[x_lengths_bkgnd, x_lengths_test],
                                                var_seq_len=True)
            print(f'Calculation of SHAP values took {time.time() - start_time} seconds')
            return feat_scores

        # else:
            # [TODO] Apply mask filter
            # return feat_scores

        return

    # [Bonus TODO] Upload model explainer and interpretability plots to Comet.ml
    def interpret_model(self, bkgnd_data=None, test_data=None, new_data=False,
                        df=None, instance_importance=True, feature_importance=False,
                        fast_calc=None):
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

        Returns
        -------
        inst_scores : torch.Tensor
            Tensor containing the importance scores of each instance in the
            given input sequences. Only calculated if instance_importance is set
            to True.
        feat_scores : torch.Tensor
            Tensor containing the importance scores of each feature, of each
            instance, in the given input sequences. Only calculated if
            feature_importance is set to True.
        '''
        if fast_calc is None:
            # Use the predefined option if fast_calc isn't set in the function call
            fast_calc = self.fast_calc

        if test_data is None:
            if fast_calc:
                # If a subset of data to interpret isn't specified, the interpreter will use all the data
                test_data = self.data
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
            test_data, x_lengths_test = self.sort_by_seq_len(test_data, seq_len_dict)
        else:
            # Sort the data by sequence length
            test_data, x_lengths_test = self.sort_by_seq_len(test_data)

        if not fast_calc and bkgnd_data is None:
            # Get the background set from the dataset
            bkgnd_data, _ = self.create_bkgnd_test_sets()

        if instance_importance:
            # Calculate the scores of importance of each instance
            self.inst_scores = self.instance_importance(test_data, x_lengths_test)

        if feature_importance:
            # Calculate the scores of importance of each feature in each instance
            self.feat_scores = self.feature_importance(bkgnd_data, test_data, x_lengths_test, fast_calc)

        if instance_importance and feature_importance:
            return inst_scores, feat_scores
        elif instance_importance and not feature_importance:
            return inst_scores
        elif not instance_importance and feature_importance:
            return feature_importance
        else:
            warnings.warn('Without setting instance_importance nor feature_importance \
                           to True, the interpret_model function won\'t do anything relevant.')
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
