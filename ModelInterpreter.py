import torch                            # PyTorch to create and apply deep learning models
import pandas as pd                     # Pandas to handle the data in dataframes
import shap                             # Module used for the calculation of approximate Shapley values

class ModelInterpreter:
    def __init__(self, model, data, seq_len_dict=None, id_column='subject_id',
                 fast_calc=True, SHAP_bkgnd_samples=1000):
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
            the fixed sequence length tensor. If the user specified a subset of
            data in test_data, this dictionary should contain the information
            corresponding only to that subset.
        id_column : string, default 'subject_id'
            Name of the column which corresponds to the subject identifier in the
            dataframe.
        fast_calc : bool, default None
            If set to True, the algorithm uses simple mask filters, occluding
            instances and replacing features with reference values, in order
            to do a fast interpretation of the model. If set to False, SHAP
            values are used for a more precise and truthful interpretation of
            the model's behavior, requiring longer computation times.
        SHAP_bkgnd_samples : int, default 1000

        '''
        # Initialize parameters according to user input
        self.model = model
        self.seq_len_dict = seq_len_dict
        self.fast_calc = fast_calc
        self.SHAP_bkgnd_samples = SHAP_bkgnd_samples

        if type(data) is torch.Tensor:
            self.data = data
        elif type(data) is pd.DataFrame:
            n_patients = data[id_column].nunique()       # Total number of patients
            n_inputs = len(data.columns)                 # Number of input features
            padding_value = 999999                       # Value to be used in the padding

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

    def instance_importance(self, data=None, seq_len_dict=None):
        '''Calculate the instance importance scores to interpret the impact of
        each instance of a sequence on the final output.

        Parameters
        ----------
        data : torch.Tensor, default None
            Optionally, the user can specify a subset of data on which model
            interpretation will be made (i.e. calculating feature and/or
            instance importance). Otherwise, all the data is used.
        seq_len_dict : dict, default None
            Dictionary containing the sequence lengths for each index of the
            original dataframe. This allows to ignore the padding done in
            the fixed sequence length tensor. If the user specified a subset of
            data in test_data, this dictionary should contain the information
            corresponding only to that subset.

        Returns
        -------
        inst_scores : torch.Tensor
            Tensor containing the importance scores of each instance in the
            given input sequences. Only calculated if instance_importance is set
            to True.
        '''
        return

    def feature_importance(self, bkgnd_data=None, test_data=None, seq_len_dict=None, fast_calc=None):
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
        seq_len_dict : dict, default None
            Dictionary containing the sequence lengths for each index of the
            original dataframe. This allows to ignore the padding done in
            the fixed sequence length tensor. If the user specified a subset of
            data in test_data, this dictionary should contain the information
            corresponding only to that subset.
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
            fast_calc = self.fast_calc

        if fast_calc is False:
            print(f'Attention: you have chosen to interpret the model using SHAP, \
                    with {self.SHAP_bkgnd_samples} background samples applied to \
                    {test_data.shape[0]*test_data.shape[1]} test samples. \
                    This might take a while. Depending on your computer\'s \
                    processing power, you should do a coffee break or even go \
                    to sleep!')

        return

    # [TODO] Create a model interpretation method that does feature importance on
    # each instance and also "instance importance", finding which timestamps had the
    # biggest impact on the model's output for a particular input data.
    # [Bonus TODO] Upload model explainer and interpretability plots to Comet.ml
    def interpret_model(self, bkgnd_data=None, test_data=None, seq_len_dict=None,
                        instance_importance=True, feature_importance=False, fast_calc=None):
        '''Method to calculate scores of feature and/or instance importance, in
        order to be able to interpret a model on a given data.

        Parameters
        ----------
        bkgnd_data : torch.Tensor or list of integers, default None
            In case of setting fast_calc to True, which makes the algorithm use
            SHAP in the feature importance, the background data used in the
            explainer can be set through this parameter, either directly in a
            PyTorch tensor or simply as a list of indeces, which will be applied
            when fetching all the data given in the ModelInterpreter
            initialization.
        test_data : torch.Tensor or list of integers, default None
            Optionally, the user can specify a subset of data on which model
            interpretation will be made (i.e. calculating feature and/or
            instance importance), either directly in a PyTorch tensor or simply
            as a list of indeces, which will be applied when fetching all the
            data given in the ModelInterpreter initialization.
            Otherwise, all the data is used.
        seq_len_dict : dict, default None
            Dictionary containing the sequence lengths for each index of the
            original dataframe. This allows to ignore the padding done in
            the fixed sequence length tensor. If the user specified a subset of
            data in test_data, this dictionary should contain the information
            corresponding only to that subset.
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
        if test_data is None:
            # If a subset of data to interpret isn't specified, the interpreter will use all the data
            test_data = self.data

    # [TODO] Develop function to explain, in text form, why a given input data has a certain output.
    # The results gather with instance and feature importance, as well as counter-examples, should
    # be used.
    # def explain_output(self, data):
