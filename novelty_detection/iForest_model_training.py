import multiprocessing
import gc
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterSampler
from aml_unsupervised_data_preprocess import ModelTrainingData
from sklearn.preprocessing import MinMaxScaler
from tf_logger_customised import CustomLogger


class IforestAlgorithm(ModelTrainingData):

    def __init__(self):

        super().__init__()
        self.n_hyperparam = self.model_parameters.get('iForest').get(
            'n_hyperparam')
        self.training_data, self.validation_data = self.get_dataset()

    def setup_hyper_param_grid(self):
        """
        This function randomly selects `n` hyper paramters.

        Return:
        List of dict
        """
        # specify parameters and distributions to sample from
        param_dist = {
            "n_estimators": np.linspace(100, 2000, num=10).astype('int64'),
            "max_samples": np.linspace(10, 500, num=10).astype('int64'),
            "max_features": np.linspace(0.1, 1.0, num=10)
        }

        param_list = list(
            ParameterSampler(param_dist,
                             n_iter=self.n_hyperparam,
                             random_state=self.rng))
        param_com_list = [
            dict((k, round(v, 6)) for (k, v) in d.items()) for d in param_list
        ]

        return param_com_list

    def model_train_n_eval(self, training_data, validation_data, combination,
                           model_iForest, n_val_samples, rng):
        """
        This function trains the model on the given input and
        evaluates it performance on F1 score and AUC values.

        Args:
        training_data: Training data pandas DF
        validation_data: Validation data pandas DF
        combination: Dict of hyperparam
        model_iForest: iForest model reference
        n_val_samples: number of samples of validation dataset for evaluation
        rng: numpy random number reference

        Return:
        Dict with hyperparam and evaluation results

        """
        X_train = training_data.sample(frac=self.fraction,
                                       replace=True,
                                       random_state=rng)

        # generate validation dataset preserving the contamination value
        sample_validation = self.stratified_val_samples(validation_data)

        # set model parameters
        model_iForest = model_iForest.set_params(
            random_state=rng,
            n_estimators=combination.get("n_estimators"),
            max_samples=combination.get("max_samples"),  # n_samples,
            max_features=combination.get("max_features"))

        # train the model
        model_iForest.fit(X_train)

        # find true labels
        y_true = [sample.label for sample in sample_validation]

        # prepare data for prediction
        X_val = [sample.drop(columns=['label']) for sample in sample_validation]

        # The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
        # The measure of normality of an observation given a tree is the depth of the leaf containing this observation,
        # which is equivalent to the number of splittings required to isolate this point
        # Negative scores represent outliers, positive scores represent inliers. >> decision function
        # score_samples = The anomaly score of the input samples. The lower, the more abnormal.

        pred_score = [
            model_iForest.score_samples(val_dataset) for val_dataset in X_val
        ]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_pred_score = [
            scaler.fit_transform(predictions.reshape(-1, 1))
            for predictions in pred_score
        ]
        inverted_anomaly_score = [
            1 - predictions for predictions in scaled_pred_score
        ]

        # performance measure using AUC for fpr and tpr
        average_auc, sd_auc = self.compute_pr_auc(y_true,
                                                  inverted_anomaly_score)

        combination.update({"avg_auc": average_auc, "std_dev_auc": sd_auc})

        gc.collect()

        return combination

    def mp_evaluation_hyperpram(self, train, validate):
        """
        This function executes the multi process evaluation of hyper-parameters.
        For the given list of hyperparam combinations, this function runs a
        batch equal to the number specified processes.

        Args:
        train: Training datset
        validate: validation dataset

        Return:
        A pandas DF

        """

        max_number_processes = self.n_processes
        pool_2 = multiprocessing.Pool(max_number_processes)
        ctx = multiprocessing.get_context()

        logging.info('Custom logs iForest: Get hyper-parameters')
        param_comb_list = self.setup_hyper_param_grid()

        # create validation dataset as per the contamination value
        val_sample = self.val_contamination_sample(validate)

        # isolation forest implementation
        model_iForest = IsolationForest()

        output_df = []
        logging.info('Custom logs iForest: Execute multi-process HP tuning ')

        batch_result = [
            pool_2.apply_async(self.model_train_n_eval,
                               args=(train, val_sample, combination,
                                     model_iForest, self.n_val_samples,
                                     self.rng))
            for combination in param_comb_list
        ]
        try:
            output = [p.get() for p in batch_result]
        except multiprocessing.TimeoutError:
            logging.error(
                'Custom logs iForest: Process not responding for evaluation')
        else:
            for results in output:
                output_df.append(pd.DataFrame(results, index=[0]))

            test_df = pd.concat(output_df)

        return test_df

    def execute_model_iForest(self):

        ts = datetime.now()
        salt = ts.strftime("%Y_%m_%d_%H_%M_%S")
        filename = 'iForest_model_taining_{}.log'.format(salt)

        log_ref = CustomLogger(filename)
        log_ref.setLogconfig()

        logging.info('Custom logs iForest: Number of hyper parameters = %d',
                     self.n_hyperparam)
        logging.info(
            'Custom logs iForest: Fraction of training dataset for hp tuning = %.5f',
            self.fraction)
        logging.info('Custom logs iForest: Number of validation samples = %d',
                     self.n_val_samples)
        logging.info(
            'Custom logs iForest: contamination for validation set = %.5f',
            self.contamination)
        logging.info('Custom logs iForest: Initiate model tuning process')

        model_results = self.mp_evaluation_hyperpram(self.training_data,
                                                     self.validation_data)

        return model_results
