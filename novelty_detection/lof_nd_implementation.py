import gc
import logging
import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import MinMaxScaler
from tf_logger_customised import CustomLogger
from aml_unsupervised_data_preprocess import ModelTrainingData


class lofNoveltyDetection(ModelTrainingData):

    def __init__(self):
        super().__init__()
        self.n_hyperparam = self.model_parameters.get('iForest').get('n_hyperparam')
        self.training_data, self.validation_data = self.get_dataset()

    def setup_hyper_param_grid(self):
        """
        This function randomly selects `n` hyper paramters.

        Return:
        List of dict
        """

        param_dist = {
            "n_neighbors": np.linspace(5, 200, num=20).astype('int64'),
            "leaf_size": np.linspace(30, 100, num=20).astype('int64')
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
                           model_lof_novelty):
        """
        This function trains the model on the given inputs.

        Args:
        training_data: Training data pandas DF
        validation_data: Validation data pandas DF
        combination: Dict of hyperparam
        model_iForest: iForest model reference

        Return:
        Dict with hyperparam and evaluation results

        """
        X_train = training_data.sample(frac=self.fraction,
                                       replace=True,
                                       random_state=self.rng)

        # generate validation dataset preserving the contamination value
        sample_validation = self.stratified_val_samples(validation_data)

        y_true = [sample.label for sample in sample_validation]

        # set model parameters
        model_lof_novelty = model_lof_novelty.set_params(
            n_neighbors=combination.get('n_neighbors'),
            leaf_size=combination.get('leaf_size'))

        # train the model
        model_lof_novelty.fit(X_train)

        X_val = [sample.drop(columns=['label']) for sample in sample_validation]

        # It is the opposite as bigger is better, i.e. large values correspond to inliers.
        pred_score = [
            model_lof_novelty.score_samples(val_dataset)
            for val_dataset in X_val
        ]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_pred_score = [
            scaler.fit_transform(predictions.reshape(-1, 1))
            for predictions in pred_score
        ]
        inverted_anomaly_score = [
            1 - predictions for predictions in scaled_pred_score
        ]

        average_auc, sd_auc = self.compute_pr_auc(
            y_true, inverted_anomaly_score)

        combination.update({"avg_auc": average_auc, "std_dev_auc": sd_auc})
        gc.collect()

        return combination

    def mp_evaluation_hyperpram(self, train, validate):
        """
        This function executes the multi process evaluation of hyper parameters.
        For the given list of hyperparam combinations, this function runs a
        batch equal to the number of available CPU.

        Args:
        train: Training datset
        validate: validation dataset

        Return:
        A pandas DF

        """

        max_number_processes = self.n_processes
        pool_2 = multiprocessing.Pool(max_number_processes)
        ctx = multiprocessing.get_context()

        logging.info('Custom logs: Generating hyperparameter list')
        param_comb_list = self.setup_hyper_param_grid()
        model_lof_novelty = LocalOutlierFactor(novelty=True)

        # create validation dataset as per the contamination value
        val_sample = self.val_contamination_sample(validate)

        output_df = []
        logging.info('Custom logs LOF Novelty: Execute multi-process HP tuning')

        batch_result = [
            pool_2.apply_async(self.model_train_n_eval,
                               args=(train, val_sample, combination,
                                     model_lof_novelty))
            for combination in param_comb_list
        ]
        try:
            output = [p.get() for p in batch_result]
        except multiprocessing.TimeoutError:
            logging.error(
                'Custom logs LOF Novelty: Process not responding for evaluation'
            )
        else:
            for results in output:
                output_df.append(pd.DataFrame(results, index=[0]))

            test_df = pd.concat(output_df)

        return test_df
    
    def execute_model_lof_nov_detection(self):

        ts = datetime.now()
        salt = ts.strftime("%Y_%m_%d_%H_%M_%S")
        filename = 'lof_nd_model_taining_{}.log'.format(salt)

        log_ref = CustomLogger(filename)
        log_ref.setLogconfig()

        logging.info('Custom logs LOF novelty: Setup model parameters')

        logging.info('Custom logs LOF novelty: Number of hyper parameters = %d',
                     self.n_hyperparam)
        logging.info(
            'Custom logs LOF novelty: Fraction of training dataset for hp tuning = %.5f',
            self.fraction)
        logging.info('Custom logs LOF novelty: Number of validation samples = %d',
                     self.n_val_samples)
        logging.info(
            'Custom logs LOF novelty: contamination for validation set = %.5f',
            self.contamination)
        logging.info('Custom logs LOF novelty: Initiate model tuning process')

        model_results = self.mp_evaluation_hyperpram(self.training_data, self.validation_data)
        
        return model_results
