import time
import multiprocessing
import gc
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from rrcf import rrcf
from sklearn import metrics
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import MinMaxScaler
from tf_logger_customised import CustomLogger
from aml_unsupervised_data_preprocess import ModelTrainingData



class rrcfMultiProcessForest(ModelTrainingData):

    def __init__(self):
        
        super().__init__()
        self.n_hyperparam = self.model_parameters.get('iForest').get('n_hyperparam')
        self.training_data, self.validation_data = self.get_dataset()

    def setup_hyper_param_grid(self):
        """
        This function generates a list of hyperparameter dict

        Args:
        n_hyperparam: number of hyperparameter combination to evaluate
        rng: A numpy random class reference

        Return:
        List of dict

        """
        # for rrcf 1/num_samples_per_tree = contamination
        # contamination = proportion of expected outliers in the training data
        param_dist = {
            "n_estimators": np.linspace(100, 1000, num=self.n_hyperparam).astype('int64')
        }

        return param_dist

    def get_train_samples(self, training_data):
        """
        This function generates sample of the training DF for unit tree training

        Args:
        training_data: Pandas DF - Training dataset
        contamination - suggested contamination value

        Return:
        Pandas DF
        """
        num_samples_per_tree = int(1 / self.contamination)
        sample_training = training_data.sample(n=num_samples_per_tree,
                                               replace=True,
                                               random_state=self.rng)

        index = sample_training.index
        sample_training = sample_training.to_numpy()

        return (sample_training, index)

    def rrcf_forest(self, training_data, n_estimators):
        """
        Training function that fits a tree on the given dataset

        Args:
        training_data: pandas DF
        combination: dict of hyperparameter set

        Returns:
        A list of RCTree class object or forest

        """
        samples = [
            self.get_train_samples(training_data)
            for i in range(0, n_estimators)
        ]
        forest = [
            rrcf.RCTree(train, index_labels=train_index, random_state=self.rng)
            for train, train_index in samples
        ]

        return forest

    def codisp_datapoint(self, row, index, tree):
        """
        This function computes the collusive displacement
        value for the new datapoint.

        Args:
        row: row as a numpy array (new data point)
        index: index value (unique identifier)
        tree: leaarned trees/RCTree class object

        Return:
        collusive displacement value
        """

        # add new leaf and check codisp
        new_leaf = tree.insert_point(row, index=index)
        codisplacement = tree.codisp(new_leaf)
        tree.forget_point(index)

        return codisplacement

    def codisp_dataset_tree(self, dataset, tree):
        """
        This function evaluates the set of datapoints on a given learned tree.

        Args:
        dataset:pandas DF
        tree: leaarned trees/RCTree class object

        Return:
        pandas series of computed collusive displacement value
        """
        dim = dataset.shape[1]
        tree_codisp = dataset.apply(lambda row: self.codisp_datapoint(
            row=row.to_numpy().reshape((1, dim)), index=row.name, tree=tree),
                                    axis=1)

        return tree_codisp

    def eval_dataset_perf(self, val, forest):
        """
        This function evaluates the performance of model (forest) 
        on the single validation dataset.

        Args:
        val: validation dataset sample
        forest: model or tained forest

        Return:
        A tuple of F1 score and AUC values
        """

        y_true = val.label
        val.drop(columns=['label'], inplace=True)

        codisp_val = [self.codisp_dataset_tree(val, tree) for tree in forest]
        ouput_val_sample = pd.concat([series for series in codisp_val], axis=1)
        avg_forest_codisp = ouput_val_sample.apply(lambda row: np.mean(row),
                                                   axis=1)

        avg_forest_codisp = avg_forest_codisp.values.reshape(-1, 1)
        
        precision, recall, _ = metrics.precision_recall_curve(y_true=y_true,
                                                  probas_pred=avg_forest_codisp,
                                                  pos_label=-1)
        
        auc_value = metrics.auc(recall, precision)
        
        return auc_value

    def train_eval_unit_hp(self, train, validation, n_estimators):
        """
        This function trains a forest on the given hyper parameters and
        evaluates the model using bootstraping on the validation dataset.

        Args:
        train: Training datset
        validation: Validation dataset
        combination: dict with hyperparameter
        """

        model_init_time = time.time()
        forest = self.rrcf_forest(train, n_estimators)
        model_train_time = time.time()

        eval_init_time = time.time()
        sample_validate_trnsfrm = self.stratified_val_samples(validation)
        sample_validate_trnsfrm = [
            sample.set_index(np.random.rand(sample.shape[0]))
            for sample in sample_validate_trnsfrm
        ]

        # compute metric for each sample
        result_auc_value = [
            self.eval_dataset_perf(val, forest)
            for val in sample_validate_trnsfrm
        ]
        
        # average AUC over all samples
        average_auc = np.mean(result_auc_value)
        sd_auc = np.std(result_auc_value)

        eval_end_time = time.time()

        train_time = model_train_time - model_init_time
        eval_samples = eval_end_time - eval_init_time
        hp_time = time.time() - model_init_time

        temp_results = {
            'n_estimators':n_estimators,
            'average_auc': average_auc,
            'sd_auc': sd_auc,
            'training_time': train_time,
            'evaluation_time': eval_samples,
            'hp_eval_time': hp_time
        }

        return temp_results

    def tune_model(self, train_trnsfrm, validate_trnsfrm):
        """
        This function tunes the model by generating a RF and evaluating
        the model on multiple samples for a given hyperparameter set.

        Args:
        train_trnsfrm: transformed training set
        validate_trnsfrm: transformed validation datset
        """

        param_comb_list = self.setup_hyper_param_grid()

        max_number_processes = multiprocessing.cpu_count()
        pool_2 = multiprocessing.Pool(max_number_processes)

        logging.info('Custom logs rrcf: Initiating parallel hp tuning process')

        batch_result = [
            pool_2.apply_async(self.train_eval_unit_hp,
                               args=(train_trnsfrm.copy(), 
                                     validate_trnsfrm.copy(), 
                                     n_estimators))
            for n_estimators in param_comb_list.get('n_estimators')
        ]
        try:
            output = [p.get() for p in batch_result]
        except multiprocessing.TimeoutError:
            logging.error('Custom log rrcf: Process not responding for evaluation')
        else:
            logging.info('Custom logs rrcf: computing metrics')
            assert len(output) == self.n_hyperparam
            results = pd.DataFrame(output)

        return results
    
    def execute_rrcf(self):
        """
        Runs the model evaluation/ hyper-parameter tuning 
        for Robust random cut forest model
        """
        ts = datetime.now()
        salt = ts.strftime("%Y_%m_%d_%H_%M_%S")
        filename = 'rrcf_model_{}.log'.format(salt)

        log_ref = CustomLogger(filename)
        log_ref.setLogconfig()

        logging.info('Custom logs rrcf: Number of hyper parameters = %d',
                     self.n_hyperparam)
        logging.info(
            'Custom logs rrcf: Fraction of training dataset for hp tuning = %.5f',
            self.fraction)
        logging.info('Custom logs rrcf: Number of validation samples = %d',
                     self.n_val_samples)
        logging.info('Custom logs rrcf: contamination for validation set = %.5f',
                     self.contamination)
        logging.info('Custom logs rrcf: Initiate model tuning process')

        model_results = self.tune_model(self.training_data, 
                                        self.validation_data)

        return model_results