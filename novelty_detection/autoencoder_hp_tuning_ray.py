import os
import json
import logging
import gc
from datetime import datetime
import ray
from callbacks import CustomMetrics
from sklearn import metrics
from sklearn.model_selection import ParameterSampler
import pandas as pd
import numpy as np
from tf_logger_customised import CustomLogger
from aml_unsupervised_data_preprocess import ModelTrainingData
import warnings
warnings.filterwarnings('ignore')


@ray.remote
class TestClassRayImpln:

    def __init__(self, ref_data, path_csv):

        self.ref_data = ref_data
        self.path_csv = path_csv
        self.n_hyperparams = ref_data.model_parameters.get('autoencoders').get(
            'n_hyperparam')
        self.fraction = ref_data.model_parameters.get('training').get(
            'train_frac_hp_tuning')
        self.learning_rate = ref_data.model_parameters.get('autoencoders').get(
            'learning_rate')
        self.logs_base_dir = ref_data.model_parameters.get('autoencoders').get(
            'logs_base_dir')
        self.n_val_samples = ref_data.model_parameters.get('training').get(
            'n_val_samples')

        self.path = os.path.dirname(os.path.realpath('__file__'))
        self.model_config = json.loads(
            open(os.path.join(self.path,
                              'auen_model_configuration.json')).read())
        self.path_trainset = ref_data.model_parameters.get('training').get(
            'path_trainset')
        self.path_devset = ref_data.model_parameters.get('training').get(
            'path_devset')
        self.train_filepath = os.path.join(self.path_csv, self.path_trainset)
        self.validate_filepath = os.path.join(self.path_csv, self.path_devset)
        self.rng = ref_data.rng
        self.verbose_flag = False

    def sample_train(self):
        """
        reads training data from the disk and generate a training sample

        returns:
        pandas DF - training data
        """

        train = pd.read_csv(self.train_filepath,
                            header=None,
                            index_col=False,
                            dtype=str)

        train = train.fillna(0)
        train = train.astype(np.float32)

        X_train = train.sample(frac=self.fraction,
                               replace=True,
                               random_state=self.rng)
        return X_train

    def validation(self):
        """
        reads validation data from the disk and generate a training sample

        returns:
        pandas DF - validation data
        """

        validation_data = pd.read_csv(self.validate_filepath,
                                      header=None,
                                      index_col=False,
                                      dtype=np.float32)
        validation_data.rename(columns={264: 'label'}, inplace=True)
        validation_data['label'] = validation_data.label.astype(np.int16)

        val_sample = self.ref_data.val_contamination_sample(validation_data)

        return val_sample

    def hyper_parameters(self):
        """
        Generates n randomly selected hyperparameter combination

        returns:
        list of dict
        """
        logging.info('Custom logs AEN: generate hyperparameters')
        param_dist = {
            'epochs':
                np.linspace(20, 100, num=10).astype('int64'),
            'batch_size': [64, 128, 256, 512],
            # l2 regularization
            'activation_reg':
                np.linspace(0.01, 0.0001, num=10).astype(np.float32),
            'leakyalpha':
                np.linspace(0.01, 0.3, num=10).astype(np.float64),
        }

        param_list = list(
            ParameterSampler(param_dist,
                             n_iter=self.n_hyperparams,
                             random_state=self.rng))
        param_com_list = [
            dict((k, round(v, 6)) for (k, v) in d.items()) for d in param_list
        ]

        return param_com_list

    def config_n_train(self, model_name, layers, columns, learning_rate, logdir,
                       epochs, input_data, validation_sample, verbose_flag,
                       batch_size, activation_reg, leakyalpha):
        """
        Configures the model based on the layers information.
        Trains and store results in Tensorboad for each hyperparameter value.

        Args:
        model_name: Name of the NN configuration
        layers: List of dict with layer configuration
        columns: Number of columns in the traiing dataset
        learning_rate: Laearning rate for the model
        logdir: Directory name for the Tensorboard
        epochs: Number fo epochs for training
        input_data: Pandas DF for training data
        validation_sample: Pandas DF for validation data
        verbose_flag: Verbose flag for model training
        batch_size: Batch size for model training
        activation_reg: regularization on the activation function
        leakyalpha: leaky value for the leaky relu activation function
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LeakyReLU
        from tensorflow.keras.activations import relu
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import TensorBoard
        from tensorflow.keras.regularizers import l2

        logging.info('Custom logs AEN: configure model')

        model = Sequential()
        for i, options in enumerate(layers):

            if 'activity_regularizer' in options.keys():
                options.update({'activity_regularizer': l2(activation_reg)})

            if i == len(layers) - 1:
                model.add(Dense(units=columns, **options))
            else:
                model.add(Dense(**options))
                if 'leaky' in model_name:
                    model.add(LeakyReLU(leakyalpha))

        # set custom callbacks to implement AUC evaluation
        logging.info('Custom logs AEN: configure custom callbacks')
        # profile_batch must be a non-negative integer or a comma separated string of
        # pair of positive integers. A pair of positive integers signify a range of batches to profile.
        custom_callback = [
            TensorBoard(logdir, profile_batch='100, 150', write_images=True),
            CustomMetrics(validation_sample, os.path.join(logdir, 'metric'),
                          self.ref_data)
        ]

        model.compile(optimizer=Adam(lr=learning_rate),
                      loss='mean_squared_error')

        # Model training
        logging.info('Custom logs AEN: run model training - %s', model_name)
        model.fit(x=input_data.values,
                  y=input_data.values,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=verbose_flag,
                  callbacks=custom_callback)

        logging.info('Custom logs AEN: %s summary', model_name)
        model.summary(print_fn=logging.info)

    def configure_tensorboard_path(self, model_name, hyperparameters):
        """
        This function configures the a separate folder for each model execution.

        Args:
        model_name: name of the model
        hyperparameters: hyper parameters dict
        """
        logging.info('Custom logs AEN: configure tensorboard path')

        run_name = 'e_{}_b_{}_a_{}_la{}_'.format(
            hyperparameters.get('epochs'), hyperparameters.get('batch_size'),
            str(hyperparameters.get('activation_reg'))[0:5],
            str(hyperparameters.get('leakyalpha'))[0:5])

        logdir = os.path.join(self.logs_base_dir, model_name)
        logdir = os.path.join(
            logdir, (run_name + datetime.now().strftime("%Y%m%d-%H%M%S")))
        os.makedirs(logdir, exist_ok=True)

        return logdir

    def single_hp_tuning(self, model_name, hp):
        """
        Execute model training for the given hyperparameter

        Args:
        training_data: Pandas DF
        validate_trnsfrm: Pandas DF
        model_name: Name of the NN configuration
        layers: List of dict with layer configuration
        hp: Dict of hyperprameter configuration

        """
        # sample training data
        logging.info('Custom logs AEN: for -- %s', hp)
        logging.info('Custom logs AEN: generate trianing sample')
        X_train = self.sample_train()

        # get parameters
        columns = X_train.shape[1]
        logdir = self.configure_tensorboard_path(model_name, hp)
        epochs = hp.get('epochs')
        batch_size = hp.get('batch_size')
        activation_reg = float(hp.get('activation_reg'))
        leakyalpha = hp.get('leakyalpha')
        layers = self.model_config.get(model_name)
        validate_trnsfrm = self.validation()

        # build and train model
        model = self.config_n_train(model_name, layers, columns,
                                    self.learning_rate, logdir, epochs, X_train,
                                    validate_trnsfrm, self.verbose_flag,
                                    batch_size, activation_reg, leakyalpha)
        gc.collect()

    def multi_hp_tuning(self, model_name):

        self.setup_logfile(model_name)

        hyperparameters = self.hyper_parameters()
        [self.single_hp_tuning(model_name, hp) for hp in hyperparameters]

    def setup_logfile(self, model_name):
        """
        """
        from tf_logger_customised import CustomLogger

        salt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = 'ray_aen_{}_{}.log'.format(model_name, salt)

        log_ref = CustomLogger(filename)
        log_ref.setLogconfig()


def execute_model_autoencoders():

    ref_data = ModelTrainingData()

    path = os.path.dirname(os.path.realpath('__file__'))
    model_config = json.loads(
        open(os.path.join(path, 'auen_model_configuration.json')).read())

    evaluation_results = []

    actors = [
        TestClassRayImpln.remote(ref_data, path)
        for _ in range(len(model_config))
    ]
    model_names = list(model_config.keys())

    temp_results = []
    for i in range(len(model_names)):

        temp_results.append(actors[i].multi_hp_tuning.remote(model_names[i]))

    useless_variable = ray.get(temp_results)


def init_ray():
    ray.init(num_cpus=8,
             ignore_reinit_error=True,
             redis_port=8265,
             log_to_driver=True)


def main():
    if ray.is_initialized():
        ray.shutdown()
        init_ray()
    else:
        init_ray()

    execute_model_autoencoders()


if __name__ == '__main__':
    main()
