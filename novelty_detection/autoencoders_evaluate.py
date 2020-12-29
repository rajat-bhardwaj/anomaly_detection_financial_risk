import logging
import time
import multiprocessing
import gc
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


def generate_mse(sample_preds, sample_input):
    """
    Generates mean square error for each customer in the validation
    dataset.

    args:
    sample_preds: prediction for the validation sample
    sample_input: input validation sample

    return:
    numpy array of mse values
    """

    # mean square error per customer for the given validation sample dataframe
    mse = np.mean(np.power(sample_input.sub(sample_preds, fill_value=0), 2),
                  axis=1)

    return mse


def eval_custom_CV(ref_data, validate_trnsfrm, model):
    """
    Evaluates the model using customised cross validation technique.

    Args:
    ref_data: reference variable to access functions
    validate_trnsfrm: Validation data with specified contamination value
    model: trained model

    returns
    average AUC value over all samples
    """
    start_time = time.time()

    sample_validation = ref_data.stratified_val_samples(validate_trnsfrm)

    y_true = [sample.label for sample in sample_validation]

    # prepare data for prediction
    X_val = [sample.drop(columns=['label']) for sample in sample_validation]

    pred = [model.predict(val_dataset) for val_dataset in X_val]

    pred_score = [generate_mse(pred[i], X_val[i]) for i in range(len(X_val))]

    # performance measure using AUC for fpr and tpr
    average_auc, sd_auc = ref_data.compute_pr_auc(y_true, pred_score)

    logging.info('Custom logs AEN: compute pr time %.5f',
                 (time.time() - start_time))
    logging.info(
        'Custom logs AEN: Evaluation metric avg AUC = %.5f, sd AUC = %.5f',
        average_auc, sd_auc)

    return (average_auc, sd_auc)
