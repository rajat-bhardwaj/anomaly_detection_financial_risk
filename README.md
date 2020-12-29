# Anomaly Detection in financial risk
This repository contains example solutions as described in the [blog](https://rajat-bhardwaj.github.io/) to identify financial risk in the gambling industry.

## Condition monitoring
This model checks the current activity on an account and compares it with historic data of the same account. The model is described in this [post](https://rajat-bhardwaj.github.io/2020/01/04/aml-anomaly-detection.html)


## Novelty detection
It contains python implementation of the following four models. It uses python multiprocessing module to evaluate hyper-parameters on a single machine.

1. Isolation Forest
2. Robust Random Cut Forest
3. LOF (local outlier factor)
4. Autoencoders implemented via. Tensorflow

The implementation uses 8 different configurations available in `auen_model_configuration.json` file. Each of these models/configurations can use further `n` hyper-parameter evaluations. This can be configured in `training_config.json` file under `autoencoders`.


Few notes:
> Isolation forest and LOF return a lower value for outliers. (as implemented in scikit-learn). The RRCF and Autoencoders provide higher values for anomalies. Therefore, the results from iForest and LOF are first scaled from 0-1 and then inverted to follow the results in the same directions.

> If you find any issues with the code or have any feedback, please create a new issue in this repository.
