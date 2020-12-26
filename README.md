# Anomaly Detection in financial risk
This repository contains example solutions as described in the [blog](https://rajat-bhardwaj.github.io/) to identify financial risk in the gambling industry.


Few notes:
> Isolation forest and LOF return a lower value for outliers. (as implemented in scikit-learn).

> RRCF and Autoencoders provide higher values for anomalies. Therefore, the results from iForest and Lof are scaled from 0-1 and inverted to follow the results in the same directions.
