# multi-task-gradient-boosting

# Usage
Refer run experiment code. If you want to split csv file into n pairs(training data and test data).

# Run Experiment Code
+ MALSAR_experiment
    + rMTFL: run example_rMTFL.m
    + Lasso: run example_Lasso.m
    + Trace: run example_Trace.m
    + Dirty: run example_Dirty.m
    
+ Multi-Task GBDT
    + run main.cpp

# Parameter Guide for Multi-Task GBDT 
+ log_path: The file path for storing log
+ path: The path of dataset
+ eval_metric: evaluation for performance
+ dataset_name: dataset name
+ feature_size: The number of feature
+ task_num: The number of task
+ max_num_round: The iterations of two-staged model
+ common_num_round: The max iterations of common training stage
+ regularization: The regularization of gain score (options: entropy, variance)
+ beta: The coefficient of variance (The value is zero for entropy-based method)
+ early_stopping_round: like xgboost (0 indicate does not use it, default value is 10)

# Experiment Results
Results from the real datasets for RMSE over 10 repetitions. The statistically best model is highlighted in bold.

(Independent-XGBoost: Train T models for T tasks, Aggregate-XGBoost: Train one model for T tasks(regard T tasks as one task))

|Dataset|Measure|Trace|LASSO|rMTFL|Dirty|Independent-XGBoost|Aggregate-XGBoost|Variance-based Multi-Task GBDT|Entropy-based Multi-Task GBDT|
|---|:---:|---:|:---:|---:|:---:|---:|:---:|---:|:---:|
|school|RMSE|11.452 +/- 0.012|11.216 +/- 0.010|10.467 +/- 0.027|10.411 +/- 0.029|10.767 +/- 0.007|11.099 +/- 0.012|**8.993 +/- 0.232**|8.998 +/- 0.239|