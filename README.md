# multi-task-gradient-boosting

# Parameter Guide
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
+ early_stopping_round: Just like xgboost (0 indicate does not use it, default value is 10)
+ THREAD_NUM: The number of parallel threads (modify at utils.h, default value is 4)
