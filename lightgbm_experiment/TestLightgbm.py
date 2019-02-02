# coding: utf-8
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit


def get_data_for_independent_lightgbm(data, task_num):
    df = pd.read_csv(data, header=None)
    list = []
    for i in range(1, task_num+1):
       list.append(df[df.iloc[:,-1]==i].iloc[:,:-1])
    return list


def repetition(df, n_splits, test_size, eval_metric):
    # Shuffle split
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=33)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print(X.shape, y.shape)
    score = []
    for train_index, test_index in ss.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gbm = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=100, min_child_samples=1, random_state=33,n_jobs=-1)
        gbm.fit(X_train,y_train)
        y_pred = gbm.predict(X_test)
        if eval_metric == 'rmse':
            score.append((mean_squared_error(y_test, y_pred))**0.5)
        if eval_metric == 'nrmse':
            score.append(((mean_squared_error(y_test, y_pred))**0.5)/(np.max(y_test)-np.min(y_test)))
    return np.mean(score)


# 预测还要每个任务单独预测评估
def repetition_agg(list, n_splits, test_size, eval_metric):
    # Shuffle split
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=33)

    y_test_set = []
    y_pred_set = []

    for i in range(n_splits):
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        y_pred_list = []

        for df in list:
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            index_list = [(train_index, test_index) for train_index, test_index in ss.split(X)]
            train_index, test_index = index_list[i]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_test_list.append(X_test)
            y_test_list.append(y_test)
        train_sample_cnt = 0
        for t in X_train_list:
            train_sample_cnt += len(t)
        X_train_agg = np.zeros((train_sample_cnt, X_train_list[0].shape[1]))
        y_train_agg = np.zeros(train_sample_cnt)
        cnt = 0
        for t in X_train_list:
            X_train_agg[cnt:cnt+t.shape[0]] = t
            cnt += t.shape[0]
        cnt = 0
        for t in y_train_list:
            y_train_agg[cnt:cnt+len(t)] = t
            cnt += t.shape[0]

        gbm = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=100, min_child_samples=1, random_state=33, n_jobs=-1)
        gbm.fit(X_train_agg, y_train_agg)
        y_test_set.append(y_test_list)
        for test in X_test_list:
            y_pred = gbm.predict(test)
            y_pred_list.append(y_pred)
        y_pred_set.append(y_pred_list)

    task_num = len(y_pred_set[0])
    agg_score = []
    for i in range(task_num):
        score = []
        for j in range(n_splits):
            if eval_metric == 'rmse':
                score.append((mean_squared_error(y_test_set[j][i], y_pred_set[j][i]))**0.5)
            elif eval_metric == 'nrmse':
                score.append(((mean_squared_error(y_test_set[j][i], y_pred_set[j][i]))**0.5)/(np.max(y_test_set[j][i])-np.min(y_test_set[j][i])))
        agg_score.append(np.mean(score))
    return np.mean(agg_score)


def indepenent_experiment(data, task_num, eval_metric):
    list = get_data_for_independent_lightgbm(data, task_num)
    score = []
    for e in list:
        s = repetition(e, 10, 0.25, eval_metric)
        score.append(s)
    return np.mean(score)


def aggregate_experiment(data, task_num, eval_metric):
    list = get_data_for_independent_lightgbm(data, task_num)
    return repetition_agg(list, 10, 0.25, eval_metric)



score = indepenent_experiment('./data/school.csv', 139, 'rmse')
print("The mean RMSE score of independent lightgbm: {}".format(score))

score = aggregate_experiment('./data/school.csv', 139, 'rmse')
print("The mean RMSE score of aggregate lightgbm: {}".format(score))


