# -*- coding: utf-8 -*-
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit


def csv_shuffle_splits_persist(origin_data, task_num, data_name):
    df = pd.read_csv(origin_data, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    ss = ShuffleSplit(n_splits=10, test_size=0.25, random_state=33)

    i=0
    for train_index, test_index in ss.split(X):
        i+=1
        sample_train_df = df.iloc[train_index,:]
        sample_test_df = df.iloc[test_index,:]
        sample_train_df.to_csv('data/school/school_train_re' + str(i) + '.csv', index=False, header=None)
        sample_test_df.to_csv('data/school/school_test_re' + str(i) + '.csv', index=False, header=None)

        test_input = []
        train_input = []
        test_output = []
        train_output = []
        for j in range(1, task_num+1):
            train_input.append(sample_train_df[sample_train_df.iloc[:,-1]==j].iloc[:,:-2].values.astype(np.double))
            test_input.append(sample_test_df[sample_test_df.iloc[:,-1]==j].iloc[:,:-2].values.astype(np.double))
            y_train = sample_train_df[sample_train_df.iloc[:,-1]==j].iloc[:,-2].values.astype(np.double)
            train_output.append(y_train.reshape(y_train.shape[0],1))
            y_test = sample_test_df[sample_test_df.iloc[:,-1]==j].iloc[:,-2].values.astype(np.double)
            test_output.append(y_test.reshape(y_test.shape[0],1))
        test_input = np.array(test_input).reshape(1,task_num)
        train_input = np.array(train_input).reshape(1,task_num)
        test_output = np.array(test_output).reshape(1,task_num)
        train_output = np.array(train_output).reshape(1,task_num)
        scipy.io.savemat('data/'+ data_name +'_mat/'+data_name+'_re{}.mat'.format(i), {'test_input':test_input, 'test_output':test_output, 'train_input':train_input, 'train_output':train_output})

csv_shuffle_splits_persist('./data/school.csv', 139, 'school')

