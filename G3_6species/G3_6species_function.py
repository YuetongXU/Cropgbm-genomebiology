# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from multiprocessing import Pool
import lightgbm as lgb
from pandas import read_csv


def prepare_data(dir_path, species, cv_times):
    geno_path = dir_path + species + '_geno.csv'
    phe_path = dir_path + species + '_pheno.csv'
    geno = read_csv(geno_path, header=0, index_col=0)
    phe = read_csv(phe_path, header=0, index_col=0)

    cv_path = dir_path + species + '_CVFs.csv'
    cv_data = read_csv(cv_path, header=0, index_col=0).iloc[:, :cv_times]
    cv_data.columns = ['cv' + str(i) for i in range(cv_times)]
    cv_data.replace(5, 0, inplace=True)
    phe_cvid = phe.merge(cv_data, left_index=True, right_index=True)

    return geno, phe_cvid


def prepare_params(grid_search_params_dict):
    param_list = []

    for learning_rate in grid_search_params_dict['learning_rate_list']:
        for max_depth in grid_search_params_dict['max_depth_list']:
            for n_estimators in grid_search_params_dict['n_estimators_list']:
                for min_data_in_leaf in grid_search_params_dict['min_data_in_leaf_list']:
                    for num_leaves in grid_search_params_dict['num_leaves_list']:

                        params_dict = {
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'n_estimators': n_estimators,
                            'min_data_in_leaf': min_data_in_leaf,
                            'num_leaves': num_leaves,

                            'cvfold': grid_search_params_dict['cvfold'],
                            'phe_name': grid_search_params_dict['phe_name'],
                        }

                        param_list.append(params_dict)

    return param_list


def train_predict(geno_data, phe_data, params_dict, cv_time: str, model_savepath=None):

    cvfold = params_dict['cvfold']
    phe_name = params_dict['phe_name']

    train_params = {
        'learning_rate': params_dict['learning_rate'],
        'max_depth': params_dict['max_depth'],
        'min_data_in_leaf': params_dict['min_data_in_leaf'],
        'num_leaves': params_dict['num_leaves'],
        'objective': 'regression',
        'verbosity': -1,
        'num_threads': 10
    }
    n_estimators = params_dict['n_estimators']

    pearson_list = []

    for i in range(cvfold):
        train_phe = phe_data[phe_data[cv_time] != i][phe_name].dropna(axis=0)
        test_phe = phe_data[phe_data[cv_time] == i][phe_name].dropna(axis=0)
        train_geno = geno_data.loc[train_phe.index.values, :]
        test_geno = geno_data.loc[test_phe.index.values, :]

        train_set = lgb.Dataset(train_geno, label=train_phe)
        train_boost = lgb.train(train_params, train_set, n_estimators)
        if model_savepath is not None:
            train_boost.save_model(model_savepath + '_' + cv_time + '_fold' + str(i) + '.lgb_model')

        predict_phe = train_boost.predict(test_geno)
        pearson_i, p_value_i = stats.pearsonr(predict_phe, test_phe)
        pearson_list.append(pearson_i)

    pearson = np.mean(pearson_list)
    print(cv_time, pearson)

    return pearson


def cv(geno_data, phe_data, params_dict: dict, save_path=None):

    cv_times = params_dict['cv_times']
    pool_max = params_dict['pool_max']

    pool_list = []
    pool = Pool(pool_max)
    for cv_time in range(cv_times):
        cv_time = 'cv' + str(cv_time)
        pool_list.append(pool.apply_async(
            train_predict, [geno_data, phe_data, params_dict, cv_time, save_path]))

    pool.close()
    pool.join()

    pearson_list = []
    for pearson in pool_list:
        pearson_list.append(pearson.get())

    return pearson_list


def grid_search(geno_data, phe_data, save_path, cv_time: str,
                grid_search_params_dict: dict):

    params_dict_list = prepare_params(grid_search_params_dict)

    pool = Pool(grid_search_params_dict['pool_max'])

    pearson_list = []
    for params_dict in params_dict_list:
        pearson_list.append(pool.apply_async(
            train_predict, [geno_data, phe_data, params_dict, cv_time]))

    pool.close()
    pool.join()

    for i, pearson in enumerate(pearson_list):
        params_dict_list[i]['pearson'] = pearson.get()

    # output
    with open(save_path, 'w+') as file:

        header = ['learning_rate', 'max_depth', 'n_estimators', 'min_data_in_leaf', 'num_leaves', 'pearson']

        print('\t'.join(header))
        file.write('\t'.join(header) + '\n')

        for params_dict in params_dict_list:
            params_values = []
            for i in header:
                params_values.append(str(params_dict[i]))

            print('\t'.join(params_values))
            file.write('\t'.join(params_values) + '\n')






