#  -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas import read_csv
import scipy.stats as stats
from multiprocessing import Pool


def prepare_cv_data(phe_data, save_path, cv_times, cvfold):
    sample_block = int(phe_data.shape[0] / 5)
    phe_index = phe_data.index.to_numpy(copy=True)
    for cvi in range(cv_times):
        random.shuffle(phe_index)
        for i in range(cvfold):
            if i == cvfold - 1:
                phe_data.loc[phe_index[sample_block * i:], 'cv'+str(cvi)] = i
            else:
                phe_data.loc[phe_index[sample_block * i: sample_block * (i + 1)], 'cv'+str(cvi)] = i
    phe_data.sort_index(inplace=True)
    phe_data.to_csv(save_path, header=True, index=True)
    return phe_data


def train_predict(geno_data, phe_data, params_dict, cv_time: str, model_savepath=None):

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

    train_phe = phe_data[phe_data[cv_time] == 1][phe_name].dropna(axis=0)
    test_phe = phe_data[phe_data[cv_time] != 1][phe_name].dropna(axis=0)
    train_geno = geno_data.loc[train_phe.index.values, :]
    test_geno = geno_data.loc[test_phe.index.values, :]

    train_set = lgb.Dataset(train_geno, label=train_phe)
    train_boost = lgb.train(train_params, train_set, n_estimators)
    if model_savepath is not None:
        train_boost.save_model(model_savepath + '_' + cv_time + '.lgb_model')

    predict_phe = train_boost.predict(test_geno)
    pearson, p_value = stats.pearsonr(predict_phe, test_phe)
    print(phe_name, 'All', cv_time, pearson)

    return pearson


def train_predict_reduce(geno_data, phe_data, params_dict, cv_time: str, fnumber_snpid, fnumber: int, model_savepath=None):

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

    train_phe = phe_data[phe_data[cv_time] == 1][phe_name].dropna(axis=0)
    test_phe = phe_data[phe_data[cv_time] != 1][phe_name].dropna(axis=0)

    geno_fnumber = geno_data.loc[:, fnumber_snpid].copy()

    train_geno = geno_fnumber.loc[train_phe.index.values, :]
    test_geno = geno_fnumber.loc[test_phe.index.values, :]

    train_set = lgb.Dataset(train_geno, label=train_phe)
    train_boost = lgb.train(train_params, train_set, n_estimators)
    if model_savepath is not None:
        train_boost.save_model(model_savepath + '_' + cv_time + '.lgb_model')

    predict_phe = train_boost.predict(test_geno)
    pearson, p_value = stats.pearsonr(predict_phe, test_phe)
    print(phe_name, fnumber, cv_time, pearson)

    return pearson


def train_predict_cv(geno_data, phe_data, params_dict, cv_time: str, model_savepath=None):

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
            train_predict_cv, [geno_data, phe_data, params_dict, cv_time, save_path]))

    pool.close()
    pool.join()

    pearson_list = []
    for pearson in pool_list:
        pearson_list.append(pearson.get())

    return pearson_list


def extree_info(model_path, num_boost_round, tree_info_dict=None):
    if tree_info_dict is None:
        tree_info_dict = {}
        for tree_index in range(0, num_boost_round):
            tree_info_dict['tree_' + str(tree_index)] = {}

    for i, row in enumerate(open(model_path)):
        if row.find('feature_names') != -1:
            features_name_list = row.strip().split('=')[-1].split(' ')
            continue
        if row.find('Tree=') != -1:
            tree_index = row.strip().split('=')[-1]
            continue
        if row.find('split_feature') != -1:
            features_index_list = row.strip().split('=')[-1].split(' ')
            features_index_list = [int(i) for i in features_index_list]
            continue
        if row.find('split_gain') != -1:
            features_gain_list = row.strip().split('=')[-1].split(' ')
            features_gain_list = [float(i) for i in features_gain_list]
            seq_index = 0
            for index in features_index_list:
                feature_name = features_name_list[index]
                feature_gain = features_gain_list[seq_index]
                try:
                    tree_info_dict['tree_' + tree_index][feature_name].append(feature_gain)
                except KeyError:
                    tree_info_dict['tree_' + tree_index][feature_name] = [feature_gain]
                seq_index += 1

    return tree_info_dict


def exfeature_by_regression(tree_info_dict, num_boost_round, save_path=None):
    alltree = pd.DataFrame()
    for itree in range(0, num_boost_round):
        feature_id, feature_gain = [], []
        itree_info_dict = tree_info_dict['tree_' + str(itree)]
        for feature_name in itree_info_dict:
            feature_id.append(feature_name)
            feature_gain.append(sum(itree_info_dict[feature_name]))
        itree_df = pd.DataFrame({'featureid': feature_id, 'tree_' + str(itree): feature_gain})
        try:
            alltree = alltree.merge(itree_df, how='outer', on='featureid')
        except KeyError:
            alltree = itree_df

    alltree.fillna(0, inplace=True)
    feature_gain_sum = alltree.sum(axis=1)
    alltree.insert(1, 'featureGain_sum', feature_gain_sum)
    alltree = alltree.sort_values(by='featureGain_sum', axis=0, ascending=False)
    alltree.to_csv(save_path, index=None)


def summery_cv_feature(model_prefix, cv_times: int, cvfold: int, n_estimators: int):

    for cv_time in range(cv_times):
        model_cv_path = model_prefix + '_cv' + str(cv_time)
        for cv_foldi in range(cvfold):
            model_path = model_cv_path + '_fold' + str(cv_foldi) + '.lgb_model'
            if (cv_time == 0) and (cv_foldi == 0):
                tree_info_dict = extree_info(model_path, n_estimators)
            else:
                tree_info_dict = extree_info(model_path, n_estimators, tree_info_dict)

    feature_save = model_prefix + '_cv' + str(cv_times) + '.feature'
    exfeature_by_regression(tree_info_dict, n_estimators, feature_save)


