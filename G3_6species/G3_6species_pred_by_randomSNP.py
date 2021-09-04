# -*- coding: utf-8 -*-
from pandas import read_csv
import G3_6species_function as f
import numpy as np


dir_path = '/data/xyt/G3_6species/'
species = 'maize'
method = 'lgb'
phe_name_list = ['HT', 'FT', 'YLD']
window_size_list = [10, 50, 100, 500, 1000, 2000]
random_times = 10
cv_times = 100
cvfold = 5
pool_max = 30

# prepare data
geno, phe_cvid = f.prepare_data(dir_path, species, cv_times)

# predict by random SNP
for phe_namei in phe_name_list:

    # fixed params
    params_dict = {
        'method': method,
        'species': species,
        'phe_name': phe_namei,
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 160,
        'min_data_in_leaf': 20,
        'num_leaves': 10,
        'cv_times': cv_times,
        'cvfold': cvfold,
        'pool_max': pool_max
    }

    for window_size in window_size_list:
        random_pearson = []
        for i in range(random_times):
            snp_select_path = dir_path + species + '_snp_' + str(window_size) + 'kb_random' + str(i) + '.csv'
            snp_select = read_csv(snp_select_path, header=None, index_col=None)
            snp_select = snp_select.values.flatten()
            geno_select = geno.loc[:, snp_select]

            cv_pearson = f.cv(geno_select, phe_cvid, params_dict)
            cv_pearson_mean = np.mean(cv_pearson)
            cv_pearson_std = np.std(cv_pearson)
            random_pearson.append(cv_pearson_mean)

            print(species, phe_namei, method, 'window-size_'+str(window_size),
                  'random_'+str(i), cv_pearson_mean, cv_pearson_std)

        random_pearson_mean = np.mean(random_pearson)
        random_pearson_std = np.std(random_pearson)
        print(species, phe_namei, method, 'window-size_'+str(window_size),
              random_pearson_mean, random_pearson_std)
