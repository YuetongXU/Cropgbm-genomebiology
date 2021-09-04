# -*- coding: utf-8 -*-
import G3_6species_function as f
import numpy as np


dir_path = '/data/xyt/G3_6species/'
species_list = ['rice', 'sorghum', 'soy', 'spruce', 'switchgrass', 'maize']
method = 'lgb'
phe_list = ['HT', 'FT', 'YLD']
cv_times = 100
cvfold = 5
pool_max = 30

for species in species_list:
    for phe_namei in phe_list:
        geno, phe_cvid = f.prepare_data(dir_path, species, cv_times)

        # CV
        params_dict = {
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

        model_save = dir_path + species + phe_namei
        cv_pearson_list = f.cv(geno, phe_cvid, params_dict, model_save)
        cv_pearson_mean = np.mean(cv_pearson_list)
        cv_pearson_std = np.std(cv_pearson_list)

        print(species, phe_namei, method, cv_pearson_mean, cv_pearson_std)





