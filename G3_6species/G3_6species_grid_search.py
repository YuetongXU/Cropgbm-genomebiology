# -*- coding: utf-8 -*-
import G3_6species_function as f


dir_path = '/data/xyt/G3_6species/'
species = 'maize'
method = 'lgb'
phe_namei = 'HT'
cv_times = 10
cvfold = 5
pool_max = 80

# prepare data
geno, phe_cvid = f.prepare_data(dir_path, species, cv_times)

# grid search
grid_search_params_dict = {
    'learning_rate_list': [0.05, 0.1, 0.2, 0.5],
    'max_depth_list': [5, 20, 40],
    'min_data_in_leaf_list': [1, 5, 20],
    'num_leaves_list': [10, 20, 40],
    'n_estimators_list': [40, 80, 160, 400],

    'cvfold': cvfold,
    'pool_max': pool_max,
    'phe_name': phe_namei
}

for cv_time in range(cv_times):
    cv_time = 'cv'+str(cv_time)
    grid_search_path = dir_path + species + '_' + phe_namei + '_' + method + '_' + cv_time + '_grid_search.tsv'
    print('Species: %s\nPhe_name: %s\nCV_times: %s' % (species, phe_namei, cv_time))
    f.grid_search(geno, phe_cvid, grid_search_path, cv_time, grid_search_params_dict)

















