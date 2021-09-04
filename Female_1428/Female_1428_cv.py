#  -*- coding: utf-8 -*-
from pandas import read_csv
import Female_1428_function as f
from multiprocessing import Pool


cv_times = 30
cvfold = 5
pool_max = 30
n_estimators = 160
phe_list = ['DTT', 'PH', 'EW']

dir_path = '/data/xyt/data_female_1428/'

# prepare CV data
phe_path = dir_path + 'Phenotype.female_1428_values.cvid.csv'
cvid_path = dir_path + 'Phenotype.female_1428_values.cvid_0.8.csv'

phe_cvid = read_csv(phe_path, header=0, index_col=0)
replace_columns = ['cv' + str(x) for x in range(30)]
phe_cvid[replace_columns] = phe_cvid[replace_columns].astype('int')
phe_cvid[replace_columns] = phe_cvid[replace_columns].replace({0: 1, 2: 1, 3: 1})
phe_cvid.to_csv(cvid_path, header=True, index=True)

# load data
data_path = dir_path + 'Genotype.female_1428_012.txt'
geno_data = read_csv(data_path, index_col=0, header=0)
phe_cvid = read_csv(cvid_path, index_col=0, header=0)

for phe_namei in phe_list:

    params_dict = {
                'phe_name': phe_namei,
                'learning_rate': 0.05,
                'max_depth': 5,
                'n_estimators': n_estimators,
                'min_data_in_leaf': 20,
                'num_leaves': 10,
                'cv_times': cv_times,
                'cvfold': cvfold,
                'pool_max': pool_max
            }

    model_save = dir_path + 'female_1428_' + phe_namei

    pool = Pool(pool_max)
    for cv_time in range(cv_times):
        cv_time = 'cv' + str(cv_time)
        pool.apply_async(f.train_predict, [geno_data, phe_cvid, params_dict, cv_time])

    pool.close()
    pool.join()

    for cv_time in range(cv_times):
        model_path = model_save + '_cv' + str(cv_time) + '.lgb_model'
        feature_save = model_save + '_cv' + str(cv_time) + '.feature'

        tree_info_dict = f.extree_info(model_path, n_estimators)
        f.exfeature_by_regression(tree_info_dict, n_estimators, feature_save)

