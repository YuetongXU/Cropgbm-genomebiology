#  -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import read_csv
import Female_1428_function as f
from multiprocessing import Pool


dir_path = '/data/xyt/data_female_1428/'
phe_list = ['DTT', 'PH', 'EW']
fnumber_list = [12, 24, 48, 96, 192, 384, 1000, 2000, 3000, 4000]
cv_times = 30
random_times = 10
cvfold = 5
pool_max = 30
n_estimators = 160


geno_path = dir_path + 'Genotype.female_1428_012.txt'
geno_data = read_csv(geno_path, header=0, index_col=0)
all_snpid = geno_data.columns.values.copy()

for phe_namei in phe_list:

    phe_ig_path = dir_path + 'Phenotype.female_1428_values.cvid_0.8.csv'
    phe_ig_data = read_csv(phe_ig_path, header=0, index_col=0)

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

    for fnumber in fnumber_list:

        # Select SNP by ig
        pool = Pool(pool_max)
        for cv_time in range(cv_times):
            cv_time = 'cv' + str(cv_time)
            feature_path = dir_path + 'cv30_0.8_lgb_model/female_1428_' + phe_namei + '_' + cv_time + '.feature'
            feature_data = read_csv(feature_path, header=0, index_col=0)
            snpid = feature_data.index.values.copy()
            fnumber_snpid = snpid[: fnumber]

            pool.apply_async(f.train_predict_reduce, [geno_data, phe_ig_data, params_dict,
                                                      cv_time, fnumber_snpid, fnumber])

        pool.close()
        pool.join()

        # Select SNP by uniform
        uniform_index = np.linspace(0, len(all_snpid), fnumber, endpoint=False)
        uniform_index = [int(x) for x in uniform_index]
        uniform_snpid = all_snpid[uniform_index]
        uniform_snpid_df = pd.DataFrame({'snpid': uniform_snpid})
        uniform_snpid_df.to_csv(dir_path + 'female_1428_' + str(fnumber) + 'SNP_uniform.snpid',
                                header=True, index=False)

        pool = Pool(pool_max)
        for cv_time in range(cv_times):
            cv_time = 'cv' + str(cv_time)
            pool.apply_async(f.train_predict_reduce, [geno_data, phe_ig_data, params_dict,
                                                      cv_time, uniform_snpid, fnumber])

        pool.close()
        pool.join()

    print('Uniform is OVER')

    # Select SNP by random
    for r in range(random_times):
        random_index = np.random.permutation(32559)
        random_4000snpid = all_snpid[random_index[: 4000]]
        random_4000snpid_df = pd.DataFrame({'snpid': random_4000snpid})
        random_save = dir_path + 'female_1428_' + phe_namei + '_random' + str(r)
        random_4000snpid_df.to_csv(random_save + '.4000snpid', header=True, index=False)

        for fnumber in fnumber_list:
            random_snpid = random_4000snpid[: fnumber]

            pool = Pool(pool_max)
            for cv_time in range(cv_times):
                cv_time = 'cv' + str(cv_time)
                pool.apply_async(f.train_predict_reduce, [geno_data, phe_ig_data, params_dict,
                                                          cv_time, random_snpid, fnumber])

            pool.close()
            pool.join()

    print('Random is OVER')



