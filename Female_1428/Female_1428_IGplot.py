#!/usr/bin/env python
#  -*- coding: utf-8 -*-
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import lightgbm as lgb
import Female_1428_function as f


phe_list = ['DTT', 'PH', 'EW']
dir_path = '/data/xyt/data_female_1428/'
geno_path = dir_path + 'Genotype.female_1428_012.txt'
phe_path = dir_path + 'Phenotype.female_1428_values.txt'
n_estimators = 160

geno_data = read_csv(geno_path, header=0, index_col=0)
phe_data = read_csv(phe_path, header=0, index_col=0)

for phe_namei in phe_list:
    params_dict = {
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_data_in_leaf': 20,
            'num_leaves': 10,
        }

    phei_data = phe_data[phe_namei].dropna(axis=0)
    genoi_data = geno_data.loc[phei_data.index.values, :]
    train_set = lgb.Dataset(genoi_data, label=phei_data)
    train_boost = lgb.train(params_dict, train_set, n_estimators)

    model_prefix = dir_path + 'female_1428_' + phe_namei
    train_boost.save_model(model_prefix + '.lgb_model')
    tree_info_dict = f.extree_info(model_prefix + '.lgb_model', n_estimators)
    f.exfeature_by_regression(tree_info_dict, n_estimators, model_prefix + '.feature')


# plot IG
dir_path = '/data/xyt/cropgbm_female_1428/'
phe_list = ['DTT', 'PH', 'EW']
top_num = 12

unit_length = 1e6
chrom_interval = 15 * unit_length
window_size = 10 * unit_length
step_size = 1 * unit_length
chrom_length = [307038713, 244420837, 235648871, 246989316, 223899308, 174029885, 182379431, 181121275, 159765265,
                150979161]

chrom_start = []
chrom_end_i = 0
for i in range(10):
    chrom_start.append(chrom_end_i)
    chrom_end_i = chrom_end_i + chrom_length[i] + chrom_interval

chrom_start = np.array(chrom_start)
chrom_length = np.array(chrom_length)

with PdfPages(dir_path + 'female_1428_SNPIG.pdf') as pdf:
    for phe_namei in phe_list:

        save_path = dir_path + 'female_1428_' + phe_namei

        # extract SNP info
        feature_data = read_csv(save_path + '.feature', header=0, index_col=0)
        snp_id = feature_data.index.values
        snp_ig = feature_data['featureGain_sum'].values

        snp_loc = []
        for snpi in snp_id:
            snpi_chr = int(snpi.split('.s_')[0].split('chr')[1])
            snpi_loc = int(snpi.split('.s_')[1])
            snpi_loc = snpi_loc + chrom_start[snpi_chr-1]
            snp_loc.append(snpi_loc)

        window_ig = []
        window_loc = []
        snp_loc = np.array(snp_loc)
        for window_start_i in range(0, int(sum(chrom_length) + chrom_interval * 9), int(step_size)):
            window_loc.append(window_start_i)
            window_end_i = window_start_i + window_size
            window_i_ig = sum(snp_ig[np.where((window_start_i <= snp_loc) & (snp_loc < window_end_i))])
            window_ig.append(window_i_ig)

        # plot
        plt.figure(figsize=(25, 6))

        # ig
        plt.plot(window_loc, window_ig, c='black', linewidth=0.8)

        # chrom
        color_list = ['gray', 'black'] * 5
        for k in range(10):
            c_start = chrom_start[k]
            c_end = chrom_start[k] + chrom_length[k]
            if phe_namei == 'DTT':
                line_y = -1000
            elif phe_namei == 'PH':
                line_y = -20000
            else:
                line_y = -10000
            plt.plot((c_start, c_end), [line_y, line_y], c=color_list[k], linewidth=6)

        chrom_end = chrom_start + chrom_length
        chrom_mid = (chrom_start + chrom_end)/2
        plt.xticks(chrom_mid, range(1, 11))
        plt.xlabel('Chromosome', fontsize=20)
        plt.ylabel('IG SUM', fontsize=20)
        plt.title('1428cv ' + phe_namei + ' top384 IG SUM', fontsize=20)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

