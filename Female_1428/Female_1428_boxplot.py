# -*- coding: utf-8 -*-
import pandas as pd
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


dir_path = '/home/xyt/Data/cropgbm_female_1428/'

data_ig = read_csv(dir_path + 'Female_1428_cv30_reduce_igSNP.csv', header=0, index_col=None)
data_random = read_csv(dir_path + 'Female_1428_cv30_reduce_randomSNP_summery.csv', header=0, index_col=None)
data_rrblup = read_csv(dir_path + 'Female_1428_cv30_allSNP_rrblup.csv', header=0, index_col=0).T

data_random = data_random.loc[:, ['Phe', 'fnumber', 'cvid', 'pearson_mean']]
data_random.columns = ['Phe', 'fnumber', 'cvid', 'pearson']

all_snp = data_ig[data_ig['fnumber'] == 'All']
data_random = pd.concat([data_random, all_snp], ignore_index=True)

data_ig['snp_source'] = ['IG']*data_ig.shape[0]
data_random['snp_source'] = ['random']*data_random.shape[0]

data = pd.concat([data_ig, data_random], ignore_index=True)
data['fnumber'] = data['fnumber'].astype('str')

save_path = dir_path + 'Female_1428_cv30_reduceSNP'
phe_name_list = ['DTT', 'PH', 'EW']
with PdfPages(save_path + '.pdf') as pdf:
    for phe_namei in phe_name_list:
        data_phe_namei = data[data['Phe'] == phe_namei]

        plt.figure(figsize=(12, 9))
        sns.boxplot(data=data_phe_namei, x='fnumber', y='pearson', width=0.3, linewidth=2.0, hue='snp_source',
                    order=['All', '4000', '3000', '2000', '1000', '384', '192', '96', '48', '24', '12'])

        line_y = data_rrblup[phe_namei].mean()
        plt.plot((-0.3, 11), [line_y, line_y], c='r', linewidth=2, linestyle='--')

        plt.title('Maize '+phe_namei+' 30-CV', fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(0, 0.8)
        plt.xlabel(None, fontsize=20)
        plt.xlabel('Number of SNPs', fontsize=20)
        plt.ylabel('Pearson Correlation', fontsize=20)

        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        plt.tight_layout()
        pdf.savefig()




