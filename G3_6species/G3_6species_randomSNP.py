# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import read_csv


species = 'maize'
window_size_list = [10000, 50000, 100000, 500000, 1000000, 2000000]
random_times = 10
dir_path = '/data/xyt/G3_6species/'

# Filter SNP by MAF
geno_path = dir_path + species + '_geno.csv'
data = read_csv(geno_path, header=0, index_col=0)
print(species, 'SNP data shape:', data.shape[0])

maf_data = (data == -1).astype(int).sum(axis=0)
maf_data = maf_data/data.shape[0]
maf_path = dir_path + species + '_snp_maf.csv'
maf_data.to_csv(maf_path, index=True, header=False)

maf_15_35 = maf_data[(0.15 <= maf_data) & (maf_data <= 0.35)].dropna()
maf_15_35_chr, maf_15_35_loc, maf_15_35_snp = [], [], []
for snp in maf_15_35.index.values:
    snp_info = snp.split('_')
    try:
        maf_15_35_loc.append(int(snp_info[1]))
        maf_15_35_chr.append(snp_info[0])
        maf_15_35_snp.append(snp)
    except ValueError:
        pass

snp_data = pd.DataFrame({'chr': maf_15_35_chr, 'loc': maf_15_35_loc, 'snpid': maf_15_35_snp})
snp_data.to_csv(dir_path + species + '_snp_maf_15-35.csv', header=True, index=False)
print(species, 'SNP MAF 0.15-0.35 data shape:', snp_data.shape[0])


# Random select SNP by window-size
snp_w_data = snp_data.copy()
for window_size in window_size_list:
    snp_w_data['loc'] = (snp_data['loc']/window_size).astype(int)
    for i in range(random_times):
        snp_select_data = snp_w_data.groupby(['chr', 'loc'])['snpid'].apply(np.random.choice)
        snp_select_path = dir_path + species + '_snp_' + str(int(window_size/1000)) + 'kb_random' + str(i)+'.csv'
        snp_select_data.to_csv(snp_select_path, header=False, index=False)

















