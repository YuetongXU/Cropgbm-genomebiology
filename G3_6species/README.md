# Data Information

[![DOI](https://zenodo.org/badge/386224715.svg)](https://zenodo.org/badge/latestdoi/386224715)

This experiment uses the database in the G3 article: 

> Benchmarking Parametric and Machine Learning Models for Genomic Prediction of Complex Traits. Christina B Azodi, Emily Bolger, Andrew McCarren, Mark Roantree, Gustavo de los Campos, Shin-Han Shiu. G3 Genes|Genomes|Genetics, Volume 9, Issue 11, 1 November 2019, Pages 3691–3702, https://doi.org/10.1534/g3.119.400498
 

The database contains the genotype and phenotype data of 6 species. The specific information is as follows:

<br>

<font size=5>G3 6 species information</font>

|   |Rice | Maize | Soy | Spruce | Switchgrass | Sorghum |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
Phenotype | HT, FT, YLD | HT, FT, YLD | HT, R8, YLD | HT, DBH, DE | HT, FT, ST | HT, GM, YLD
SNP num  | 57543 | 244782 | 4235 | 6931 | 217151 | 56300
SNP type | GBS | GBS | SNP-Chip| SNP-Chip| GBS | GBS
Sample num | 327 | 391 | 5014 | 1722 | 514 | 451
Population | Diversity Panel | Diversity Panel | NAM | Partial DM | Diversity Panel | Diversity Panel


>HT: height. FT: flowering time. YLD: yield. GM: grain moisture. R8: time to R8 developmental stage. DBH: diameter at breast height. DE: wood density. ST: standability.

<br>

# Hyperparameters Grid-search


We use the method of grid search (_G3_6species_grid_search.py_) to detect the best combination of the following 5 hyperparameters of LightGBM in the HT phenotype of the G3_6species database:

<br>

<font size=5>Hyperparameters Grid-search (5-CV 5-fold)</font>

<center>

|Hyperparameter | Values|
|:---:|:---:|
|Learning rate | 0.05, 0.1, 0.2, 0.5|
|max_depth | 5, 20, 40|
|min_data_in_leaf | 1, 5, 20|
|num_leaves |10, 20, 40|
|n_estimators |40, 80, 160, 400|

</center>

<br>

The best combination of hyperparameter values for different species obtained by grid search is as follows:

<br>

<font size=5>Best-Parameters Grid-search (5-CV 5-fold)</font>

<center>

species | learning_rate | max_depth | n_estimators | min_data_in_leaf | num_leaves
:---:|:---:|:---:|:---:|:---:|:---:
rice | 0.1 | 20 | 160 | 20 | 10
sorghum | 0.1 | 5 | 40 | 5 | 20
soy | 0.1 | 20 | 40 | 20 | 20
spruce | 0.1 | 5 | 40 | 20 | 10
switchgrass | 0.05 | 5 | 40 | 20 | 10
maize | 0.05 | 5 | 160 | 20 | 10

</center>

<br>

# Fixed Hyperparameter


Considering that grid search is time-consuming, it is recommended to use fixed parameters that are robust in each species in the practical application of LGB (_learning_rate = 0.05, max_depth = 5, n_estimators = 160, min_data_in_leaf = 20, num_leaves = 10_) to make predictions for each species. Although the prediction accuracy is slightly lower than the best parameters, it saves a lot of time for users to find the best parameters through grid search. The prediction results of the HT phenotypes of 6 species using fixed parameters are as follows (100-CV 5-fold, the way of dividing the data set by each CV is consistent with the reference):

<br>

<font size=5>Fixed-Parameters Prediction Pearson (HT, 100-CV, 5-fold)</font>

<center>

| |Maize | Rice | Sorghum | Soy | Spruce | Switchgrass
:---:|:---:|:---:|:---:|:---:|:---:|:---:
rrBLUP | **0.44** | 0.34 | **0.63** | 0.46 | 0.32 | **0.61**
BRA | **0.44** | 0.39 | **0.63** | 0.46 | 0.32 | **0.61**
BA | 0.42 | 0.38 | **0.63** | 0.47 | 0.32 | **0.61**
BB | 0.43 | 0.38 | **0.63** | 0.46 | 0.32 | **0.61**
BL | **0.44** | 0.39 | 0.62 | 0.46 | 0.32 | **0.61**
SVRlin | 0.41 | 0.38 | 0.62 | 0.43 | 0.19 | 0.60
SVRpoly | 0.43 | 0.38 | **0.63** | 0.41 | 0.33 | **0.61**
SVRrbf | 0.39 | 0.38 | **0.63** | 0.04 | 0.34 | 0.60
RF | 0.43 | 0.40 | 0.58 | 0.36 | **0.35** | 0.57
GTB | 0.37 | 0.38 | 0.58 | 0.40 | 0.33 | 0.56
ANN | 0.17 | 0.08 | 0.45 | 0.44 | 0.28 | 0.45
LGB | 0.40 | **0.41** | 0.56 | **0.48** | 0.30 | 0.60

</center>

* Use the script _G3_6species_cv.py_
* LGB hyperparameters: _learning_rate = 0.05, max_depth = 5, n_estimators = 160, min_data_in_leaf = 20, num_leaves = 10_
* For the detailed results of each CV, refer to: _/data/cv100_pearson_6species_HT.csv_

<br>


# Random SNP


Use the MAF $\in$ [0.15, 0.35] to filter the 244782 SNPs of Maize in the G3_6species database to obtain 100068 SNPs (_/data/maize_snp_maf_15-35.csv_).

Among 100068 SNPs, chromosomes are divided into block lengths of 10KB, 50KB, 100KB, 500KB, 1MB and 2MB, respectively. Randomly select 1 SNP in each block. Here we have randomized 10 times in total.

<br>

<center>

Block | 10KB | 50KB | 100KB | 500KB | 1MB | 2MB
:---:|:---:|:---:|:---:|:---:|:---:|:---:
SNP | 20025 | 13186 | 10353 | 3667 | 1978 | 1029

</center>

* Use the script _G3_6species_randomSNP.py_
* See the SNP_ID randomly selected each time in: _/data/random_10/_

<br>

Using the above 6 * 10 groups of SNP sets as training features of LGB to predict HT, FT, and YLD phenotypes, perform 100-CV, 5-Fold, and the results are as follows:

<br>

<font size=5>LGB Maize 10-Random 100-CV 5-Fold</font>

<center>

| | ALL SNP | 10KB | 50KB | 100KB | 500KB | 1MB | 2MB
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
HT | 0.4041 | 0.3939 | 0.3945 | 0.3864 | 0.3841 | 0.3760 | 0.3583
FT | 0.7082 | 0.6890 | 0.6782 | 0.6773 | 0.6753 | 0.6535 | 0.6220
YLD | 0.5051 | 0.4967 | 0.4960 | 0.4874 | 0.4918 | 0.4647 | 0.4509

</center>

* Use the script _G3_6species_pred_by_randomSNP.py_
* LGB hyperparameters: _learning_rate = 0.05, max_depth = 5, n_estimators = 160, min_data_in_leaf = 20, num_leaves = 10_
* For the detailed results of each CV of LGB, refer to: _/data/cv100_pearson_maize_lgb.csv (ALL SNP)_, _random10_cv100_maize_HT/FT/YLD/summery_lgb.csv_

<br>

<font size=5>rrBLUP Maize 10-Random 100-CV 5-Fold</font>

<center>

| | ALL SNP | 10KB | 50KB | 100KB | 500KB | 1MB | 2MB
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
HT | 0.4627 | 0.4556 | 0.4528 | 0.4432 | 0.4294 | 0.4129 | 0.4041
FT | 0.7643 | 0.7590 | 0.7549 | 0.7533 | 0.7199 | 0.6973 | 0.6384
YLD | 0.5420 | 0.5372 | 0.5381 | 0.5373 | 0.5199 | 0.5075 | 0.4942

</center>

* Use the script _G3_6species_randomSNP_rrblup.r_, _G3_6species_allSNP_rrblup.r_
* For the detailed results of each CV of rrBLUP, refer to: _/data/cv100_pearson_maize_rrblup.csv (ALL SNP)_, _random10_cv100_maize_rrblup.csv_


