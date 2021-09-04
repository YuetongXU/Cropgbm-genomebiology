# SNP Selection

[![DOI](https://zenodo.org/badge/386224715.svg)](https://zenodo.org/badge/latestdoi/386224715)

## Purpose

We explored the accuracy changes when using different numbers of SNPs to predict phenotypes, and we hope to minimize the SNPs required for prediction without loss of prediction accuracy.

<br>

## method

1. The dataset uses Female_1428, 80% of the samples are randomly selected as the trainset, and the remaining 20% are used as the testset: data/Phenotype.female_1428_values.cvid.csv
2. After the ligbtgbm training is completed, the Pearson correlation coefficient between the predicted phenotype and the true phenotype is calculated, 
   and the IG value of each SNP in the model is extracted. Sort SNPs in descending order according to the IG value.
3. Select `top4000`, `top3000`, `top2000`, `top1000`, `top384`, `top192`, `top96`, `top48`, `top24`, `top12` SNPs as sample features to train the model.
   Calculate the pearson correlation coefficient between the predicted phenotype and the true phenotype.
4. In order to prove the effectiveness of the SNP selected by lightgbm, we conducted two sets of control experiments:
    * Randomly select an equal number of SNPs from all 32559 SNPs, repeat 10 times: data/randomSNP_cv30_snpid/
    * Extract equal number of SNPs from all 32559 SNPs at equal intervals: data/uniformSNP_cv30_snpid/
5. Repeat the above steps 30 times

<br>

## result

The prediction accuracy of modeling with different gradient topSNP, uniformSNP, and randomSNP as features is as follows:

<br>

<font size=5>LGB Maize Female-1428 30-Repeat</font>

<center>

Phe|Method|All|top4000|top3000|top2000|top1000|top384|top192|top96|top48|top24|top12|
---|---|---|---|---|---|---|---|---|---|---|---|---
DTT|IG-top|0.6119|0.6119|0.6119|0.6119|0.6115|0.6044|0.5928|0.5796|0.5523|0.4979|0.4289
DTT|uniform|0.6119|0.5966|0.6099|0.5868|0.5831|0.5297|0.5255|0.5027|0.4265|0.3419|0.2601
DTT|random|0.6119|0.5982|0.5946|0.5884|0.5793|0.5544|0.5146|0.4791|0.4451|0.3747|0.2747
PH|IG-top|0.6433|0.6433|0.6433|0.6433|0.6432|0.6354|0.6271|0.6143|0.5862|0.5401|0.4771
PH|uniform|0.6433|0.6374|0.6379|0.6357|0.6127|0.5867|0.5647|0.5274|0.4721|0.4099|0.2984
PH|random|0.6433|0.6351|0.6323|0.6294|0.6172|0.5936|0.5696|0.5331|0.4792|0.3917|0.3004
EW|IG-top|0.4881|0.4881|0.4881|0.4881|0.4875|0.4783|0.4599|0.4440|0.4113|0.3711|0.3222
EW|uniform|0.4881|0.4829|0.4819|0.4769|0.4624|0.4362|0.4131|0.3712|0.3309|0.2637|0.1944
EW|random|0.4881|0.4754|0.4752|0.4715|0.4590|0.4400|0.4215|0.3861|0.3493|0.3028|0.2185

</center>

* Use scripts: Female_1428_cv.py, Female_1428_reduce_SNP.py
* Recorded in the table is the mean value of the prediction accuracy of 30 repetitions. For the specific results of each repetition, please refer to the file: data/Female_1428_cv30_reduce_(ig/uniform/random)SNP.csv

<br>

The prediction accuracy of rrblup using all SNPs as the frontal feature modeling is as follows:

<br>

<font size=5>rrBLUP Maize Female-1428 30-Repeat</font>

<center>

Phe|DTT|PH|EW
---|---|---|---
Pearson|0.6360|0.6775|0.4804

</center>

* The table records the mean value of the prediction accuracy of 30 repetitions. 
  For the specific results of each repetition, please refer to the file: data/Female_1428_cv30_allSNP_rrblup.csv
* Use script: Female_1428_cv_rrblup.r

<br>


# SNP InformationGain

## Purpose

Observe the IG distribution of SNPs used by the ligbtgbm

## method

Use all samples of the dataset Female_1428 as the training set, extract the IG of the SNP used in the model and draw it into a line graph

## result

* Use the script Female_1428_IGplot.py
* IG of SNP in each model: data/female_1428_(DTT/PH/EW).feature
* The line graph drawn by the IG of SNP in each model: data/female_1428_SNPIG.pdf












