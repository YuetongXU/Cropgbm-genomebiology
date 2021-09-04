#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tested in scikit-learn-0.24.2
# pip intall scikit-learn rpy2-3.4.5 matplotlib xgboost lightgbm catboost
# R: install.packages("rrBLUP")

__author__ = 'YanJ'
__version__ = 'v1.0'
__date__ = '2021-06'

import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import os, sys, getopt
import argparse
import pandas as pd
from pandas import read_csv
import numpy as np
from math import sqrt
from scipy import interp, optimize, stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pylab import *
import itertools
from itertools import cycle
import joblib
import copy 
from operator import itemgetter  

from sklearn.metrics import roc_curve, auc, classification_report, r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score 
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, LeavePOut, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, LassoLars, BayesianRidge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from catboost import CatBoostClassifier, CatBoostRegressor

import rpy2.robjects as robjects
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
pandas2ri.activate()


def oned2TwodY(y, n):
	y = np.ndarray.tolist(y)
	newy=[[0 for i in range(len(y))] for i in range(n)]
	for k in range(len(y)):
			c = int(y[k] - 1)
			newy[c][k] = 1
	return newy 


# regression 
def evaluation(observed_y, pred_y, sample, name):
# output r2/r2_spearman/mse/rmse/top10ol/top20ol/top30ol
	name = name
	y2 = observed_y
	my_eachY = pred_y
	sample = sample
	cor = cal_pearson(y2, my_eachY)
	cor_sp = cal_spearman(y2, my_eachY)
	mse = cal_mse(y2, my_eachY)
	rmse = cal_rmse(y2, my_eachY)
	y_dict = {k:v for k,v in zip(sample,y2)}
	my_dict = {k:v for k,v in zip(sample,my_eachY)}
	top10y = top_n(y_dict, 0.1)
	top10my = top_n(my_dict, 0.1)
	top10overlap = len(list_overlap(top10y, top10my))*10.0/len(sample)
	top20y = top_n(y_dict, 0.2)
	top20my = top_n(my_dict, 0.2)
	top20overlap = len(list_overlap(top20y, top20my))*5.0/len(sample)
	top30y = top_n(y_dict, 0.3)
	top30my = top_n(my_dict, 0.3)
	top30overlap = len(list_overlap(top30y, top30my))*3.3333/len(sample)
	print ("%s : %s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (name, cor, cor_sp, mse, rmse, top10overlap, top20overlap, top30overlap))
	return (cor, cor_sp, mse, rmse, top10overlap, top20overlap, top30overlap)

#classify: 2 class
def evaluation2(observed_y, pred_y, pred_y_prob, sample, name, output_prefix):
# output top_ol/mid_ol/low_ol
	y2 = observed_y
	my_eachY = pred_y
	my_eachY_prob = pred_y_prob
	sample = sample
	name = name
	y_dict = {k:v for k,v in zip(sample,y2)}
	my_dict = {k:v for k,v in zip(sample,my_eachY)}
	top_y = [k for k,v in y_dict.items() if v == 0]
	top_my = [k for k,v in my_dict.items() if v == 0.0]
	top_overlap = 0
	if len(top_y) > 0:
		top_overlap = len(list_overlap(top_y, top_my))*1.0/len(top_y)
	mid_y = [k for k,v in y_dict.items() if v == 2]
	mid_my = [k for k,v in my_dict.items() if v == 2.0]
	mid_overlap = 0
	if len(mid_y) > 0:
		mid_overlap = len(list_overlap(mid_y, mid_my))*1.0/len(mid_y)
	overall_overlap = (len(list_overlap(top_y, top_my)) + len(list_overlap(mid_y, mid_my)) )*1.0/len(y2)
	auc = plotRoc2(observed_y, pred_y_prob, output_prefix)
	print ("%s : %s\t%s\t%s\t%s\n" % (name, auc, overall_overlap, top_overlap, mid_overlap))

#multiclassify: 3 class roc	
def plotRoc2(observed_y, pred_y_prob, output_prefix):
	y2 = observed_y
	my_eachY_prob = pred_y_prob
	name = output_prefix
	fig = plt.figure(figsize=(4,4),dpi=100)
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	top_auc = 0.0
	colors = cycle(["red"])
	labels = cycle(["AUC"])
	lw = 1
	plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='grey')
	for n, mycolor, mylabel in zip(range(1), colors, labels):
		probas_ = my_eachY_prob
		newy = np_utils.to_categorical(y2)
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(newy[:,n], probas_[:,n])
		mean_tpr = interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		mean_tpr[-1] = 1.0
		plt.plot(mean_fpr, mean_tpr, color=mycolor, label='%s (%0.3f)' % (mylabel, roc_auc), lw=lw)
		mean_tpr = 0.0
		if top_auc == 0.0:
			top_auc = roc_auc
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC ('+name+')')
	plt.legend(loc="lower right")
	plt.show()
	fig.savefig(name+'_ROC.pdf')
	return top_auc	

#multiclassify: 3 class
def evaluation3(observed_y, pred_y, pred_y_prob, sample, name, output_prefix):
# output top_ol/mid_ol/low_ol
	y2 = observed_y
	my_eachY = pred_y
	my_eachY_prob = pred_y_prob
	sample = sample
	name = name
	y_dict = {k:v for k,v in zip(sample,y2)}
	my_dict = {k:v for k,v in zip(sample,my_eachY)}
	top_y = [k for k,v in y_dict.items() if v == 0]
	top_my = [k for k,v in my_dict.items() if v == 0.0]
	
	top_overlap = 0
	mid_overlap = 0
	low_overlap = 0
	if len(top_y) > 0:
		top_overlap = len(list_overlap(top_y, top_my))*1.0/len(top_y)
	mid_y = [k for k,v in y_dict.items() if v == 1]
	mid_my = [k for k,v in my_dict.items() if v == 1.0]
	if len(mid_y) > 0:
		mid_overlap = len(list_overlap(mid_y, mid_my))*1.0/len(mid_y)
	low_y = [k for k,v in y_dict.items() if v == 2]
	low_my = [k for k,v in my_dict.items() if v == 2.0]
	if len(low_y) > 0:
		low_overlap = len(list_overlap(low_y, low_my))*1.0/len(low_y)
	overall_overlap = (len(list_overlap(top_y, top_my)) + len(list_overlap(mid_y, mid_my)) + len(list_overlap(low_y, low_my)))*1.0/len(y2)
	auc = plotRoc3(observed_y, pred_y_prob, output_prefix)
	print ("%s : %s\t%s\t%s\t%s\t%s\n" % (name, auc, overall_overlap, top_overlap, mid_overlap, low_overlap))

#multiclassify: 3 class roc	
def plotRoc3(observed_y, pred_y_prob, output_prefix):
	y2 = observed_y
	my_eachY_prob = pred_y_prob
	name = output_prefix
	fig = plt.figure(figsize=(4,4),dpi=100)
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	top_auc = 0.0
	colors = cycle(["red", "green", "blue"])
	labels = cycle(["Top","Mid","Low"])
	lw = 1
	plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='grey')
	for n, mycolor, mylabel in zip(range(3), colors, labels):
		probas_ = my_eachY_prob
		newy = np_utils.to_categorical(y2)
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(newy[:, n], probas_[:, n])
		mean_tpr = interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		mean_tpr[-1] = 1.0
		plt.plot(mean_fpr, mean_tpr, color=mycolor, label='%s (%0.3f)' % (mylabel, roc_auc), lw=lw)
		mean_tpr = 0.0
		if top_auc == 0.0:
			top_auc = roc_auc
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC ('+name+')')
	plt.legend(loc="lower right")
	plt.show()
	fig.savefig(name+'_ROC.pdf')
	return top_auc



def top_n(dict, percent):
	if sys.version.split('.')[0]==str(2):
		sorted_dict = sorted(dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
	if sys.version.split('.')[0]==str(3):
		sorted_dict = sorted(dict.items(), key = itemgetter(1), reverse=True)
	top_num=int(len(sorted_dict)*percent)
	top_list = sorted_dict[0:top_num]
	top_id = [str(each[0]) for each in top_list]
	return top_id

def list_overlap(a,b):
	overlap_len = set(a).intersection(b)
	return overlap_len
	
def t2dto3d(X, unit):
	return np.array(X).reshape(X.shape[0],int(X.shape[1]/unit),unit)
	
def readfile2list(file):
	f1 = open(file)
	list = []
	for line in f1.readlines():
		list.append(line.strip());	
	f1.close()
	return list

def maskTest(input_arr, mask_elements,invert=True):
		input_arr = np.array(input_arr)
		mask_elements = np.array(mask_elements)
		mask = np.isin(input_arr, mask_elements, invert=invert)
		return  mask

def standardizeT(X):  
	m, n = X.shape
	# 训练集：归一化每一个特征
	t1 = []
	t2 = []
	for j in range(n):  
		features = X[:,j]  
		meanVal = features.mean(axis=0)
		std = features.std(axis=0)
		t1.append(meanVal)
		t2.append(std)
		if std != 0:  
			X[:, j] = (features-meanVal)/std  
		else: 
			X[:, j] = 0  
	return (X, t1, t2)
  
def standardizeV(X, t1, t2):
	m, n = X.shape
	# 验证集：归一化每一个特征  
	for j in range(n):
		features = X[:,j]
		meanVal = t1[j]
		std = t2[j]
		if std != 0:
			X[:, j] = (features-meanVal)/std
		else:
			X[:, j] = 0
	return X

def normalizeT(X):  
	# Min-Max normalization	 sklearn.preprocess的MaxMinScalar  
	m, n = X.shape  
	# 训练集：归一化每一个特征  
	t1 = []
	t2 = []
	for j in range(n):
		features = X[:,j]
		minVal = features.min(axis=0)
		maxVal = features.max(axis=0)
		diff = maxVal - minVal
		t1.append(minVal)
		t2.append(diff)
		if diff != 0:
			X[:,j] = (features-minVal)/diff
		else:
			X[:,j] = 0
	return (X, t1, t2)
	
def normalizeV(X, t1, t2): 
	# Min-Max normalization	 sklearn.preprocess的MaxMinScalar  
	m, n = X.shape  
	# 验证集：归一化每一个特征  
	for j in range(n):
		features = X[:,j]
		minVal = t1[j]
		diff = t2[j]
		if diff != 0:
			X[:,j] = (features-minVal)/diff
		else:
			X[:,j] = 0
	return X

def cal_mse(a,b):
	mse = mean_squared_error(a,b)
	return mse

def cal_rmse(a,b):
	mse = mean_squared_error(a,b)
	rmse = pow(mse, 1/2)
	return rmse

def cal_mse2(target, prediction):
	error = []  
	for i in range(len(target)):
		error.append(target[i] - prediction[i])   
	squaredError = []  
	for val in error:  
		squaredError.append(val*val)
	return sum(squaredError)/len(squaredError)

def cal_msle(a,b):
	try:
		msle = mean_squared_log_error(a,b)
		return msle
	except:
		msle = 'nan'
		return msle

# this is for cal_pearson()
def multiply(a,b):
	#a,b两个列表的数据一一对应相乘之后求和
	sum_ab=0.0
	for i in range(len(a)):
		temp=a[i]*b[i]
		sum_ab+=temp
	return sum_ab

def cal_pearson(x,y):
	n=len(x)
	#求x_list、y_list元素之和
	sum_x=sum(x)
	sum_y=sum(y)
	#求x_list、y_list元素乘积之和
	sum_xy=multiply(x,y)
	#求x_list、y_list的平方和
	sum_x2 = sum([pow(i,2) for i in x])
	sum_y2 = sum([pow(j,2) for j in y])
	molecular=sum_xy-(float(sum_x)*float(sum_y)/n)
	#计算Pearson相关系数，molecular为分子，denominator为分母
	denominator=sqrt((sum_x2-float(sum_x**2)/n)*(sum_y2-float(sum_y**2)/n))
	if denominator != 0:
		return molecular/denominator
	else:
		return "NA"

#calculate spearman r2
def cal_spearman(y1,y2):
	cor = stats.spearmanr(y1,y2)
	return cor.correlation
	

# this is for plotCor()
def f_1(x, A, B):  
	return A*x + B

def plotCor(predict,real,title,plotname):
	mpl.rcParams['font.size'] = 9.0
	fig = plt.figure(figsize=(3,3),dpi=600)
	densityScat = fig.add_subplot(111)
	plt.sca(densityScat)
	x=np.array(predict,dtype=np.float64)
	y=np.array(real,dtype=np.float64)
	plt.scatter(x, y, s=5)
	plt.title(title)
	A1, B1 = optimize.curve_fit(f_1, x, y)[0]  
	x1 = np.arange(min(x), max(x), 0.01)  
	y1 = A1*x1 + B1
	plt.plot(x1, y1, "red") 
	plt.show()
	fig.savefig(plotname)


def modelfit(alg, X, y, printFeatureImportance=True, cv_folds=5):
	#Fit the algorithm on the data
	alg.fit(X, y)
	
	#Predict training set:
	dtrain_predictions = alg.predict(X)

	#Perform cross-validation:
	cv_score = cross_val_score(alg, X, y, cv=cv_folds, scoring='r2')
	mse_score = cross_val_score(alg, X, y, cv=cv_folds, scoring='neg_mean_squared_error')

	#Print model report:
	print ("r2 Score (Train): %f " % r2_score(y, dtrain_predictions))
	print ("r2 Score (CV) : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g " % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
	print ("mse Score (Train): %f " % mean_squared_error(y, dtrain_predictions))
	print ("mse Score (CV) : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g " % (np.mean(mse_score),np.std(mse_score),np.min(mse_score),np.max(mse_score)))
	
	#Print Feature Importance:
	if printFeatureImportance:
		feat_imp = pd.Series(alg.feature_importances_,).sort_values(ascending=False)
		#feat_imp.plot(kind='bar', title='Feature Importances')
		#plt.ylabel('Feature Importance Score')
		print (feat_imp)


def modelSearch(clf):
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	params = clf.cv_results_['params']
	for mean, std, params in zip(means, stds, params):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	print (clf.best_params_)
	print (clf.best_score_)
	return (clf.best_params_,clf.best_score_)

def readInputFiles2(snpinfo, data, n):
	n=n
	df = pd.read_csv(data)
	info = pd.read_csv(snpinfo)
	sinfo = info[info.iloc[:,1]==1]
	df1 = df.loc[:,list(sinfo.iloc[:,0])]
	dff = df1.fillna(value=1)
	#dff = df1.fillna(value=df1.mode().to_dict(orient='records')[0])
	X = np.array(dff,"float64")
	y_raw = np.array(df.iloc[:,n],"float64")
	my_imputer = SimpleImputer()
	y = my_imputer.fit_transform(y_raw.reshape(-1, 1)).reshape(-1,)
	trait = list(df.columns.values)[n]
	sample = list(df.iloc[:,0])
	return (X, y, trait, sample, y_raw)	


#回归模型
def regression(reg_name):
	if reg_name == 'knn':
		classifier = KNeighborsRegressor()
	if reg_name == 'mlp':
		classifier = MLPRegressor(random_state=1)
	if reg_name == 'svr':
		classifier = SVR()
	if reg_name == 'rf':
		classifier = RandomForestRegressor(random_state=1)
	if reg_name == 'gb':
		classifier = GradientBoostingRegressor(random_state=1)
	if reg_name == 'xgb':
		classifier = XGBRegressor(seed=1)
	if reg_name == 'lgb':
		classifier = LGBMRegressor()
	if reg_name == 'cb':
		classifier = CatBoostRegressor()
	return (classifier)

#分类模型
def classify(cla_name):
	if cla_name == 'knn':
		classifier = KNeighborsClassifier()
	if cla_name == 'mlp':
		classifier = MLPClassifier(random_state=1)
	if cla_name == 'svr':
		classifier = SVC(probability=True)
	if cla_name == 'rf':
		classifier = RandomForestClassifier(random_state=1)
	if cla_name == 'gb':
		classifier = GradientBoostingClassifier(random_state=1)
	if cla_name == 'xgb':
		classifier = XGBClassifier(seed=1)
	if cla_name == 'lgb':
		classifier = LGBMClassifier()
	if cla_name == 'cb':
		classifier = CatBoostClassifier()
	return (classifier)


def tuning_knn(X, y, rs=10, cvn=5, r2='r2', njobs=-1):
	X = X
	y = y
	k_range = [5,10,20,40]
	leaf_range = [5,10,30,50]
	weight_options = ['uniform','distance']
	algorithm_options = ['auto']
	param_test = dict(n_neighbors = k_range, weights = weight_options, algorithm=algorithm_options, leaf_size=leaf_range)
	
	#0. baseline model
	print ("\n#============KNN============\n##0. baseline model:\n")
	gbm0 = KNeighborsRegressor()
	modelfit(gbm0, X, y, printFeatureImportance=False,cv_folds=cvn)
	
	#1. gridSearch
	print ("\n##1. gridSearch:\n")
	gsearch1 = RandomizedSearchCV(estimator = KNeighborsRegressor(), param_distributions = param_test, scoring=r2,n_jobs=njobs, cv=cvn, n_iter=15)
	gsearch1.fit(X,y)
	(best_params, best_params_score) = modelSearch(gsearch1)
	
	print ("\nBEST SCORE: %s\nBEST PARA:  KNeighborsRegressor( n_neighbors = %s, weights = \'%s\', algorithm = \'%s\', leaf_size = %s )\n" % (best_params_score, best_params['n_neighbors'], best_params['weights'], best_params['algorithm'], best_params['leaf_size']))


def tuning_mlp(X, y, rs=10, cvn=5, r2='r2',  njobs=-1):
	X = X
	y = y
	param_test = {'hidden_layer_sizes':[10,50,100,500], 'activation':['relu','logistic']}
	
	#0. baseline model
	print ("\n#============MLP============\n##0. baseline model:\n")
	gbm0 = MLPRegressor(random_state=rs)
	modelfit(gbm0, X, y, printFeatureImportance=False,cv_folds=cvn)
	
	#1. gridSearch
	print ("\n##1. gridSearch:\n")
	gsearch1 = GridSearchCV(estimator = MLPRegressor(random_state=rs), param_grid = param_test, scoring=r2, n_jobs=njobs, cv=cvn)
	gsearch1.fit(X,y)
	(best_params, best_params_score) = modelSearch(gsearch1)
	
	print ("\nBEST SCORE: %s\nBEST PARA:  MLPRegressor( hidden_layer_sizes = %s, activation = \'%s\', random_state = %s )\n" % (best_params_score, best_params['hidden_layer_sizes'], best_params['activation'],rs))
	
	
def tuning_svr(X, y, rs=10, cvn=5, r2='r2', njobs=-1):
	X = X
	y = y
	param_test = {'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto'], 'C': [0.05,0.5,1,5,20]}
	
	#0. baseline model
	print ("\n#============SVR============\n##0. baseline model:\n")
	gbm0 = SVR()
	modelfit(gbm0, X, y, printFeatureImportance=False,cv_folds=cvn)
	
	#1. gridSearch
	print ("\n##1. gridSearch:\n")
	gsearch1 = GridSearchCV(estimator = SVR(), param_grid = param_test, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch1.fit(X,y)
	(best_params, best_params_score) = modelSearch(gsearch1)
	
	if best_params['kernel'] == 'rbf':
		print ("\nBEST SCORE: %s\nBEST PARA:  SVR( kernel = \'%s\', C = %s, gamma =\'%s\' )\n" % (best_params_score, best_params['kernel'], best_params['C'], best_params['gamma']))
	else:
		print ("\nBEST SCORE: %s\nBEST PARA:  SVR( kernel = \'%s\', C = %s )\n" % (best_params_score, best_params['kernel'], best_params['C']))

def tuning_rf(X, y, rs=10, cvn=5, r2='r2', njobs=-1):
	X = X
	y = y
	#parameters tests
	n_features=np.shape(X)[1]
	param_test1 = {'n_estimators':[10,50,100,500,1000]}
	param_test2 = {'max_depth':[None,5,10], 'min_samples_split':[2,10,20,40]}
	param_test3 = {'min_samples_leaf':[1,5,10,20]}
	
	#0. baseline model
	print ("\n#============RF============\n##0. baseline model:\n")
	gbm0 = RandomForestRegressor(random_state=rs)
	modelfit(gbm0, X, y, printFeatureImportance=False,cv_folds=cvn)
	
	#1. gridSearch n_estimators
	print ("\n##1. gridSearch n_estimators:\n")
	gsearch1 = GridSearchCV(estimator = RandomForestRegressor(random_state=rs), param_grid = param_test1, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch1.fit(X,y)
	(best_n_estimators, best_n_estimators_score) = modelSearch(gsearch1)
	
	#2. gridSearch max_depth
	print ("\n##2. gridSearch max_depth:\n")
	gsearch2 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators['n_estimators'],random_state=rs), param_grid = param_test2, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch2.fit(X,y)
	(best_max_depth, best_max_depth_score) = modelSearch(gsearch2)
	
	#3. gridSearch min_samples_leaf
	print ("\n##3. gridSearch min_samples_leaf:\n")
	gsearch3 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],min_samples_split=best_max_depth['min_samples_split'],random_state=rs), param_grid = param_test3, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch3.fit(X,y)
	(best_min_samples_leaf, best_min_samples_leaf_score) = modelSearch(gsearch3)
			

	print ("\nBEST SCORE: %s\nBEST PARA:  RandomForestRegressor( n_estimators = %s, max_depth = %s, min_samples_split = %s, min_samples_leaf = %s, random_state = %s )\n" % (best_min_samples_leaf_score, best_n_estimators['n_estimators'], best_max_depth['max_depth'], best_max_depth['min_samples_split'], best_min_samples_leaf['min_samples_leaf'], rs))


def tuning_gb(X, y, rs=10, cvn=5, r2='r2', njobs=-1):
	X = X
	y = y
	
	n_features=np.shape(X)[1]
	#parameters tests
	param_test1 = {'n_estimators':[10,50,100,500,1000]}
	param_test2 = {'max_depth':[3,10,15,30,50], 'min_samples_split':[2,10,20,40]}
	param_test3 = {'min_samples_leaf':[1,5,10,20]}
	param_test5= {'subsample':[0.8,1.0]}
	param_test6a= {'learning_rate':[0.01,0.05,0.1,0.2,0.5]}


	#0. baseline model
	print ("\n#============GB============\n##0. baseline model:\n")
	gbm0 = GradientBoostingRegressor(random_state=rs)
	modelfit(gbm0, X, y, printFeatureImportance=False,cv_folds=cvn)
	
	#1. gridSearch n_estimators
	print ("\n##1. gridSearch n_estimators:\n")
	gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(random_state=rs), param_grid = param_test1, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch1.fit(X,y)
	(best_n_estimators, best_n_estimators_score) = modelSearch(gsearch1)
	
	#2. gridSearch max_depth
	print ("\n##2. gridSearch max_depth:\n")
	gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=best_n_estimators['n_estimators'],random_state=rs), param_grid = param_test2, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch2.fit(X,y)
	(best_max_depth, best_max_depth_score) = modelSearch(gsearch2)
	
	#3. gridSearch min_samples_leaf
	print ("\n##3. gridSearch min_samples_leaf:\n")
	gsearch3 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],min_samples_split=best_max_depth['min_samples_split'],random_state=rs), param_grid = param_test3, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch3.fit(X,y)
	(best_min_samples_leaf, best_min_samples_leaf_score) = modelSearch(gsearch3)
	
	#5. gridSearch subsample
	print ("\n##5. gridSearch subsample:\n")
	gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],min_samples_split=best_max_depth['min_samples_split'], min_samples_leaf=best_min_samples_leaf['min_samples_leaf'], random_state=rs), param_grid = param_test5, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch5.fit(X,y)
	(best_subsample, best_subsample_score) = modelSearch(gsearch5)

	#6. ajust learning rate
	print ("\n##6. ajust learning rate:\n")
	gsearch6 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=int(best_n_estimators['n_estimators'])*2,max_depth=best_max_depth['max_depth'],min_samples_split=best_max_depth['min_samples_split'], min_samples_leaf=best_min_samples_leaf['min_samples_leaf'],subsample=best_subsample['subsample'],random_state=rs), param_grid = param_test6a, scoring=r2,n_jobs=njobs, cv=cvn)		
	gsearch6.fit(X,y)
	(learning_rate_a, learning_rate_score_a) = modelSearch(gsearch6)

	
	#chose final model
	best_learning_rate=0.1
	best_learning_rate_score=best_subsample_score
	final_n_estimators=best_n_estimators['n_estimators']
	if (learning_rate_score_a > best_learning_rate_score):
		best_learning_rate = learning_rate_a['learning_rate']
		best_learning_rate_score = learning_rate_score_a
		final_n_estimators = int(best_n_estimators['n_estimators'])*2

	print ("\nBEST SCORE: %s\nBEST PARA:  GradientBoostingRegressor( learning_rate = %s, n_estimators = %s, max_depth = %s, min_samples_split = %s, min_samples_leaf = %s, subsample = %s, random_state = %s )\n" % (best_learning_rate_score, best_learning_rate, final_n_estimators, best_max_depth['max_depth'], best_max_depth['min_samples_split'], best_min_samples_leaf['min_samples_leaf'], best_subsample['subsample'],rs))


def tuning_xgb(X, y, rs=10, cvn=5, r2='r2', njobs=1):
	X = X
	y = y
	#parameters tests
	param_test1 = {'n_estimators':[50,100,200,500]}
	param_test2 = {'max_depth':[3,5,10], 'min_child_weight':[1,2]}
	param_test3 = {'gamma':[0,0.05,0.1]}
	param_test4 = {'subsample':[0.8,1.0],'colsample_bytree':[0.8,1.0]}
	param_test5 = {'reg_alpha':[0, 0.01]}
	param_test6 = {'reg_lambda':[0, 0.01]}
	param_test7a = {'learning_rate':[0.05]}

	
	#0. baseline model
	print ("\n#============XGB============\n##0. baseline model:\n")
	xgb0 = XGBRegressor(seed=rs)
	modelfit(xgb0, X, y, printFeatureImportance=False,cv_folds=cvn)
	
	
	#1. gridSearch n_estimators
	gsearch1 = GridSearchCV(estimator = XGBRegressor(seed=rs), param_grid = param_test1, scoring=r2, n_jobs=njobs,  cv=cvn)
	gsearch1.fit(X,y)
	(best_n_estimators, best_n_estimators_score) = modelSearch(gsearch1)
	
	#2. gridSearch max_depth
	print ("\n##2. gridSearch max_depth:\n")
	gsearch2 = GridSearchCV(estimator = XGBRegressor(n_estimators=best_n_estimators['n_estimators'],seed=rs), param_grid = param_test2, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch2.fit(X,y)
	(best_max_depth, best_max_depth_score) = modelSearch(gsearch2)
	
	#3. gridSearch gamma
	print ("\n##3. gridSearch gamma:\n")
	gsearch3 = GridSearchCV(estimator = XGBRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],min_child_weight=best_max_depth['min_child_weight'],seed=rs), param_grid = param_test3, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch3.fit(X,y)
	(best_gamma, best_gamma_score) = modelSearch(gsearch3)
	
	#4. gridSearch subsample
	print ("\n##4. gridSearch subsample:\n")
	gsearch4 = GridSearchCV(estimator = XGBRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],min_child_weight=best_max_depth['min_child_weight'], gamma=best_gamma['gamma'],seed=rs), param_grid = param_test4, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch4.fit(X,y)
	(best_subsample, best_subsample_score) = modelSearch(gsearch4)
	
	#5. gridSearch reg_alpha
	print ("\n##5. gridSearch reg_alpha:\n")
	gsearch5 = GridSearchCV(estimator = XGBRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],min_child_weight=best_max_depth['min_child_weight'], gamma=best_gamma['gamma'], subsample=best_subsample['subsample'], seed=rs), param_grid = param_test5, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch5.fit(X,y)
	(best_reg_alpha, best_reg_alpha_score) = modelSearch(gsearch5)
	
	#6. gridSearch reg_lambda
	print ("\n##5. gridSearch reg_lambda:\n")
	gsearch6 = GridSearchCV(estimator = XGBRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],min_child_weight=best_max_depth['min_child_weight'], gamma=best_gamma['gamma'], subsample=best_subsample['subsample'], reg_alpha=best_reg_alpha['reg_alpha'], seed=rs), param_grid = param_test6, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch6.fit(X,y)
	(best_reg_lambda, best_reg_lambda_score) = modelSearch(gsearch6)
	
	
	#7. ajust learning rate
	print ("\n##7. ajust learning rate:\n")
	gsearch7 = GridSearchCV(estimator = XGBRegressor(n_estimators=int(best_n_estimators['n_estimators'])*2,max_depth=best_max_depth['max_depth'],min_child_weight=best_max_depth['min_child_weight'], gamma=best_gamma['gamma'], subsample=best_subsample['subsample'], reg_alpha=best_reg_alpha['reg_alpha'], reg_lambda=best_reg_lambda['reg_lambda'], seed=rs), param_grid = param_test7a, scoring=r2,n_jobs=njobs, cv=cvn)		
	gsearch7.fit(X,y)
	(learning_rate_a, learning_rate_score_a) = modelSearch(gsearch7)

	
	#chose final model
	best_learning_rate=0.1
	best_learning_rate_score=best_reg_lambda_score
	final_n_estimators=best_n_estimators['n_estimators']
	if (learning_rate_score_a > best_learning_rate_score):
		best_learning_rate = learning_rate_a['learning_rate']
		best_learning_rate_score = learning_rate_score_a
		final_n_estimators = int(best_n_estimators['n_estimators'])*2

	print ("\nBEST SCORE: %s\nBEST PARA:  XGBRegressor( learning_rate = %s, n_estimators = %s, max_depth = %s, min_child_weight = %s, gamma = %s, subsample = %s, reg_alpha = %s, reg_lambda = %s, seed  = %s )\n" % (best_learning_rate_score, best_learning_rate, final_n_estimators, best_max_depth['max_depth'], best_max_depth['min_child_weight'], best_gamma['gamma'], best_subsample['subsample'], best_reg_alpha['reg_alpha'], best_reg_lambda['reg_lambda'],rs))

	
def tuning_lgb(X, y, rs=10, cvn=5, r2='r2', njobs=1):
	X = X
	y = y
	#parameters tests
	param_test1 = {'n_estimators':[10,50,100,500,1000]}
	param_test2 = {'max_depth':[-1,5,10], 'num_leaves':[2, 5, 21, 31, 51, 100]}
	param_test3 = {'min_child_samples': [1, 5, 10, 20, 40], 'min_child_weight':[0.001]}
	param_test4 = {'feature_fraction': [0.8, 1.0], 'bagging_fraction': [0.8, 1.0]}
	param_test5 = {'reg_alpha': [0, 0.01],'reg_lambda': [0, 0.01]}
	param_test7a = {'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.5]}

	
	#0. baseline model
	print ("\n#============LGB============\n##0. baseline model:\n")
	lgb0 = LGBMRegressor(seed=rs)
	modelfit(lgb0, X, y, printFeatureImportance=False,cv_folds=cvn)
	
	#1. gridSearch n_estimators
	gsearch1 = GridSearchCV(estimator = LGBMRegressor(seed=rs), param_grid = param_test1, scoring=r2, n_jobs=njobs,  cv=cvn)
	gsearch1.fit(X,y)
	(best_n_estimators, best_n_estimators_score) = modelSearch(gsearch1)
	
	#2. gridSearch max_depth
	print ("\n##2. gridSearch max_depth:\n")
	gsearch2 = GridSearchCV(estimator = LGBMRegressor(n_estimators=best_n_estimators['n_estimators'],seed=rs), param_grid = param_test2, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch2.fit(X,y)
	(best_max_depth, best_max_depth_score) = modelSearch(gsearch2)
	
	#3. gridSearch min_child
	print ("\n##5. gridSearch min_child:\n")
	gsearch3 = GridSearchCV(estimator = LGBMRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],num_leaves=best_max_depth['num_leaves'], seed=rs), param_grid = param_test3, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch3.fit(X,y)
	(best_min_child, best_min_child_score) = modelSearch(gsearch3)
	
	#4. gridSearch feature_fraction
	print ("\n##5. gridSearch feature_fraction:\n")
	gsearch4 = GridSearchCV(estimator = LGBMRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],num_leaves=best_max_depth['num_leaves'], min_child_samples=best_min_child['min_child_samples'],min_child_weight=best_min_child['min_child_weight'], seed=rs), param_grid = param_test4, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch4.fit(X,y)
	(best_feature_fraction, best_feature_fraction_score) = modelSearch(gsearch4)

	#5. gridSearch reg_alpha
	print ("\n##5. gridSearch reg_alpha:\n")
	gsearch5 = GridSearchCV(estimator = LGBMRegressor(n_estimators=best_n_estimators['n_estimators'],max_depth=best_max_depth['max_depth'],num_leaves=best_max_depth['num_leaves'], min_child_samples=best_min_child['min_child_samples'],min_child_weight=best_min_child['min_child_weight'], feature_fraction=best_feature_fraction['feature_fraction'],bagging_fraction=best_feature_fraction['bagging_fraction'], seed=rs), param_grid = param_test5, scoring=r2,n_jobs=njobs, cv=cvn)
	gsearch5.fit(X,y)
	(best_reg_alpha, best_reg_alpha_score) = modelSearch(gsearch5)

	#7. ajust learning rate
	print ("\n##7. ajust learning rate:\n")
	gsearch7 = GridSearchCV(estimator = LGBMRegressor(n_estimators=int(best_n_estimators['n_estimators'])*2,max_depth=best_max_depth['max_depth'],num_leaves=best_max_depth['num_leaves'], min_child_samples=best_min_child['min_child_samples'],min_child_weight=best_min_child['min_child_weight'], feature_fraction=best_feature_fraction['feature_fraction'],bagging_fraction=best_feature_fraction['bagging_fraction'],reg_alpha=best_reg_alpha['reg_alpha'], reg_lambda=best_reg_alpha['reg_lambda'], seed=rs), param_grid = param_test7a, scoring=r2,n_jobs=njobs, cv=cvn)		
	gsearch7.fit(X,y)
	(learning_rate_a, learning_rate_score_a) = modelSearch(gsearch7)

	
	#chose final model
	best_learning_rate=0.1
	best_learning_rate_score=best_reg_alpha_score
	final_n_estimators=best_n_estimators['n_estimators']
	if (learning_rate_score_a > best_learning_rate_score):
		best_learning_rate = learning_rate_a['learning_rate']
		best_learning_rate_score = learning_rate_score_a
		final_n_estimators = int(best_n_estimators['n_estimators'])*2

	print ("\nBEST SCORE: %s\nBEST PARA:  LGBMRegressor( learning_rate = %s, n_estimators = %s, max_depth = %s, num_leaves = %s, min_child_samples = %s, min_child_weight = %s, feature_fraction = %s, bagging_fraction = %s, reg_alpha = %s, reg_lambda = %s, seed  = %s )\n" % (best_learning_rate_score, best_learning_rate, final_n_estimators, best_max_depth['max_depth'], best_max_depth['num_leaves'], best_min_child['min_child_samples'], best_min_child['min_child_weight'],best_feature_fraction['feature_fraction'], best_feature_fraction['bagging_fraction'], best_reg_alpha['reg_alpha'], best_reg_alpha['reg_lambda'],rs))
	
def tuning_cb(X, y, rs=10, cvn=5, r2='r2', njobs=1):
	X = X
	y = y
	#parameters tests
	param_test1 = {'depth': [3,5,10], 'l2_leaf_reg': [3,4], 'iterations': [50,100,200,500], 'learning_rate' : [0.1,0.05]}
	#0. baseline model
	
	print ("\n#============CB============\n##0. baseline model:\n")
	cb0 = CatBoostRegressor()
	modelfit(cb0, X, y, printFeatureImportance=False,cv_folds=cvn)
	
	#1. gridSearch 
	gsearch1 = GridSearchCV(estimator = CatBoostRegressor(), param_grid = param_test1, scoring=r2, n_jobs=njobs,  cv=cvn)
	gsearch1.fit(X,y)
	(best_params, best_params_score) = modelSearch(gsearch1)
	print ("\nBEST SCORE: %s\nBEST PARA:  CatBoostRegressor( learning_rate = %s, iterations = %s, depth = %s, l2_leaf_reg = %s)\n" % (best_params_score, best_params['learning_rate'], best_params['iterations'], best_params['depth'], best_params['l2_leaf_reg']))


def main():

	
	#初始化参数
	argv = init()
	training_data=argv.training
	test_data=argv.test
	snpinfo=argv.snpinfo
	output_prefix=argv.output
	method=argv.method
	name=method
		
	#性状列号,默认第一列
	n=int("1")
	if argv.n:
		n = int(argv.n)

	#回归还是分类问题
	type='reg'
	if argv.type:
		type = argv.type


	############
	# 数据准备 #
	############

	if argv.type:
		# read data files for reg and cla
		if type == 'cla' or type == 'reg':
			#读取训练数据
			(X1, y1, trait1, sample1, y1_raw) = readInputFiles2(snpinfo, training_data, n)
			if method == 'rbknf' or method == 'rbkaf':
				(Xk, yk, traitk, samplek, yk_raw) = (copy.deepcopy(X1), copy.deepcopy(y1), trait1, copy.deepcopy(sample1), copy.deepcopy(y1_raw))

			#generate test data when defined
			if argv.testlist or argv.test:
				if argv.testlist:
					(X2, y2, trait2, sample2, y2_raw) = (copy.deepcopy(X1), copy.deepcopy(y1), trait1, copy.deepcopy(sample1), copy.deepcopy(y1_raw))
					sample = readfile2list(argv.testlist)
					maskBool = maskTest(sample1, sample, invert=False)
					X2 = X2[maskBool]
					y2 = y2[maskBool]
					y2_raw = y2_raw[maskBool]
				else:
					#读取测试数据
					(X2, y2, trait2, sample, y2_raw) = readInputFiles2(snpinfo, test_data, n)
				
			#extract training data when defined
			if argv.trainlist:
				sampleT = readfile2list(argv.trainlist)
				maskBool = maskTest(sample1, sampleT, invert=False)
				sample1 = list(np.array(sample1)[maskBool])
				X1 = X1[maskBool]
				y1 = y1[maskBool]
				y1_raw = y1_raw[maskBool]
			
			#mask TestSamples from Training data
			if argv.mask == 'yes':
				maskBool = maskTest(sample1, sample, invert=True)
				X1 = X1[maskBool]
				y1 = y1[maskBool]
				y1_raw = y1_raw[maskBool]

		if argv.testlist or argv.test:
			#转换特征编码方式onehot编码
			if argv.onehot == 'yes':
				C = np.concatenate((X1, X2),axis=0)
				enc = OneHotEncoder()
				C = enc.fit_transform(C).toarray()
				(X1,X2) = np.split(C, [X1.shape[0]])
			#归一化
			if argv.norm == 'yes':
				(X1, t1, t2 ) = normalizeT(X1)
				X2 = normalizeV(X2, t1, t2)
		#when no test data provied
		else:
			if argv.onehot == 'yes':
				enc = OneHotEncoder()
				X1 = enc.fit_transform(X1).toarray()
			#归一化0-1
			if argv.norm == 'yes':
				(X1, t1, t2 ) = normalizeT(X1)

	
################################
######~~~~~ 回归问题 ~~~~~###### 
################################ 
	if type == 'reg':
		############
		# 模型调参 #
		############
		if argv.assess == 'yes':
			cv=5
			if argv.cv:
				cv=int(argv.cv)

			if method == 'knn':
				tuning_knn(X1, y1, rs=1, cvn=cv, r2='r2', njobs=1)
			if method == 'mlp':
				tuning_mlp(X1, y1, rs=1, cvn=cv, r2='r2', njobs=1)
			if method == 'svr':
				tuning_svr(X1, y1, rs=1, cvn=cv, r2='r2', njobs=1)
			if method == 'rf':
				tuning_rf(X1, y1, rs=1, cvn=cv, r2='r2', njobs=1)
			if method == 'gb':
				tuning_gb(X1, y1, rs=1, cvn=cv, r2='r2', njobs=1)
			if method == 'xgb':
				tuning_xgb(X1, y1, rs=1, cvn=cv, r2='r2', njobs=1)
			if method == 'lgb':
				tuning_lgb(X1, y1, rs=1, cvn=cv, r2='r2', njobs=1)
			if method == 'cb':
				tuning_cb(X1, y1, rs=1, cvn=cv, r2='r2', njobs=1)


		############
		# 模型预测 #
		############
		else:
			
			#加载默认模型
			knn = regression('knn')
			mlp = regression('mlp')
			svr = regression('svr')
			rf = regression('rf')
			gb = regression('gb')
			xgb = regression('xgb')
			lgb = regression('lgb')
			cb = regression('cb')
			
			#如果提供模型参数，重载模型

			if argv.para:
				f = open(argv.para,'r')
				para = ''
				for line in f.readlines():
					arr = line.strip().split(':')
					if arr[0] == 'BEST PARA':
						para = arr[1]
				f.close()
				f = open('my_reg.py','w')
				f.write('''
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, LassoLars, BayesianRidge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
def my_reg(reg_name):
	classifier = %s
	return (classifier)
''' % (para))
				f.close()
				from my_reg import my_reg
				mypara = my_reg('mypara')
				
			
			#define a model to predict
			my_each = []
			my_name = []
			if method == 'knn':
				my_each = [knn]
				my_name = ['knn']
			if method == 'mlp':
				my_each = [mlp]
				my_name = ['mlp']
			if method == 'svr':
				my_each = [svr]
				my_name = ['svr']
			if method == 'rf':
				my_each = [rf]
				my_name = ['rf']
			if method == 'gb':
				my_each = [gb]
				my_name = ['gb']
			if method == 'xgb':
				my_each = [xgb]
				my_name = ['xgb']
			if method == 'lgb':
				my_each = [lgb]
				my_name = ['lgb']
			if method == 'cb':
				my_each = [cb]
				my_name = ['cb']
			if method == 'mypara':
				my_each = [mypara]
				my_name = ['mypara']

			
			if my_name:
				for (each,name) in zip(my_each, my_name):
					eachY = y1
					if argv.model:
						#加载模型
						clf = joblib.load(argv.model + '_' + name + '.pkl')
						#预测结果
						my_eachY = clf.predict(X2)
					else:	
						#保存模型
						joblib.dump(each.fit(X1,eachY), output_prefix + '_' + name + '.pkl')
						#加载模型
						clf = joblib.load(output_prefix + '_' + name + '.pkl')
						#预测结果
						my_eachY = clf.predict(X2)
	
			
			# rrblup models
			if method == 'rbsnf' or method == 'rbsaf' or method == 'rbknf' or method == 'rbkaf':
				argv.norm = 'no'
				snp_no_fix="""
					function(x1,x2,y1){
					  library(rrBLUP)
					  trainG <- x1
					  testG <- x2
					  ytrain <- as.numeric(y1)
					  phen_answer<-mixed.solve(ytrain, Z=trainG, K=NULL, SE = FALSE, return.Hinv=FALSE)
					  beta <- phen_answer$beta
					  phD <- phen_answer$u
					  e <- as.matrix(phD)
					  res <- list(beta = beta, e = e, phD = phD)
					  ypred <- testG %*% res$e
					  ypred <- ypred[,1] + as.numeric(res$beta)
					  ypred
					}
				"""
				snp_and_fix="""
					function(x1,x2,y1,fix1,fix2){
					  library(rrBLUP)
					  trainG <- x1
					  testG <- x2
					  ytrain <- as.numeric(y1)
					  trainfix <- fix1
					  testfix <- fix2
					  phen_answer<-mixed.solve(ytrain, Z=trainG, K=NULL, SE = FALSE, return.Hinv=FALSE, X = trainfix)
					  beta <- phen_answer$beta
					  phD <- phen_answer$u
					  e <- as.matrix(phD)
					  res <- list(beta = beta, e = e, phD = phD)
					  ypred <- testG %*% res$e
					  beta <- matrix(res$beta,nrow = 2)
					  beta <- testfix %*% beta
					  ypred <- ypred[,1] + beta
					  ypred
					}
				"""
				
				kinship_no_fix="""
					function(x,yk,idx2){
					  library(sommer)
					  amat <- x
					  y <- yk
					  y[idx2] <- NA
					  y <- as.numeric(y)
					  rownames(amat) <- seq(1,dim(amat)[1])
					  ETA <- list(A = list(Z = diag(length(y)),K = amat))
					  ans <- mmer(y,Z = ETA,silent = T)
					  ypred <- ans$fitted.y[idx2]
					  ypred
					}
				"""

				kinship_and_fix="""
					function(x,yk,fix,idx2){
					  library(sommer)
					  amat <- x
					  y <- yk
					  y[idx2] <- NA
					  y <- as.numeric(y)
					  rownames(amat) <- seq(1,dim(amat)[1])
					  ETA <- list(A = list(Z = diag(length(y)),K = amat))
					  ans <- mmer(y,Z = ETA,X = fix,silent = T)
					  ypred <- ans$fitted.y[idx2]
					  ypred
					}
				"""			
				
				
				if method == 'rbsnf':
					name = 'rbsnf'
					function = robjects.r(snp_no_fix)
					x1_r, x1_c = X1.shape
					x2_r, x2_c = X2.shape
					rOut = function(robjects.r.matrix(X1, nrow=x1_r, ncol=x1_c), robjects.r.matrix(X2, nrow=x2_r, ncol=x2_c), robjects.FloatVector(y1))
					with localconverter(ro.default_converter + pandas2ri.converter):
						my_eachY = ro.conversion.rpy2py(rOut)
			
				if method == 'rbsaf':
					name = 'rbsaf'
					fixidx = np.array(readfile2list(argv.fixidx),dtype=np.int)
					featureidx = list(set(list(range(0,X1.shape[1])))^set(fixidx))
					trainG = X1[:,featureidx]
					trainG_r,trainG_c = trainG.shape
					testG = X2[:,featureidx]
					testG_r,testG_c = testG.shape
					trainfix = X1[:,fixidx]
					trainfix_r, trainfix_c = trainfix.shape
					testfix = X2[:,fixidx]
					testfix_r, testfix_c = testfix.shape
					function = robjects.r(snp_and_fix)
					rOut = function(robjects.r.matrix(trainG, nrow=trainG_r, ncol=trainG_c), robjects.r.matrix(testG, nrow=testG_r, ncol=testG_c), robjects.FloatVector(y1), robjects.r.matrix(trainfix, nrow=trainfix_r, ncol=trainfix_c), robjects.r.matrix(testfix, nrow=testfix_r, ncol=testfix_c))
					with localconverter(ro.default_converter + pandas2ri.converter):
						my_eachY = ro.conversion.rpy2py(rOut).reshape(-1,)
			
				if method == 'rbknf':
					name = 'rbknf'
					maskBool = maskTest(samplek, sample, invert=False)
					idx = np.where(maskBool == True)
					idx2 = idx[0]+1
					function = robjects.r(kinship_no_fix)
					Xk_r, Xk_c = Xk.shape
					rOut = function(robjects.r.matrix(Xk, nrow=Xk_r, ncol=Xk_c), robjects.FloatVector(yk), robjects.FloatVector(idx2))
					with localconverter(ro.default_converter + pandas2ri.converter):
						my_eachY = ro.conversion.rpy2py(rOut)

				if method == 'rbkaf':
					name = 'rbkaf'
					fixidx = np.array(readfile2list(argv.fixidx),dtype=np.int)
					featureidx = list(set(list(range(0,Xk.shape[1])))^set(fixidx))
					X = Xk[:,featureidx]
					x_r, x_c = X.shape
					fix = Xk[:,fixidx]
					fix_r, fix_c = fix.shape
					maskBool = maskTest(samplek, sample, invert=False)
					idx = np.where(maskBool == True)
					idx2 = idx[0]+1
					function = robjects.r(kinship_and_fix)
					rOut = function(robjects.r.matrix(X, nrow=x_r, ncol=x_c), robjects.FloatVector(yk), robjects.r.matrix(fix, nrow=fix_r, ncol=fix_c), robjects.FloatVector(idx2))
					with localconverter(ro.default_converter + pandas2ri.converter):
						my_eachY = ro.conversion.rpy2py(rOut)
			

				
			# output prediction results
			f=open(output_prefix + '_' + name + '.preditc_result.txt','w')
			k="\t".join([str(each) for each in sample])
			k='sample' + "\t" + k
			f.write(k+"\n")
			
			k="\t".join([str(each) for each in my_eachY])
			k = trait2 + "\t" + k
			f.write(k+"\n")
			f.close()
			
			y2_idx = [i for i,x in enumerate(y2_raw) if x!='NA']
			y2 = [y2[i] for i in y2_idx]
			my_eachY = [my_eachY[i] for i in y2_idx]
			
			(cor, r2s, mse, rmse, top10overlap, top20overlap, top30overlap) = evaluation(y2, my_eachY, sample, name)
			if cor != "NA":
				plotCor(y2, my_eachY, name + ': cor = ' + str(cor) , output_prefix + '_' + name + '.cor.png')

################################
######~~~~~ 分类问题 ~~~~~###### 
################################ 
	if type == 'cla':
		############
		# 模型评估 #
		############
		# performe CV evaluation in training data
		
		if argv.assess == 'yes':
			e = int("2")
			if argv.evaluate:
				e = int(argv.evaluate)
				
			cv = StratifiedShuffleSplit(n_splits=5)
			num = int("1")
			colors = cycle(["red"])
			colors2 = cycle(["seashell"])
			labels = cycle(["AUC"])
			if e == 3:
				num = int("3")
				colors = cycle(["red", "green", "blue"])
				colors2 = cycle(["seashell", "honeydew", "aliceblue"])
				labels = cycle(["Top","Mid","Low"])
				
			
			knn = classify('knn')
			mlp = classify('mlp')
			svr = classify('svr')
			rf = classify('rf')
			gb = classify('gb')
			xgb = classify('xgb')
			lgb = classify('lgb')
			cb = classify('cb')
			
			my_each = []
			my_name = []
			if method == 'knn':
				my_each = [knn]
				my_name = ['knn']
			if method == 'mlp':
				my_each = [mlp]
				my_name = ['mlp']
			if method == 'svr':
				my_each = [svr]
				my_name = ['svr']
			if method == 'rf':
				my_each = [rf]
				my_name = ['rf']
			if method == 'gb':
				my_each = [gb]
				my_name = ['gb']
			if method == 'xgb':
				my_each = [xgb]
				my_name = ['xgb']
			if method == 'lgb':
				my_each = [lgb]
				my_name = ['lgb']
			if method == 'cb':
				my_each = [cb]
				my_name = ['cb']

			for (each,name) in zip(my_each, my_name):
				fig = plt.figure(figsize=(4,4),dpi=100)
				mean_tpr = 0.0
				mean_fpr = np.linspace(0, 1, 100)	
				lw = 1
				plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='grey')
				
				if e == 2:
					y1 = np.ndarray.tolist(y1)
					y1 = [1 if int(x) == 2 else x for x in y1]
					y1 = np.array(y1,"float64")
				
				for n, mycolor, mycolor2, mylabel in zip(range(num), colors, colors2, labels):
					for (train, test) in cv.split(X1, y1):
						probas_ = each.fit(X1[train], y1[train]).predict_proba(X1[test])
						newy = np_utils.to_categorical(y1[test])
						fpr, tpr, thresholds = roc_curve(newy[:,n], probas_[:, n])
						mean_tpr += interp(mean_fpr, fpr, tpr)
						mean_tpr[0] = 0.0
						roc_auc = auc(fpr, tpr)
						plt.plot(fpr, tpr, lw=0.5, color=mycolor2, linestyle='--', zorder=1)
					mean_tpr /= cv.get_n_splits(X1, y1)
					mean_tpr[-1] = 1.0
					mean_auc = auc(mean_fpr, mean_tpr)
					plt.plot(mean_fpr, mean_tpr, color=mycolor, label='%s (%0.3f)' % (mylabel, mean_auc), lw=lw, zorder=2)
					mean_tpr = 0.0
				plt.xlim([-0.05, 1.05])
				plt.ylim([-0.05, 1.05])
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('ROC ('+name+')')
				plt.legend(loc="lower right")
				plt.show()
				fig.savefig(output_prefix+'_ROC.pdf')

		
		############
		# 模型预测 #
		############
		else:
			
			#加载默认模型
			knn = classify('knn')
			mlp = classify('mlp')
			svr = classify('svr')
			rf = classify('rf')
			gb = classify('gb')
			xgb = classify('xgb')
			lgb = classify('lgb')
			cb = classify('cb')
			
			#如果提供模型参数，重载模型
			if argv.para:
				f = open(argv.para,'r')
				para = ''
				for line in f.readlines():
					arr = line.strip().split(':')
					if arr[0] == 'BEST PARA':
						para = arr[1]
				f.close()
				f = open('my_reg.py','w')
				f.write('''
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, LassoLars, BayesianRidge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import BaggingRegressor,ExtraTreesRegressor,AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
def my_reg(reg_name):
	classifier = %s
	return (classifier)
''' % (para))
				f.close()
				from my_reg import my_reg
				mypara = my_reg('mypara')
			
			#define a model to predict
			my_each = []
			my_name = []
			if method == 'knn':
				my_each = [knn]
				my_name = ['knn']
			if method == 'mlp':
				my_each = [mlp]
				my_name = ['mlp']
			if method == 'svr':
				my_each = [svr]
				my_name = ['svr']
			if method == 'rf':
				my_each = [rf]
				my_name = ['rf']
			if method == 'gb':
				my_each = [gb]
				my_name = ['gb']
			if method == 'xgb':
				my_each = [xgb]
				my_name = ['xgb']
			if method == 'lgb':
				my_each = [lgb]
				my_name = ['lgb']
			if method == 'cb':
				my_each = [cb]
				my_name = ['cb']

			else:
				for (each,name) in zip(my_each, my_name):
					eachY = y1
					if argv.model:
						#加载模型
						clf = joblib.load(argv.model + '_' + name + '.pkl')
						#预测结果
						my_eachY = clf.predict(X2)
						my_eachY_prob = clf.predict_proba(X2)
					else:	
						#保存模型
						joblib.dump(each.fit(X1,eachY), output_prefix + '_' + name + '.pkl')
						#加载模型
						clf = joblib.load(output_prefix + '_' + name + '.pkl')
						#预测结果
						my_eachY = clf.predict(X2)
						my_eachY_prob = clf.predict_proba(X2)
				
			# output prediction results
			f=open(output_prefix + '_' + name + '.preditc_result.txt','w')
			k="\t".join([str(each) for each in sample])
			k='sample' + "\t" + k
			f.write(k+"\n")
			
			k="\t".join([str(int(each)) for each in my_eachY])
			k = trait2 + "\t" + k
			f.write(k+"\n")
			
			k="\t".join([str(each) for each in my_eachY_prob])
			k = trait2 + "_prob\t" + k
			f.write(k+"\n")
			f.close()
			
			if argv.evaluate:
				e = int(argv.evaluate)
				if e == 2:
					evaluation2(y2, my_eachY, my_eachY_prob, sample, name, output_prefix)
				if e == 3:
					evaluation3(y2, my_eachY, my_eachY_prob, sample, name, output_prefix)



def init():
	parser = argparse.ArgumentParser(description = 'Test ML and rrblup models. | %s | %s | %s\n' % (__author__, __version__, __date__))
	parser.add_argument('--training', help = 'training datasets',required = True)
	parser.add_argument('--test', help = 'test datasets',required = False)
	parser.add_argument('--testlist', help = 'alternative to the parameter [--test]. extract test data from the training data. keep the same order as input', required = False)
	parser.add_argument('--trainlist', help = 'using the trainlist data as training data', required = False)
	parser.add_argument('--norm', help = '[yes/no] if normalize data ', default = 'yes', required = False)
	parser.add_argument('--n', help = 'trait col number, default=1',required = False)
	parser.add_argument('--snpinfo', help = 'features file',required = True)
	parser.add_argument('--output', help = 'output file prefix', required = True)
	parser.add_argument('--model', help = 'pre-build model files prefix, default: build from training data', required = False)
	parser.add_argument('--onehot', help = '[yes/no] if or not transform features to onehot encode', default = 'no', required = False)
	parser.add_argument('--mask', help = '[yes/no] if mask samples in training datasets from test datasets',default = 'yes', required = False)
	parser.add_argument('--assess', help = '[yes/no] only do model assessment, but not do prediction', default = 'no', required = False)
	parser.add_argument('--cv', help = 'cross-validation number of assess, default is 5', default = '5', required = False)
	parser.add_argument('--method', help = '[knn/mlp/svr/rf/gb/xgb/lgb/cb/rbsnf/rbsaf/rbknf/rbkaf], which method to use (assess and predict)', required = False)
	parser.add_argument('--para', help = 'parameter file, if provided, --method shold be set to mypara', required = False)
	parser.add_argument('--type', help = 'reg/cla', default = 'reg', required = False)
	parser.add_argument('--evaluate', help = 'evaluation of classifier [2/3]', required = False)
	parser.add_argument('--fixidx', help = 'rrBlup fixidx when use fix, from 0',  required = False)
	argv = parser.parse_args()
	return argv


if __name__ == '__main__':
	main()
