# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 22:03:18 2018

@author: surface
"""

import time
import pandas as pd
import os 
import numpy as np
os.getcwd() #get current working directory
os.chdir('C:\\competition\\alimama')#change working directory
#%%
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
#%%
dataset3=pd.read_csv('data/dataset3_add.csv')
dataset2=pd.read_csv('data/dataset2_add.csv')
dataset1=pd.read_csv('data/dataset1_add.csv')
#%%
dataset3=pd.read_csv('data/dataset3.csv')
dataset2=pd.read_csv('data/dataset2.csv')
dataset1=pd.read_csv('data/dataset1.csv')
#%%

dataset2_pos=dataset2[dataset2['is_trade']==1]
dataset2_neg=dataset2[dataset2['is_trade']==0]

dataset2_neg=dataset2_neg.sample(frac=0.8,random_state=0,replace=True)
dataset2=pd.concat([dataset2_pos,dataset2_neg])
#%%
label2=dataset2[['is_trade']]
label1=dataset1[['is_trade']]
dataset3_pre=dataset3[['instance_id']]

dataset3.drop(['instance_id'],axis=1,inplace=True)
dataset1.drop(['is_trade'],axis=1,inplace=True)
dataset2.drop(['is_trade'],axis=1,inplace=True)
dataset1.drop(['instance_id'],axis=1,inplace=True)
dataset2.drop(['instance_id'],axis=1,inplace=True)
#%%
#dataset1_pos=dataset1[dataset1['is_trade']==1]
#dataset1_neg=dataset1[dataset1['is_trade']==0]
#
#dataset1_neg=dataset1_neg.sample(frac=0.8,random_state=0,replace=True)
#dataset1=pd.concat([dataset1_pos,dataset1_neg])
#
#dataset2_pos=dataset2[dataset2['is_trade']==1]
#dataset2_neg=dataset2[dataset2['is_trade']==0]
#
#dataset2_neg=dataset2_neg.sample(frac=0.8,random_state=0,replace=True)
#dataset2=pd.concat([dataset2_pos,dataset2_neg])
dataset1=dataset1.drop(['second_category_x','second_category_y'],axis=1)
dataset2=dataset2.drop(['second_category_x','second_category_y'],axis=1)
dataset3=dataset3.drop(['second_category_x','second_category_y'],axis=1)

#%%
watchlist = [(dataset1, label1)]#watchlist
#watchlist = [(dataset2, label2)]#watchlist
model = xgb.XGBClassifier(
        #objective='rank:pairwise',
        objective='binary:logistic',
 	     eval_metric='logloss',
 	     gamma=0.1,
 	     min_child_weight=1.1,
 	     max_depth=4,
 	     reg_lambda=1,
 	     subsample=0.9,
 	     colsample_bytree=0.9,
 	     colsample_bylevel=0.9,
        learning_rate=0.01,
 	     tree_method='exact',
 	     seed=0,
        n_estimators=3000 
        )
#model.fit(dataset1,label1,eval_set=watchlist)
model.fit(dataset2,label2,early_stopping_rounds=200,eval_set=watchlist)#747 1081
#model.fit(dataset1,label1,early_stopping_rounds=200,eval_set=watchlist)
"""
用dataset1训练 测试dataset2 0.0818
"""
#%%
xgb.plot_importance(model)
pyplot.show()

feature_importance=pd.Series(model.feature_importances_)
feature_importance.index=dataset2.columns
#%%
dataset3_pre['predicted_score']=model.predict_proba(dataset3)[:,1]
dataset3_pre.to_csv('20180416_0.0816_xgboost.txt',sep=" ",index=False)
dataset3_pre.drop_duplicates(inplace=True)
#%%
import lightgbm as lgb
# 线下学习
label22=np.array(label2).squeeze()
label11=np.array(label1).squeeze()
watchlist = [(dataset1, label11)]#watchlist
#watchlist = [(dataset2, label22)]#watchlist

gbm = lgb.LGBMRegressor(objective='binary',
                        is_unbalance=True,
                        num_leaves=100,
                        learning_rate=0.01,
                        n_estimators=2000,
                        colsample_bytree = 0.9,
                        subsample = 0.9,
                        seed=0
                        )
#model = lgb.LGBMClassifier(
#    random_state=666,
#    max_depth=4,
#    subsample=0.80,
#    n_estimators=100,
#    colsample_bytree=0.6,
##    reg_alpha=0.01,
#    learning_rate=0.1,
##    reg_lambda=0.01,
#    # is_unbalance=True,
#    # scale_pos_weight=1,
#    min_child_samples=40,
#    subsample_freq=2,
#)
#gbm = lgb.LGBMRegressor(objective='binary',
#                        #num_leaves=60,
#                        #is_unbalance='True',
#                        max_depth=5,
#                        colsample_bytree=0.7,
#                        min_child_weight =1.1,
#                        learning_rate=0.05,
#                        reg_lambda =0,
#                        reg_alpha=0,
#                        random_state =0,
#                        min_child_samples =20,
#                        is_unbalance =True,
#                        n_estimators=10000,
#                        #max_bin = 55, 
#                        #bagging_fraction = 0.8,
#                        #bagging_freq = 5, 
#                        #feature_fraction = 0.8,
#                        #feature_fraction_seed=9, 
#                        #bagging_seed=9,
#                        #min_data_in_leaf =6, 
#                        #min_sum_hessian_in_leaf = 11
#                        )
#                        colsample_bytree = 0.7,
#                        subsample = 0.7)
#                        min_child_weight=1.1)
gbm.fit(dataset2,label22,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)
#gbm.fit(dataset1,label11,
#    eval_set=watchlist,
#    eval_metric=['binary_logloss'],
#    early_stopping_rounds= 200)
#%%
from sklearn.metrics import log_loss
print(log_loss(y_train,y_tt))
#%%
feature_importance=pd.Series(gbm.feature_importances_)
feature_importance.index=dataset2.columns
#%%
dataset3_pre['predicted_score']=gbm.predict(dataset3,num_iteration=gbm.best_iteration_)
dataset3_pre.to_csv('20180415_0.0826_gbm.txt',sep=" ",index=False)
dataset3_pre.drop_duplicates(inplace=True)
