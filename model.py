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
from sklearn.preprocessing import LabelEncoder
#%%
dataset5=pd.read_csv('data/dataset5.csv')
dataset4=pd.read_csv('data/dataset4.csv')
dataset3=pd.read_csv('data/dataset3.csv')
dataset2=pd.read_csv('data/dataset2.csv')
dataset1=pd.read_csv('data/dataset1.csv')
#%%
dataset1=pd.concat([dataset1,dataset2,dataset3]).reset_index(drop=True)
data_val=dataset4
dataset2=pd.concat([dataset2,dataset3,dataset4]).reset_index(drop=True)
dataset3=dataset5

#dataset1=dataset1.sample(frac=0.333,random_state=0,replace=True)

#dataset2_pos=dataset2[dataset2['is_trade']==1]s
#dataset2_neg=dataset2[dataset2['is_trade']==0]
#
#dataset2_neg=dataset2_neg.sample(frac=0.8,random_state=0,replace=True)
#dataset2=pd.concat([dataset2_pos,dataset2_neg])
#%%
label1=dataset1[['is_trade']]
label2=dataset2[['is_trade']]
label_val = data_val[['is_trade']]
dataset3_pre=dataset3[['instance_id']]

dataset3.drop(['instance_id'],axis=1,inplace=True)
dataset1.drop(['is_trade'],axis=1,inplace=True)
dataset2.drop(['is_trade'],axis=1,inplace=True)
data_val.drop(['is_trade'],axis=1,inplace=True)
dataset1.drop(['instance_id'],axis=1,inplace=True)
dataset2.drop(['instance_id'],axis=1,inplace=True)
data_val.drop(['instance_id'],axis=1,inplace=True)

le = LabelEncoder()
dataset1['second_category']=le.fit_transform(dataset1['second_category'])
dataset2['second_category']=le.fit_transform(dataset2['second_category'])
dataset3['second_category']=le.fit_transform(dataset3['second_category'])
data_val['second_category']=le.fit_transform(data_val['second_category'])

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
#%%
watchlist = [(data_val, label_val)]#watchlist
#watchlist = [(dataset2, label2)]#watchlist
model = xgb.XGBClassifier(
        #objective='rank:pairwise',
        objective='binary:logistic',
 	     eval_metric='logloss',
 	     gamma=0.1,
 	     min_child_weight=1.1,
 	     max_depth=3,
 	     reg_lambda=10,
 	     subsample=0.9,
 	     colsample_bytree=0.9,
 	     colsample_bylevel=0.9,
        learning_rate=0.01,
 	     tree_method='exact',
 	     seed=0,
          missing=-1,
        n_estimators=5000 
        )
model.fit(dataset1,label1,early_stopping_rounds=200,eval_set=watchlist)#0.79454
#%%
#提交
watchlist = [(dataset2, label2)]#watchlist

model_sub = xgb.XGBClassifier(
        #objective='rank:pairwise',
        objective='binary:logistic',
 	     eval_metric='logloss',
 	     gamma=0.1,
 	     min_child_weight=1.1,
 	     max_depth=3,
 	     reg_lambda=10,
 	     subsample=0.9,
 	     colsample_bytree=0.9,
 	     colsample_bylevel=0.9,
        learning_rate=0.01,
 	     tree_method='exact',
 	     seed=0,
          missing=-1,
        n_estimators=1812 
        )
model_sub.fit(dataset2,label2,early_stopping_rounds=200,eval_set=watchlist)#1450 1604 1733 1814

#%%
xgb.plot_importance(model)
pyplot.show()
feature_importance=pd.Series(model.feature_importances_)
feature_importance.index=dataset2.columns
#%%
d=test_b[['instance_id']]
dataset3_pre['predicted_score']=model_sub.predict_proba(dataset3)[:,1]
dataset3_pre=dataset3_pre[dataset3_pre['instance_id'].isin(d['instance_id'])]
dataset3_pre.to_csv('data/20180419_0.0794285_xgboost.txt',sep=" ",index=False)
dataset3_pre.drop_duplicates(inplace=True)
#%%
import lightgbm as lgb
# 线下学习
labelvv=np.array(label_val).squeeze()#1632 0.794285
label22=np.array(label2).squeeze()
label11=np.array(label1).squeeze()
watchlist = [(data_val, labelvv)]#watchlist
#watchlist = [(dataset2, label22)]#watchlist

gbm = lgb.LGBMRegressor(objective='binary',
                        #is_unbalance=True,
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=3000,
                        colsample_bytree = 0.9,
                        subsample = 0.9,
                        seed=0
                        )

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
gbm.fit(dataset1,label11,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)
#%%
import lightgbm as lgb
# 线下学习
labelvv=np.array(label_val).squeeze()#1632 0.794285
label22=np.array(label2).squeeze()
label11=np.array(label1).squeeze()
#watchlist = [(data_val, labelvv)]#watchlist
watchlist = [(dataset2, label22)]#watchlist

gbm = lgb.LGBMRegressor(objective='binary',
                        #is_unbalance=True,
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=1632,
                        colsample_bytree = 0.9,
                        subsample = 0.9,
                        seed=0
                        )
#               
gbm.fit(dataset2,label22,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)
#%%
from sklearn.metrics import log_loss
print(log_loss(y_train,y_tt))
#%%
feature_importance=pd.Series(gbm.feature_importances_)
feature_importance.index=dataset2.columns
#%%
d=test_b[['instance_id']]
dataset3_pre=test[['instance_id']]
dataset3_pre['predicted_score']=gbm.predict(dataset3)
dataset3_pre=dataset3_pre[dataset3_pre['instance_id'].isin(d['instance_id'])]
dataset3_pre.to_csv('20180415_0.0794285_gbm.txt',sep=" ",index=False)
dataset3_pre.drop_duplicates(inplace=True)
