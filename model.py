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
#%%  yong  lastest time 和 item_id  item_brand item_city 相与
dataset1_raw=pd.concat([dataset1,dataset2,dataset3]).reset_index(drop=True)
data_val_raw=dataset4
dataset2_raw=pd.concat([dataset2,dataset3,dataset4]).reset_index(drop=True)
dataset3_raw=dataset5

#%%
label1=dataset1_raw[['is_trade']]
label2=dataset2_raw[['is_trade']]
label_val = data_val_raw[['is_trade']]
dataset3_pre=dataset3_raw[['instance_id']]

dataset3_raw.drop(['instance_id'],axis=1,inplace=True)
dataset1_raw.drop(['is_trade'],axis=1,inplace=True)
dataset2_raw.drop(['is_trade'],axis=1,inplace=True)
data_val_raw.drop(['is_trade'],axis=1,inplace=True)
dataset1_raw.drop(['instance_id'],axis=1,inplace=True)
dataset2_raw.drop(['instance_id'],axis=1,inplace=True)
data_val_raw.drop(['instance_id'],axis=1,inplace=True)

le = LabelEncoder()
dataset1_raw['second_category']=le.fit_transform(dataset1_raw['second_category'])
dataset2_raw['second_category']=le.fit_transform(dataset2_raw['second_category'])
dataset3_raw['second_category']=le.fit_transform(dataset3_raw['second_category'])
data_val_raw['second_category']=le.fit_transform(data_val_raw['second_category'])
#%%
dataset1=dataset1_raw.copy()
dataset2=dataset2_raw.copy()
dataset3=dataset3_raw.copy()
data_val=data_val_raw.copy()
#%%
dataset1=dataset1.drop('item_city_id_after','item_brand_id_after',axis=1)
dataset2=dataset2.drop('item_city_id_after','item_brand_id_after',axis=1)
dataset3=dataset3.drop('item_city_id_after','item_brand_id_after',axis=1)
data_val=data_val.drop('item_city_id_after','item_brand_id_after',axis=1)
#%%

dataset1=dataset1_raw.sample(frac=0.6,axis=1,random_state=666)
data_val=data_val_raw.sample(frac=0.6,axis=1,random_state=666)#0.0794244

dataset1=dataset1_raw.sample(frac=0.6,axis=1,random_state=111)
data_val=data_val_raw.sample(frac=0.6,axis=1,random_state=111)#0.0797

dataset1=dataset1_raw.sample(frac=0.6,axis=1,random_state=2018)
data_val=data_val_raw.sample(frac=0.6,axis=1,random_state=2018)#0.079716

#%%
dataset1=dataset1.drop('label_user_ith_click',axis=1)
dataset2=dataset2.drop('label_user_ith_click',axis=1)
dataset3=dataset3.drop('label_user_ith_click',axis=1)
data_val=data_val.drop('label_user_ith_click',axis=1)
#%%
dataset1=dataset1.drop('label_user_ith_click_normalize',axis=1)
dataset2=dataset2.drop('label_user_ith_click_normalize',axis=1)
dataset3=dataset3.drop('label_user_ith_click_normalize',axis=1)
data_val=data_val.drop('label_user_ith_click_normalize',axis=1)
#%%
dataset1=dataset1.drop('user_id','item_id','shop_id',axis=1)
dataset2=dataset2.drop('user_id','item_id','shop_id',axis=1)
dataset3=dataset3.drop('user_id','item_id','shop_id',axis=1)
data_val=data_val.drop('user_id','item_id','shop_id',axis=1)

#%%
dataset1['label_item_and_time']=(dataset1['label_is_latest_time000'].astype('bool')&dataset1['label_item_id_has_ever'].astype('bool')).astype('int')
dataset2['label_item_and_time']=(dataset2['label_is_latest_time000'].astype('bool')&dataset2['label_item_id_has_ever'].astype('bool')).astype('int')
dataset3['label_item_and_time']=(dataset3['label_is_latest_time000'].astype('bool')&dataset3['label_item_id_has_ever'].astype('bool')).astype('int')
data_val['label_item_and_time']=(data_val['label_is_latest_time000'].astype('bool')&data_val['label_item_id_has_ever'].astype('bool')).astype('int')
#%%
dataset1['label_brand_and_time']=(dataset1['label_is_latest_time000'].astype('bool')&dataset1['label_item_brand_has_before'].astype('bool')).astype('int')
dataset2['label_brand_and_time']=(dataset2['label_is_latest_time000'].astype('bool')&dataset2['label_item_brand_has_before'].astype('bool')).astype('int')
dataset3['label_brand_and_time']=(dataset3['label_is_latest_time000'].astype('bool')&dataset3['label_item_brand_has_before'].astype('bool')).astype('int')
data_val['label_brand_and_time']=(data_val['label_is_latest_time000'].astype('bool')&data_val['label_item_brand_has_before'].astype('bool')).astype('int')
#%%
dataset1['label_city_and_time']=(dataset1['label_is_latest_time000'].astype('bool')&dataset1['label_item_city_has_ever'].astype('bool')).astype('int')
dataset2['label_city_and_time']=(dataset2['label_is_latest_time000'].astype('bool')&dataset2['label_item_city_has_ever'].astype('bool')).astype('int')
dataset3['label_city_and_time']=(dataset3['label_is_latest_time000'].astype('bool')&dataset3['label_item_city_has_ever'].astype('bool')).astype('int')
data_val['label_city_and_time']=(data_val['label_is_latest_time000'].astype('bool')&data_val['label_item_city_has_ever'].astype('bool')).astype('int')
#%%
dataset1['label_shop_and_time']=(dataset1['label_is_latest_time000'].astype('bool')&dataset1['label_shop_id_has_ever'].astype('bool')).astype('int')
dataset2['label_shop_and_time']=(dataset2['label_is_latest_time000'].astype('bool')&dataset2['label_shop_id_has_ever'].astype('bool')).astype('int')
dataset3['label_shop_and_time']=(dataset3['label_is_latest_time000'].astype('bool')&dataset3['label_shop_id_has_ever'].astype('bool')).astype('int')
data_val['label_shop_and_time']=(data_val['label_is_latest_time000'].astype('bool')&data_val['label_shop_id_has_ever'].astype('bool')).astype('int')

#%%
dataset1=dataset1.drop('label_is_latest_time000',axis=1)
dataset2=dataset2.drop('label_is_latest_time000',axis=1)
dataset3=dataset3.drop('label_is_latest_time000',axis=1)
data_val=data_val.drop('label_is_latest_time000',axis=1)
#%%
dataset1=dataset1.drop('label_last_click_time_gap',axis=1)
dataset2=dataset2.drop('label_last_click_time_gap',axis=1)
dataset3=dataset3.drop('label_last_click_time_gap',axis=1)
data_val=data_val.drop('label_last_click_time_gap',axis=1)
#%%
dataset1=dataset1.drop('label_is_latest_time',axis=1)
dataset2=dataset2.drop('label_is_latest_time',axis=1)
dataset3=dataset3.drop('label_is_latest_time',axis=1)
data_val=data_val.drop('label_is_latest_time',axis=1)
#%%
label_item_id_has_ever
label_is_latest_time

label_shop_id_has_ever
label_item_city_has
label_item_brand_has_before
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
labelvv=np.array(label_val).squeeze()#1644  0.0792826  3222  0.0655296
label22=np.array(label2).squeeze()
label11=np.array(label1).squeeze()
watchlist = [(data_val, labelvv)]#watchlist
#watchlist = [(dataset2, label22)]#watchlist

gbm = lgb.LGBMRegressor(objective='binary',
                        #is_unbalance=True,
                        #min_child_weight=1.1,
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=4000,
                        colsample_bytree = 0.5,
                        subsample = 0.6,
                        seed=0
                        )
gbm.fit(dataset1,label11,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)
#%%
feature_importance=pd.Series(gbm.feature_importances_)
feature_importance.index=data_val.columns
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
                        n_estimators=3222,
                        colsample_bytree = 0.5,
                        subsample = 0.6,
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
d=test_b[['instance_id']]
dataset3_pre=test[['instance_id']]
dataset3_pre['predicted_score']=gbm.predict(dataset3,num_iteration=gbm.best_iteration_)
dataset3_pre=dataset3_pre[dataset3_pre['instance_id'].isin(d['instance_id'])]
dataset3_pre.to_csv('data/20180420_0.0655296_gbm.txt',sep=" ",index=False)
dataset3_pre.drop_duplicates(inplace=True)
#%%
#gbm = lgb.LGBMRegressor(objective='binary',
#                        #num_leaves=60,
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
