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
dataset1=dataset1.sample(frac=0.333,random_state=2018,replace=True)

#dataset2_pos=dataset2[dataset2['is_trade']==1]
#dataset2_neg=dataset2[dataset2['is_trade']==0]
#
#dataset2_neg=dataset2_neg.sample(frac=0.8,random_state=0,replace=True)
#dataset2=pd.concat([dataset2_pos,dataset2_neg])
#%%
label2=dataset2[['is_trade']]
label1=dataset1[['is_trade']]
dataset3_pre=dataset3[['instance_id']]

dataset3.drop(['instance_id','second_category'],axis=1,inplace=True)
dataset1.drop(['is_trade','second_category'],axis=1,inplace=True)
dataset2.drop(['is_trade','second_category'],axis=1,inplace=True)
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
dataset1=dataset1.drop(['hour_trans_rate','hour_trans_dif_user_rate','hour_dif_user_age_level_trans_rate','hour_dif_user_star_level_trans_rate','hour_dif_user_occupation_trans_rate','hour_dif_user_gender_trans_rate'],axis=1)
dataset2=dataset2.drop(['hour_trans_rate','hour_trans_dif_user_rate','hour_dif_user_age_level_trans_rate','hour_dif_user_star_level_trans_rate','hour_dif_user_occupation_trans_rate','hour_dif_user_gender_trans_rate'],axis=1)
dataset3=dataset3.drop(['hour_trans_rate','hour_trans_dif_user_rate','hour_dif_user_age_level_trans_rate','hour_dif_user_star_level_trans_rate','hour_dif_user_occupation_trans_rate','hour_dif_user_gender_trans_rate'],axis=1)

#%%
dataset1=dataset1.drop(['label_user_ith_click_normalize','label_user_ith_click','label_user_per_hour_trans_rate','user_per_hour_trans_desier','user_per_hour_trans_rate'],axis=1)
dataset2=dataset2.drop(['label_user_ith_click_normalize','label_user_ith_click','label_user_per_hour_trans_rate','user_per_hour_trans_desier','user_per_hour_trans_rate'],axis=1)
dataset3=dataset3.drop(['label_user_ith_click_normalize','label_user_ith_click','label_user_per_hour_trans_rate','user_per_hour_trans_desier','user_per_hour_trans_rate'],axis=1)
#%%
dataset1=dataset1.drop(['label_is_latest_time000'],axis=1)
dataset2=dataset2.drop(['label_is_latest_time000'],axis=1)
dataset3=dataset3.drop(['label_is_latest_time000'],axis=1)
#%%
dataset1=dataset1.drop(['label_now_item_sales_level_bigger_x','label_now_item_sales_level_bigger_y','label_now_item_collected_level_bigger','shop_score_service_bigger'],axis=1)
dataset2=dataset2.drop(['label_now_item_sales_level_bigger_x','label_now_item_sales_level_bigger_y','label_now_item_collected_level_bigger','shop_score_service_bigger'],axis=1)
dataset3=dataset3.drop(['label_now_item_sales_level_bigger_x','label_now_item_sales_level_bigger_y','label_now_item_collected_level_bigger','shop_score_service_bigger'],axis=1)
#%%
dataset1=dataset1.drop(['item_pv_level'],axis=1)
dataset2=dataset2.drop(['item_pv_level'],axis=1)
dataset3=dataset3.drop(['item_pv_level'],axis=1)


#%%
"""
user_buy_cnt
user_buy_rate
user_buy_dif_merchant_cnt
user_buy_dif_merchant_brand_cnt
user_buy_dif_merchant_city_cnt
user_buy_dif_merchant_price_cnt
user_buy_dif_merchant_sales_cnt
user_buy_dif_merchant_collected_cnt
user_buy_mean_merchant_sales_level_y
user_buy_dif_shop_cnt
user_buy_dif_shop_rate
user_buy_dif_shop_review_num_level_cnt
user_buy_dif_shop_star_level_cnt
user_buy_mean_shop_score_description
user_item_pv_level_buy_rate
"""
#%%
dataset1=dataset1.drop(['user_buy_cnt','user_buy_rate','user_buy_dif_merchant_cnt','user_buy_dif_merchant_brand_cnt','user_buy_dif_merchant_city_cnt','user_buy_dif_merchant_price_cnt','user_buy_dif_merchant_sales_cnt','user_buy_dif_merchant_collected_cnt','user_buy_mean_merchant_sales_level_y'],axis=1)
dataset2=dataset2.drop(['user_buy_cnt','user_buy_rate','user_buy_dif_merchant_cnt','user_buy_dif_merchant_brand_cnt','user_buy_dif_merchant_city_cnt','user_buy_dif_merchant_price_cnt','user_buy_dif_merchant_sales_cnt','user_buy_dif_merchant_collected_cnt','user_buy_mean_merchant_sales_level_y'],axis=1)
dataset3=dataset3.drop(['user_buy_cnt','user_buy_rate','user_buy_dif_merchant_cnt','user_buy_dif_merchant_brand_cnt','user_buy_dif_merchant_city_cnt','user_buy_dif_merchant_price_cnt','user_buy_dif_merchant_sales_cnt','user_buy_dif_merchant_collected_cnt','user_buy_mean_merchant_sales_level_y'],axis=1)

dataset1=dataset1.drop(['user_buy_dif_shop_cnt','user_buy_dif_shop_rate','user_buy_dif_shop_review_num_level_cnt','user_buy_dif_shop_star_level_cnt','user_buy_mean_shop_score_description','user_item_pv_level_buy_rate'],axis=1)
dataset2=dataset2.drop(['user_buy_dif_shop_cnt','user_buy_dif_shop_rate','user_buy_dif_shop_review_num_level_cnt','user_buy_dif_shop_star_level_cnt','user_buy_mean_shop_score_description','user_item_pv_level_buy_rate'],axis=1)
dataset3=dataset3.drop(['user_buy_dif_shop_cnt','user_buy_dif_shop_rate','user_buy_dif_shop_review_num_level_cnt','user_buy_dif_shop_star_level_cnt','user_buy_mean_shop_score_description','user_item_pv_level_buy_rate'],axis=1)
#%%
t=dataset1['real_hour']
dataset1.drop('real_hour',axis=1,inplace=True)
dataset1.insert(0,'real_hour',t)

t=dataset2['real_hour']
dataset2.drop('real_hour',axis=1,inplace=True)
dataset2.insert(0,'real_hour',t)

t=dataset3['real_hour']
dataset3.drop('real_hour',axis=1,inplace=True)
dataset3.insert(0,'real_hour',t)

#%%
watchlist = [(dataset1, label1)]#watchlist
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
 	     seed=2018,
         missing=-1,
         n_estimators=3000 
        )
#model.fit(dataset1,label1,eval_set=watchlist)
model.fit(dataset2,label2,early_stopping_rounds=50,eval_set=watchlist)#747 1081  929 0.081411  5:989 1108 1096
#model.fit(dataset1,label1,early_stopping_rounds=200,eval_set=watchlist)
"""
用dataset1训练 测试dataset2 0.0818
"""
#%%
xgb.plot_importance(model)
pyplot.show()

feature_importance=pd.Series(model.feature_importances_)
feature_importance.index=dataset2.columns
feature_importance2=feature_importance.reset_index()
#%%
test_b = pd.read_csv('data/round1_ijcai_18_test_b_20180418.txt',sep=" ")
test_b=test_b[['instance']]
#%%
dataset3_pre['predicted_score']=model.predict_proba(dataset3)[:,1]
dataset3_pre=dataset3_pre[dataset3_pre['instance_id'].isin(test_b['instance_id'])].reset_index(drop=True)
dataset3_pre.to_csv('20180418_0.08138_xgboost.txt',sep=" ",index=False)
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
                        learning_rate=0.05,
                        n_estimators=2000,
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
