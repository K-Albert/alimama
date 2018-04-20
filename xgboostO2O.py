# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:20:26 2018

@author: lenovo
"""
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
#%%
dataset3=pd.read_csv('data/dataset3.csv')
dataset3_pre=dataset3[['user_id','coupon_id','date_received']]
dataset3.drop(['user_id','coupon_id','date_received'],axis=1,inplace=True)
dataset2=pd.read_csv('data/dataset2.csv')
dataset1=pd.read_csv('data/dataset1.csv')
#dataset12 = pd.concat([dataset1,dataset2],axis=0)
label2=dataset2[['label']]
label1=dataset1[['label']]
#label12=dataset12[['label']]
dataset1.drop(['label'],axis=1,inplace=True)
dataset2.drop(['label'],axis=1,inplace=True)
#dataset12.drop(['label'],axis=1,inplace=True)
#%%
dataset1.drop(['total_click','use_coupon_online','day_gap_before','day_gap_after'],axis=1,inplace=True)
dataset2.drop(['total_click','use_coupon_online','day_gap_before','day_gap_after'],axis=1,inplace=True)
dataset3.drop(['total_click','use_coupon_online','day_gap_before','day_gap_after'],axis=1,inplace=True)
#dataset1.drop(['total_click','use_coupon_online','user_merchant_usecoupon','max_dis','min_dis','user_merchant_buyrate','min_dis_user','max_dis_user'],axis=1,inplace=True)
#dataset2.drop(['total_click','use_coupon_online','user_merchant_usecoupon','max_dis','min_dis','user_merchant_buyrate','min_dis_user','max_dis_user'],axis=1,inplace=True)
#dataset3.drop(['total_click','use_coupon_online','user_merchant_usecoupon','max_dis','min_dis','user_merchant_buyrate','min_dis_user','max_dis_user'],axis=1,inplace=True)
#dataset12.drop(['total_click','use_coupon_online','user_merchant_usecoupon','max_dis','min_dis','user_merchant_buyrate','min_dis_user','max_dis_user'],axis=1,inplace=True)

watchlist = [(dataset1, label1)]#watchlist
#watchlist = [(dataset2, label2)]#watchlist
#watchlist = [(dataset12, label12)]#watchlist
#%%
model = xgb.XGBClassifier(
        #objective='rank:pairwise',
        objective='binary:logistic',
 	     eval_metric='auc',
 	     gamma=0.1,
 	     min_child_weight=1.1,
 	     max_depth=5,
 	     reg_lambda=10,
 	     subsample=0.7,
 	     colsample_bytree=0.7,
 	     colsample_bylevel=0.7,
        learning_rate=0.01,
 	     tree_method='exact',
 	     seed=0,
        n_estimators=3000 
        )
#model.fit(dataset1,label1,eval_set=watchlist)
model.fit(dataset2,label2,early_stopping_rounds=200,eval_set=watchlist)
#%%
print('auc', roc_auc_score(label2, model.predict_proba(dataset2)[:,1]))

temp =  model.predict_proba(dataset2_dispose)[:,1]
temp =  MinMaxScaler().fit_transform(temp.reshape(-1,1))
temp.max()
temp.min()
print('auc', roc_auc_score(label2,temp))
#%%
#注意这里要取第二列
dataset3_pre['label'] = model.predict_proba(dataset3)[:,1]
#%%
dataset3_pre.label = MinMaxScaler().fit_transform(dataset3_pre.label.reshape(-1, 1))
dataset3_pre.columns=(['User_id','Coupon_id','Date_received','Probability'])
#dataset3_pre.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_pre.to_csv("xgb_preds.csv",index=None,header=None)

xgb.plot_importance(model)
pyplot.show()

feature_importance=pd.Series(model.feature_importances_)
feature_importance.index=dataset2.columns
#%%正负样本比例
label2.label.value_counts()
#0    127346
#1      8955
label1.label.value_counts()
#0    233996
#1     23130
#%%处理缺失值(数据清洗？)
#dataset3_dispose=dataset3[dataset3.distance.notnull()]
#dataset2_dispose=dataset2[dataset2.distance.notnull()]
#dataset1_dispose=dataset1[dataset1.distance.notnull()]
#dataset3_dispose=dataset3_dispose[dataset3_dispose.mean_dis.notnull()]
#dataset2_dispose=dataset2_dispose[dataset2_dispose.mean_dis.notnull()]
#dataset1_dispose=dataset1_dispose[dataset1_dispose.mean_dis.notnull()]
dataset3=pd.read_csv('data/dataset3.csv')
dataset2=pd.read_csv('data/dataset2.csv')
dataset1=pd.read_csv('data/dataset1.csv')
#%%
dataset3_dispose=dataset3
dataset2_dispose=dataset2
dataset1_dispose=dataset1

dataset2_dispose_pos=dataset2_dispose[dataset2_dispose['label']==1]
dataset2_dispose_neg=dataset2_dispose[dataset2_dispose['label']==0]

dataset2_dispose_neg_sample=dataset2_dispose_neg.sample(n=dataset2_dispose_pos.label.size,random_state=0)
dataset2_dispose=pd.concat([dataset2_dispose_pos,dataset2_dispose_neg_sample],axis=0)

label2=dataset2_dispose[['label']]
label1=dataset1_dispose[['label']]
dataset2_dispose.drop(['label'],axis=1,inplace=True)
dataset1_dispose.drop(['label'],axis=1,inplace=True)
dataset1_dispose.drop(['total_click','use_coupon_online','day_gap_before','day_gap_after'],axis=1,inplace=True)
dataset2_dispose.drop(['total_click','use_coupon_online','day_gap_before','day_gap_after'],axis=1,inplace=True)
dataset3_dispose.drop(['total_click','use_coupon_online','day_gap_before','day_gap_after'],axis=1,inplace=True)
dataset3_dispose.drop(['user_id','coupon_id','date_received'],axis=1,inplace=True)
#%%
dataset3_dispose['label_user_coupon_feature_receive_count'].fillna(value=0,inplace=True)
dataset3_dispose['label_user_coupon_feature_buy_count'].fillna(value=0,inplace=True)
dataset3_dispose['label_user_coupon_feature_rate'].fillna(value=0,inplace=True)
#dataset3_dispose['day_gap_before'].fillna(value=30,inplace=True)
#dataset3_dispose['day_gap_after'].fillna(value=30,inplace=True)

d=dataset3_dispose.distance[dataset3_dispose.distance.notnull()]
mean=round(d.mean())
dataset3_dispose['distance'].fillna(value=mean,inplace=True)

d=dataset3_dispose.min_dis[dataset3_dispose.min_dis.notnull()]
mean=round(d.mean())
dataset3_dispose['min_dis'].fillna(value=mean,inplace=True)
d=dataset3_dispose.max_dis[dataset3_dispose.max_dis.notnull()]
mean=round(d.mean())
dataset3_dispose['max_dis'].fillna(value=mean,inplace=True)
d=dataset3_dispose.mean_dis[dataset3_dispose.mean_dis.notnull()]
mean=round(d.mean())
dataset3_dispose['mean_dis'].fillna(value=mean,inplace=True)

d=dataset3_dispose.min_dis_user[dataset3_dispose.min_dis_user.notnull()]
mean=round(d.mean())
dataset3_dispose['min_dis_user'].fillna(value=mean,inplace=True)
d=dataset3_dispose.max_dis_user[dataset3_dispose.max_dis_user.notnull()]
mean=round(d.mean())
dataset3_dispose['max_dis_user'].fillna(value=mean,inplace=True)
d=dataset3_dispose.mean_dis_user[dataset3_dispose.mean_dis_user.notnull()]
mean=round(d.mean())
dataset3_dispose['mean_dis_user'].fillna(value=mean,inplace=True)

dataset3_dispose[['total_buy','buy_use_coupon','total_coupon_buy','buy_rate','coupon_rate_buy']]=dataset3_dispose[['total_buy','buy_use_coupon','total_coupon_buy','buy_rate','coupon_rate_buy']].replace(np.nan,0)
dataset3_dispose.replace(np.nan,0,inplace=True)



#%%
dataset2_dispose['label_user_coupon_feature_receive_count'].fillna(value=0,inplace=True)
dataset2_dispose['label_user_coupon_feature_buy_count'].fillna(value=0,inplace=True)
dataset2_dispose['label_user_coupon_feature_rate'].fillna(value=0,inplace=True)
#dataset2_dispose['day_gap_before'].fillna(value=30,inplace=True)
#dataset2_dispose['day_gap_after'].fillna(value=30,inplace=True)

d=dataset2_dispose.distance[dataset2_dispose.distance.notnull()]
mean=round(d.mean())
dataset2_dispose['distance'].fillna(value=mean,inplace=True)

d=dataset2_dispose.min_dis[dataset2_dispose.min_dis.notnull()]
mean=round(d.mean())
dataset2_dispose['min_dis'].fillna(value=mean,inplace=True)
d=dataset2_dispose.max_dis[dataset2_dispose.max_dis.notnull()]
mean=round(d.mean())
dataset2_dispose['max_dis'].fillna(value=mean,inplace=True)
d=dataset2_dispose.mean_dis[dataset2_dispose.mean_dis.notnull()]
mean=round(d.mean())
dataset2_dispose['mean_dis'].fillna(value=mean,inplace=True)

d=dataset2_dispose.min_dis_user[dataset2_dispose.min_dis_user.notnull()]
mean=round(d.mean())
dataset2_dispose['min_dis_user'].fillna(value=mean,inplace=True)
d=dataset2_dispose.max_dis_user[dataset2_dispose.max_dis_user.notnull()]
mean=round(d.mean())
dataset2_dispose['max_dis_user'].fillna(value=mean,inplace=True)
d=dataset2_dispose.mean_dis_user[dataset2_dispose.mean_dis_user.notnull()]
mean=round(d.mean())
dataset2_dispose['mean_dis_user'].fillna(value=mean,inplace=True)

dataset2_dispose[['total_buy','buy_use_coupon','total_coupon_buy','buy_rate','coupon_rate_buy']]=dataset2_dispose[['total_buy','buy_use_coupon','total_coupon_buy','buy_rate','coupon_rate_buy']].replace(np.nan,0)
dataset2_dispose.replace(np.nan,0,inplace=True)
#%%
dataset1_dispose['label_user_coupon_feature_receive_count'].fillna(value=0,inplace=True)
dataset1_dispose['label_user_coupon_feature_buy_count'].fillna(value=0,inplace=True)
dataset1_dispose['label_user_coupon_feature_rate'].fillna(value=0,inplace=True)
#dataset1_dispose['day_gap_before'].fillna(value=30,inplace=True)
#dataset1_dispose['day_gap_after'].fillna(value=30,inplace=True)

d=dataset1_dispose.distance[dataset1_dispose.distance.notnull()]
mean=round(d.mean())
dataset1_dispose['distance'].fillna(value=mean,inplace=True)

d=dataset1_dispose.min_dis[dataset1_dispose.min_dis.notnull()]
mean=round(d.mean())
dataset1_dispose['min_dis'].fillna(value=mean,inplace=True)
d=dataset1_dispose.max_dis[dataset1_dispose.max_dis.notnull()]
mean=round(d.mean())
dataset1_dispose['max_dis'].fillna(value=mean,inplace=True)
d=dataset1_dispose.mean_dis[dataset1_dispose.mean_dis.notnull()]
mean=round(d.mean())
dataset1_dispose['mean_dis'].fillna(value=mean,inplace=True)

d=dataset1_dispose.min_dis_user[dataset1_dispose.min_dis_user.notnull()]
mean=round(d.mean())
dataset1_dispose['min_dis_user'].fillna(value=mean,inplace=True)
d=dataset1_dispose.max_dis_user[dataset1_dispose.max_dis_user.notnull()]
mean=round(d.mean())
dataset1_dispose['max_dis_user'].fillna(value=mean,inplace=True)
d=dataset1_dispose.mean_dis_user[dataset1_dispose.mean_dis_user.notnull()]
mean=round(d.mean())
dataset1_dispose['mean_dis_user'].fillna(value=mean,inplace=True)

dataset1_dispose[['total_buy','buy_use_coupon','total_coupon_buy','buy_rate','coupon_rate_buy']]=dataset1_dispose[['total_buy','buy_use_coupon','total_coupon_buy','buy_rate','coupon_rate_buy']].replace(np.nan,0)
dataset1_dispose.replace(np.nan,0,inplace=True)
#%%
dataset1




#%%



label2=dataset2_dispose[['label']]
label1=dataset1_dispose[['label']]
dataset2_dispose.drop(['label'],axis=1,inplace=True)
dataset1_dispose.drop(['label'],axis=1,inplace=True)

dataset3_pre=dataset3_dispose[['user_id','coupon_id','date_received']]
dataset3_dispose.drop(['user_id','coupon_id','date_received'],axis=1,inplace=True)

dataset1_dispose.drop(['total_click','use_coupon_online','day_gap_before','day_gap_after'],axis=1,inplace=True)
dataset2_dispose.drop(['total_click','use_coupon_online','day_gap_before','day_gap_after'],axis=1,inplace=True)
dataset3_dispose.drop(['total_click','use_coupon_online','day_gap_before','day_gap_after'],axis=1,inplace=True)

#%%
dataset1_dispose.to_csv('20180410_1100/dataset1.csv',index=None)
dataset2_dispose.to_csv('20180410_1100/dataset2.csv',index=None)
dataset3_dispose.to_csv('20180410_1100/dataset3.csv',index=None)
label1.to_csv('20180410_1100/label1.csv',index=None)
label2.to_csv('20180410_1100/label2.csv',index=None)
#%%
watchlist = [(dataset1_dispose, label1)]#watchlist
#watchlist = [(dataset2_dispose, label2)]#watchlist
#%%
model = xgb.XGBClassifier(
        objective='binary:logistic',
 	     eval_metric='auc',
 	     gamma=0.1,
 	     min_child_weight=1.1,
 	     max_depth=5,
 	     reg_lambda=1,
 	     subsample=0.7,
 	     colsample_bytree=0.7,
 	     colsample_bylevel=0.7,
        learning_rate=0.01,
 	     tree_method='exact',
 	     seed=0,
         n_estimators=3000 
        )
#model.fit(dataset1,label1,eval_set=watchlist)
model.fit(dataset2_dispose,label2,early_stopping_rounds=300,eval_set=watchlist)
#%%


dataset3_pre['label'] = model.predict_proba(dataset3_dispose)[:,1]
dataset3_pre.label = MinMaxScaler().fit_transform(dataset3_pre.label.reshape(-1, 1))
dataset3_pre.columns=(['User_id','Coupon_id','Date_received','Probability'])
dataset3_pre.to_csv("xgb_preds.csv",index=None,header=None)

xgb.plot_importance(model)
pyplot.show()
feature_importance=pd.Series(model.feature_importances_)
feature_importance.index=dataset2.columns
#%%
# load model and data in
model2 = xgb.Booster(model_file='20180410_1000.model')
d2 = xgb.DMatrix(dataset1_dispose)
#preds2 = model2.predict(dtest2)
temp =  model2.predict(d2)
print('auc', roc_auc_score(label1,temp))

#%% gbdt
dataset2_dispose.to_csv('20180411_2100/dataset2_dispose.csv')
dataset1_dispose.to_csv('20180411_2100/dataset1_dispose.csv')
dataset3_dispose.to_csv('20180411_2100/dataset3_dispose.csv')

label1.to_csv('20180411_2100/label1.csv')
label2.to_csv('20180411_2100/label2.csv')
#%%

dataset1_dispose=pd.read_csv('20180410_1100/dataset1.csv')
dataset2_dispose=pd.read_csv('20180410_1100/dataset2.csv')
dataset3_dispose=pd.read_csv('20180410_1100/dataset3.csv')


label1=pd.read_csv('20180410_1100/label1.csv')
label2=pd.read_csv('20180410_1100/label2.csv')
label2.columns=['label','drop']
label2.drop('drop',axis=1,inplace=True)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

#modelgbdt = GradientBoostingClassifier(random_state=10)
modelgbdt=GradientBoostingClassifier(criterion='friedman_mse',
                                     init=None,
                                     learning_rate=0.01,
                                     loss='deviance', 
                                     max_depth=7,
                                     max_features=None,
                                     max_leaf_nodes=None,
                                     min_samples_leaf=5, 
                                     min_samples_split=10,
                                     min_weight_fraction_leaf=0.0,
                                     n_estimators=350,
                                     presort='auto', 
                                     random_state=10, 
                                     subsample=0.7, 
                                     verbose=2,
                                     warm_start=False,
                                     )
modelgbdt.fit(dataset2_dispose,label2)

y_predprob1= modelgbdt.predict_proba(dataset1_dispose)[:,1]#0.912
y_predprob2= modelgbdt.predict_proba(dataset2_dispose)[:,1]
y_predprob3= modelgbdt.predict_proba(dataset3_dispose)[:,1]
metrics.roc_auc_score(label1, y_predprob1)

y_gdbt1 = dataset1_dispose
y_gdbt1['Probability'] = y_predprob1
y_gdbt1=y_gdbt1[['Probability']]
y_gdbt1.to_csv('blending/y_predprob1.csv')

y_gdbt2 = dataset2_dispose
y_gdbt2['Probability'] = y_predprob2
y_gdbt2=y_gdbt2[['Probability']]
y_gdbt2.to_csv('blending/y_predprob2.csv')

y_gdbt3 = dataset3_dispose
y_gdbt3['Probability'] = y_predprob3
y_gdbt3=y_gdbt3[['Probability']]
y_gdbt3.to_csv('blending/y_predprob3.csv')

dataset1_pre= model.predict_proba(dataset1_dispose)[:,1]
dataset1_pre= MinMaxScaler().fit_transform(dataset1_pre.reshape(-1, 1))
y_predprob1=y_predprob1.reshape(-1, 1)
y_predprob2=y_predprob2.reshape(-1, 1)
y_predprob3=y_predprob3.reshape(-1, 1)
boostpre=0.65*y_predprob1+0.35*dataset1_pre
print('auc', roc_auc_score(label1,boostpre))


boostpre=0.65*y_predprob3+0.35*dataset3_pre.Probability

dataset3_pre.columns=(['User_id','Coupon_id','Date_received','Probability'])

d=dataset3_pre
d.Probability=boostpre
d.to_csv("xgb_preds_blend.csv",index=None,header=None)

d.Probability=y_predprob3
d.to_csv("xgb_preds_gbdt.csv",index=None,header=None)


#%%随机森林
from sklearn.ensemble import RandomForestClassifier
modelrf=RandomForestClassifier(oob_score=True,
                               random_state=10,
                               n_estimators=300,
                               max_depth=20,
                               max_features='auto',
                               verbose=2)
modelrf.fit(dataset1,label1)
#%%
y_rf1 =modelrf.predict_proba(data_val)[:,1]
y_rf3=modelrf.predict_proba(dataset3_dispose)[:,1]
print(log_loss(label_val,y_rf1))
metrics.roc_auc_score(label_vv, y_rf1)
#%%
y_rf1=y_rf1.reshape(-1, 1)
y_rf3=y_rf3.reshape(-1, 1)
dataset1_pre = dataset1_pre.reshape(-1, 1)
#%%
boostpre=0.1*y_rf1+0.25*dataset1_pre+0.65*y_predprob1
t=0.8*dataset3_pre.Probability
t=t.reshape(-1,1)
boostpre=0.05*y_rf3+0.15*y_predprob3+t
boostpre=0.1*y_rf3+0.25*dataset3_pre.Probability+0.65*y_predprob3
d=dataset3_pre
d.Probability=boostpre
d.to_csv("xgb_preds_blend.csv",index=None,header=None)#0.91457
print('auc', roc_auc_score(label1,boostpre))
#%%
from sklearn.linear_model import LogisticRegression
blend_feature=pd.concat(dataset1_pre.Probability,y_rf1,axis=1,)
blend_feature=pd.concat(blend_feature,y_predprob1,axis=1)

blend_test=pd.concat(dataset3_pre.Probability,y_rf3,axis=1,)
blend_test=pd.concat(blend_feature,y_predprob3,axis=1)

modelblend=LogisticRegression()
modelblend.fit(blend_feature,label1)
y_submission=modelblend.predict_proba(blend_test)[:,1]

