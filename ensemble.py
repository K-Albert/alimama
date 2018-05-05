# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:47:44 2018

@author: surface
"""
#%%
#dataset1_1=dataset1_raw.sample(frac=0.6,axis=1,random_state=666666)#1559 0.0794297
#data_val1=data_val_raw.sample(frac=0.6,axis=1,random_state=666666)

dataset1_2=dataset1_raw.sample(frac=0.6,axis=1,random_state=666)
data_val2=data_val_raw.sample(frac=0.6,axis=1,random_state=666)#1527  0.0793547

#dataset1_3=dataset1_raw.sample(frac=0.6,axis=1,random_state=10)
#data_val3=data_val_raw.sample(frac=0.6,axis=1,random_state=10)#1817 0.0794175

dataset1_4=dataset1_raw.sample(frac=0.6,axis=1,random_state=222)
data_val4=data_val_raw.sample(frac=0.6,axis=1,random_state=222)#1825  0.0793614
#%%
dataset1_5=dataset1_raw.sample(frac=0.6,axis=1,random_state=22)
data_val5=data_val_raw.sample(frac=0.6,axis=1,random_state=22)#1712 0.795629 no

watchlist = [(data_val5, labelvv)]#watchlist
gbm5 = lgb.LGBMRegressor(objective='binary',
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=3000,
                        colsample_bytree = 0.5,
                        subsample = 0.6,
                        seed=0
                        )
gbm5.fit(dataset1_5,label11,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)#dataset1_6=dataset1_raw.sample(frac=0.6,axis=1,random_state=6666)
#data_val6=data_val_raw.sample(frac=0.6,axis=1,random_state=6666)#1771 0.7955033
##%%
#dataset1_7=dataset1_raw.sample(frac=0.6,axis=1,random_state=11)
#data_val7=data_val_raw.sample(frac=0.6,axis=1,random_state=11)#1558 0.796186

#%%
import lightgbm as lgb
# 线下学习
labelvv=np.array(label_val).squeeze()#1632 0.794285
label22=np.array(label2).squeeze()
label11=np.array(label1).squeeze()
watchlist = [(data_val, labelvv)]#watchlist
#%%
watchlist = [(data_val1, labelvv)]#watchlist
gbm1 = lgb.LGBMRegressor(objective='binary',
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=1559,
                        colsample_bytree = 0.5,
                        subsample = 0.6,
                        seed=0
                        )
gbm1.fit(dataset1_1,label11,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)
#%%
watchlist = [(data_val2, labelvv)]#watchlist
gbm2 = lgb.LGBMRegressor(objective='binary',
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=1527,
                        colsample_bytree = 0.5,
                        subsample = 0.6,
                        seed=0
                        )
gbm2.fit(dataset1_2,label11,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)
#%%
watchlist = [(data_val3, labelvv)]#watchlist
gbm3 = lgb.LGBMRegressor(objective='binary',
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=1817,
                        colsample_bytree = 0.5,
                        subsample = 0.6,
                        seed=0
                        )
gbm3.fit(dataset1_3,label11,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)
#%%
watchlist = [(data_val4, labelvv)]#watchlist
gbm4 = lgb.LGBMRegressor(objective='binary',
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=1825,
                        colsample_bytree = 0.5,
                        subsample = 0.6,
                        seed=0
                        )
gbm4.fit(dataset1_4,label11,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)

#%%
watchlist = [(data_val6, labelvv)]#watchlist
gbm6 = lgb.LGBMRegressor(objective='binary',
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=3000,
                        colsample_bytree = 0.9,
                        subsample = 0.9,
                        seed=0
                        )
gbm6.fit(dataset1_6,label11,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)
#%%
watchlist = [(data_val7, labelvv)]#watchlist
gbm7 = lgb.LGBMRegressor(objective='binary',
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=3000,
                        colsample_bytree = 0.7,
                        subsample = 0.7,
                        seed=0
                        )
gbm7.fit(dataset1_7,label11,
    eval_set=watchlist,
    eval_metric=['binary_logloss'],
    early_stopping_rounds= 100)
#%%
feature_importance1=pd.Series(gbm1.feature_importances_)
feature_importance1.index=data_val1.columns
#%%
from sklearn.cross_validation import KFold
base_models=[gbm,gbm2,gbm4]
X=[dataset1,dataset1_2,dataset1_4]
y=label1
y=np.array(y).squeeze()
T=[data_val,data_val2,data_val4]

folds = list(KFold(len(y), n_folds=5, shuffle=True, random_state=2018))

S_train = np.zeros((X[0].shape[0], len(base_models)))
S_test = np.zeros((T[0].shape[0], len(base_models)))

for i, clf in enumerate(base_models):
    S_test_i = np.zeros((T[i].shape[0], len(folds)))
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X[i].iloc[train_idx]
        y_train = y[train_idx]
        X_holdout = X[i].iloc[test_idx]
        # y_holdout = y[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_holdout)
        S_train[test_idx, i] = y_pred
        S_test_i[:, j] = clf.predict(T[i])
        print(j)
    S_test[:, i] = S_test_i.mean(1)
    print(i)

#%%
stacker=lgb.LGBMRegressor(objective='binary',#0.0792114 740
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=890,
                        colsample_bytree = 0.5,
                        subsample = 0.6,
                        seed=0
                        )
stacker.fit(S_train, y)
y_pred = stacker.predict(S_test)[:]

from sklearn.metrics import log_loss
print(log_loss(label_val,y_pred))
#%%  result_sub
dataset2_1=dataset2_raw.sample(frac=0.6,axis=1,random_state=111)
dataset3_1=dataset3_raw.sample(frac=0.6,axis=1,random_state=111)#0.0797

dataset2_2=dataset2_raw.sample(frac=0.6,axis=1,random_state=666)
dataset3_2=dataset3_raw.sample(frac=0.6,axis=1,random_state=666)#0.0794244

dataset2_3=dataset2_raw.sample(frac=0.6,axis=1,random_state=2018)
dataset3_3=dataset3_raw.sample(frac=0.6,axis=1,random_state=2018)#0.079716
#%%
base_models=[gbm1,gbm2,gbm3]
X=[dataset2_1,dataset2_2,dataset2_3]
y=label2
y=np.array(y).squeeze()
T=[dataset3_1,dataset3_2,dataset3_3]

folds = list(KFold(len(y), n_folds=5, shuffle=True, random_state=2018))

S_train = np.zeros((X[0].shape[0], len(base_models)))
S_test = np.zeros((T[0].shape[0], len(base_models)))

for i, clf in enumerate(base_models):
    S_test_i = np.zeros((T[0].shape[0], len(folds)))
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X[i].iloc[train_idx]
        y_train = y[train_idx]
        X_holdout = X[i].iloc[test_idx]
        # y_holdout = y[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_holdout)
        S_train[test_idx, i] = y_pred
        S_test_i[:, j] = clf.predict(T[i])
        print(j)
    S_test[:, i] = S_test_i.mean(1)
    print(i)

#%%
stacker=lgb.LGBMRegressor(objective='binary',
                        min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=850,
                        colsample_bytree = 0.9,
                        subsample = 0.9,
                        seed=0
                        )
stacker.fit(S_train, y)
d=test_b[['instance_id']]
dataset3_pre=test[['instance_id']]
dataset3_pre['predicted_score']=stacker.predict(S_test)[:]
dataset3_pre=dataset3_pre[dataset3_pre['instance_id'].isin(d['instance_id'])]
#%%
dataset3_pre.to_csv('20180415_0.0794285_gbm.txt',sep=" ",index=False)
dataset3_pre.drop_duplicates(inplace=True)
#%%
class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred