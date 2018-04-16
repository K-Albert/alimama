# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 02:07:55 2018

@author: surface
"""
from sklearn.cross_validation import KFlod
#%%
n_folds=5
stacker=xgb.XGBClassifier(
        #objective='rank:pairwise',
        objective='binary:logistic',
 	     eval_metric='logloss',
 	     gamma=0.1,
 	     min_child_weight=0.9,
 	     max_depth=3,
 	     reg_lambda=10,
 	     subsample=0.9,
 	     colsample_bytree=0.9,
 	     colsample_bylevel=0.9,
        learning_rate=0.01,
 	     tree_method='exact',
 	     seed=0,
        n_estimators=3000 
        )
base_models=[xgboost,gbdt,rf]

X=dataset2
y=label2
T=dataset1
T2=dataset3

X = np.array(X)
y = np.array(y)
T = np.array(T)
T2=np.array(T2)

folds = list(KFold(len(y), n_folds=5, shuffle=True, random_state=2016))
S_train = np.zeros((X.shape[0], len(base_models)))
S_test = np.zeros((T.shape[0], len(base_models)))
S_test2 = np.zeros((T2.shape[0], len(base_models)))

for i, clf in enumerate(self.base_models):
    S_test_i = np.zeros((T.shape[0], len(folds)))
    S_test2_i = np.zeros((T2.shape[0], len(folds)))

    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_holdout)[:,1]
        S_train[test_idx, i] = y_pred
        S_test_i[:, j] = clf.predict(T)[:,1]
        S_test2_i[:, j] = clf.predict(T2)[:,1]

    S_test[:, i] = S_test_i.mean(1)
    S_test2[:, i] = S_test2_i.mean(1)
    
stacker.fit(S_train, y)
y_pred = stacker.predict_proba(S_test)[:,1]
y_pred_sub = stacker.predict_proba(S_test2)[:,1]

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