# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:23:54 2018

@author: lenovo
"""
import time
import pandas as pd
import os 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss
import lightgbm as lgbm
from numpy import random
os.getcwd() #get current working directory
os.chdir('F:\\006@天池\\0003@阿里妈妈')#change working directory
#%%
def LossFunction(y_test,y_pred):
    return log_loss(y_test,y_pred)
#%%
dataset1=pd.read_csv('data/train/dataset1_sub.csv')
dataset1=dataset1.drop(['item_id','shop_id','user_id'],axis=1)
dataset1=dataset1.drop(['real_hour'],axis=1)
dataset1=dataset1.drop(['label_user_ith_click'],axis=1)
label1=dataset1[['is_trade']]
#dataset1.drop(['is_trade'],axis=1,inplace=True)
dataset1.drop(['instance_id'],axis=1,inplace=True)
le = LabelEncoder()
dataset1['second_category']=le.fit_transform(dataset1['second_category'])
#%%
dataset2=pd.read_csv('data/train/dataset2_sub.csv')
dataset2=dataset2.drop(['item_id','shop_id','user_id'],axis=1)
dataset2=dataset2.drop(['real_hour'],axis=1)
dataset2=dataset2.drop(['label_user_ith_click'],axis=1)
dataset2_pre=dataset2[['instance_id']]
dataset2.drop(['instance_id'],axis=1,inplace=True)
dataset2['second_category']=le.fit_transform(dataset2['second_category'])
#%%
dataset2=dataset2.iloc[0:10]
#%%
def CrossValidation(X,y,clf,selectcol):
    totaltest=0
    print('Start Cross Validation')
    """This part is for the validation, modify it according to your demand"""
    for train_index,test_index in sfolder.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        X_train, X_test = X_train[selectcol], X_test[selectcol]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train,y_train, eval_set = [(X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=200)
        totaltest += LossFunction(y_test, clf.predict_proba(X_test,num_iteration=clf.best_iteration_)[:,1])
    totaltest /= 5.0
    print('Mean loss: {}'.format(totaltest))  
    return totaltest
#%%
model = {'lgb6': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000, max_depth=3, learning_rate = 0.05,colsample_bytree = 0.9,subsample = 0.9)}
modelSelect='lgb6'
sfolder = StratifiedKFold(n_splits=5,random_state=2018,shuffle=False)
selectcol=dataset1.columns.drop('is_trade') 
#%%
"""
How to use:
X=dataset1
y=dataset1['is_trade']       
totaltest=CrossValidation(X,y,model[modelSelect],selectcol)     
baseline:
""" 
X=dataset1
y=dataset1['is_trade']       
baseline=CrossValidation(X,y,model[modelSelect],selectcol) 
#%%
"""
每次从总训练集中抽出rate部分做训练，做交叉验证，记录得分
"""
random.seed(1)
rate = 0.3
allCircle =[random.randint(0,10000) for __ in range(10)] 
totaltest=[]
selectcol=dataset1.columns.drop('is_trade')      
for i in allCircle:
    dataset=dataset1.sample(frac=rate,random_state=i)
    X=dataset
    y=dataset['is_trade']
    test=CrossValidation(X,y,model[modelSelect],selectcol)
    totaltest.append(test)
    print('i is complete')
#%%
"""
观察得分，选出进行融合的模型
"""    
sortedNum=4
totaltestDic=dict(zip(allCircle,totaltest)) 
totaltestSort=sorted(totaltestDic.items(), key=lambda d: d[1])[0:sortedNum]
totaltestSort=list(dict(totaltestSort).keys())
#best 0.17330
#%%
"""
用几次预测的平均值做融合
"""
def genBasePred(clf,selectcol,totaltestSort,dataset1,dataset2):
    yPredTrain=[]
    yPredTest=[]
    
    for i in totaltestSort:
        dataset=dataset1.sample(frac=0.3,random_state=i)
        datasetTest = dataset2
        X=dataset
        y=dataset['is_trade'] 
        X_sub=datasetTest
        y_pred_train=[0 for __ in range(dataset1.shape[0])]
        y_pred_test=[0 for __ in range(datasetTest.shape[0])]
        for train_index,test_index in sfolder.split(X,y):            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            X_train, X_test = X_train[selectcol], X_test[selectcol]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf.fit(X_train,y_train, eval_set = [(X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=200)
            y_pred_train+=clf.predict_proba(dataset1[selectcol],num_iteration=clf.best_iteration_)[:,1]
            y_pred_test+=clf.predict_proba(X_sub,num_iteration=clf.best_iteration_)[:,1]

        y_pred_train /= 5.0#pay attention
        y_pred_test /= 5.0#pay attention
        
        yPredTrain.append(y_pred_train)
        yPredTest.append(y_pred_test)
        print('i')
    return yPredTrain,yPredTest  
#%%
"""
用某一次随机分出的训练集和测试集找出 最佳迭代次数
然后 用全部训练集（这里的全部指的是 从 总训练集中抽出的）重新训练
"""
def genBasePred(clf,selectcol,totaltestSort,dataset1,dataset2):
    yPredTrain=[]
    yPredTest=[]

    for i in totaltestSort:
        dataset=dataset1.sample(frac=0.3,random_state=i)
        datasetTest = dataset2
        X=dataset
        y=dataset['is_trade'] 
        X_sub=datasetTest
        y_pred_train=[0 for __ in range(dataset1.shape[0])]
        y_pred_test=[0 for __ in range(datasetTest.shape[0])]
        train_set = []
        test_set = []
        for train_index,test_index in sfolder.split(X,y): 
            train_set.append(train_index)
            test_set.append(test_index)
            
        X_train, X_test = X.iloc[train_set[0]], X.iloc[test_set[0]]
        X_train, X_test = X_train[selectcol], X_test[selectcol]
        y_train, y_test = y.iloc[train_set[0]], y.iloc[test_set[0]]
        clf.fit(X_train,y_train, eval_set = [(X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=200)
        iterNum=clf.best_iteration_
        clf.fit(X[selectcol],y, verbose=False,num_iteration=iterNum)
        
        y_pred_train=clf.predict_proba(dataset1[selectcol])[:,1]
        y_pred_test=clf.predict_proba(X_sub)[:,1]
        
        yPredTrain.append(y_pred_train)
        yPredTest.append(y_pred_test)
        print('i')
    return yPredTrain,yPredTest    
#%%    
#拿yPred 去进行LR或线性组合
yPredTrain,yPredTest=genBasePred(model[modelSelect],selectcol,totaltestSort,dataset1,dataset2)
#%%
yPredTrain=np.array(yPredTrain)
yPredTrain=yPredTrain.T

yPredTest=np.array(yPredTest)
yPredTest=yPredTest.T
#交叉验证
X=pd.DataFrame(yPredTrain).squeeze()
y=pd.DataFrame(dataset1['is_trade']).squeeze()
sfolder = StratifiedKFold(n_splits=5,random_state=2018,shuffle=False)
clf=lgbm.LGBMClassifier(objective='binary',
                        #min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.05,
                        n_estimators=3000,
                        colsample_bytree = 0.9,
                        subsample = 0.9,
                        seed=0
                        )
test=CrossValidation(X,y,clf,list(X.columns))
#重新训练 用全部训练集训练 ，最佳迭代次数如何求出？
stacker=lgbm.LGBMClassifier(objective='binary',
                        #min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.01,
                        n_estimators=1000,
                        colsample_bytree = 0.9,
                        subsample = 0.9,
                        seed=0
                        )
stacker.fit(yPredTrain, dataset1['is_trade'])
score=LossFunction(dataset1['is_trade'],stacker.predict(yPredTrain))
#%%
"""
LGBM回归模型
寻找最佳参数
"""
from sklearn.grid_search import GridSearchCV
params={'learning_rate':np.linspace(0.05,0.25,5), 
        'max_depth':[x for x in range(3,5,1)],
        'subsample':np.linspace(0.7,1,4)}
        #'n_estimators':[x for x in range(50,100,10)]}

clf =lgbm.LGBMClassifier(objective='binary',
                        #min_child_samples=100,
                        max_depth=3,
                        learning_rate=0.05,
                        n_estimators=3000,
                        colsample_bytree = 0.9,
                        subsample = 0.9,
                        seed=0
                        )
grid = GridSearchCV(clf, params, cv=5, scoring="neg_log_loss")
grid.fit(yPredTrain, dataset1['is_trade'])
#%%
grid.best_score_    #查看最佳分数(此处为f1_score)
grid.best_params_   #查看最佳参数
#获取最佳模型
grid.best_estimator_
#用最佳模型进行预测
best_model=grid.best_estimator_
predict_y=best_model.predict_proba(yPredTest,num_iteration=best_model.best_iteration_)[:,1]
#score=LossFuncetion(dataset1['is_trade'],best_model.predict_proba(X_sub,num_iteration=clf.best_iteration_)[:,1])
#%%
"""
最简单线性回归模型
"""
from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(yPredTrain, dataset1['is_trade'])
# Make predictions using the testing set
diabetes_y_pred = regr.predict(yPredTest)

score=LossFunction(dataset1['is_trade'],regr.predict(yPredTrain))







   