# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:17:23 2018

@author: lenovo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

trainData = pd.read_csv('data/round1_ijcai_18_train_20180301.txt',sep=" ")
trainData = trainData.drop_duplicates(['instance_id'])
trainData = trainData.reset_index(drop=True)

testData = pd.read_csv('data/round1_ijcai_18_test_a_20180301.txt',sep=" ")
def time2cov(time_):
    '''
    时间是根据天数推移，所以日期为脱敏，但是时间本身不脱敏
    :param time_:
    :return:
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))
def splitPredicCategory(s):
    s=s.split(';')
    category_list=[]
    for i in s:
        category_list.append(i.split(':')[0])
    category= ';'.join(category_list)
    return category
def splitPredicProperty(s):
    s=s.split(';')
    property_list=[]
    a=1
    for i in s:
        property_split=i.split(':')
        if len(property_split)==1:
            a=1
        else:
            property_list.append(property_split[1])
    property= ','.join(property_list) 
    return property
def predictPropertyRight(s):
    s=s.split(':')
    s1=s[0].split(';')
    s2=s[1].split(',')
    property_num=len(s1)
    predict_right_num=0
    for i in s2:
        if i in s1:
            predict_right_num=predict_right_num+1
    return predict_right_num/property_num
def predictCategoryRight(s):
    s=s.split(':')
    s1=s[0].split(';')
    s2=s[1].split(';')
    category_num=len(s1)
    predict_right_num=0
    for i in s2:
        if i in s1:
            predict_right_num=predict_right_num+1
    return predict_right_num/category_num   
def predictPropertyRightNum(s):
    s=s.split(':')
    s1=s[0].split(';')
    s2=s[1].split(',')
    property_num=len(s1)
    predict_right_num=0
    for i in s2:
        if i in s1:
            predict_right_num=predict_right_num+1
    return predict_right_num  
def map_hour(x):
    if (x>=7)&(x<=12):
        return 1
    elif (x>=13)&(x<=18):
        return 2
    elif ((x>=19)&(x<=23))|(x==0):
        return 3
    else:
        return 4
#def map_age(x):
#    if x    

#%%
trainData['context_timestamp'] =pd.to_datetime(trainData['context_timestamp'].apply(time2cov))
testData['context_timestamp'] = pd.to_datetime(testData['context_timestamp'].apply(time2cov))
trainData.insert(loc=0, column='day', value=trainData.context_timestamp.dt.day)
trainData.insert(loc=0, column='hour', value=trainData.context_timestamp.dt.hour)

testData.insert(loc=0, column='day', value=testData.context_timestamp.dt.day)
testData.insert(loc=0, column='hour', value=testData.context_timestamp.dt.hour)

num1 = trainData[trainData.is_trade == 1].is_trade.count()
num0 = trainData[trainData.is_trade == 0].is_trade.count()

print('-----'*5)
print('为1的比例: ', num1/(num1+num0))
print('-----'*5)
print('train maxTime: ', trainData['context_timestamp'].max())
print('train minTime: ', trainData['context_timestamp'].min())
print('test minTime: ', testData['context_timestamp'].min())
print('test maxTime: ', testData['context_timestamp'].max())

print('-----'*5)
print('trainData cat:')
for col in trainData.columns:
    if 'id' in col:
        print(col, '\tnum: ', len(trainData[trainData.day==24][col].unique()))
print('-----'*5)
print('testData cat:')
for col in testData.columns:
    if 'id' in col:
        print(col, '\tnum: ', len(testData[col].unique()))

print('-----'*5)
print('rData cat:')
rData = trainData[trainData.day==24].sample(frac=0.33)
for col in trainData.columns:
    if 'id' in col:
        print(col, '\tnum: ', len(rData[col].unique()))

print('对比上面的这三个结果， 我们可以推断出测试集是由于最后一天的随机采样得到')
#%%
alldata = trainData.append(testData)

plt.figure()
sns.countplot(alldata.day)

g = sns.FacetGrid(alldata, col="day")
g.map(sns.distplot, 'hour')

g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'hour', 'is_trade')

sns.factorplot(x='hour', y='is_trade', col='day', data=trainData)
sns.factorplot(x='context_page_id', y='is_trade', data=trainData)

plt.figure()
sns.barplot(x='day', y='is_trade', data=trainData)

plt.figure()
sns.countplot(alldata.hour)


plt.figure()
sns.countplot(trainData[trainData['is_trade']==1].hour)


plt.figure(figsize=(10,5))
sns.barplot(x="hour", y="is_trade", data=trainData);


featuresUsed = ['item_price_level', 'item_sales_level', 'item_collected_level',
                'item_pv_level', 'user_age_level', 'user_star_level',
                'shop_review_num_level', 'shop_review_positive_rate',
                'shop_star_level', 'shop_score_service', 'shop_score_delivery',
                'shop_score_description', 'is_trade']
corrmat = trainData[featuresUsed].corr()
plt.figure()
sns.heatmap(corrmat)

def getStringVal(s, num):
    for i in range(num-1):
        if ';' in s:
            pos = s.index(';')
            s = s[pos+1:]
        else:
            return -1
    if ';' in s:
        pos = s.index(';')
        s = s[:pos]
    return s


#trainData['item_category_list'].apply(lambda x: getStringVal(x,1)).unique()

#trainData['item_category_list'].apply(lambda x: getStringVal(x, 2)).unique()

#trainData['item_category_list'].head()


trainData['item_category_list'].apply(lambda x: getStringVal(x, 1)).head()

trainData['item_cat_id'] = trainData['item_category_list'].apply(lambda x: getStringVal(x, 2))
trainData['item_cat_len'] = trainData['item_category_list'].apply(lambda x: x.count(';'))
trainData['item_cat2_id'] = trainData['item_category_list'].apply(lambda x: getStringVal(x, 3))

trainData['item_cat2_id'].unique()
trainData['item_cat_id'] .unique()

plt.figure()
sns.countplot(trainData.item_cat_len)

plt.figure()
g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'item_cat_len', 'is_trade')

plt.figure()
g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'item_cat_id', 'is_trade')

plt.figure()
g = sns.FacetGrid(trainData, col="day")
g.map(sns.barplot, 'item_cat2_id', 'is_trade')















