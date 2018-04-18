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
#%%
trainData = pd.read_csv('data/round1_ijcai_18_train_20180301.txt',sep=" ")
trainData = trainData.drop_duplicates(['instance_id'])
trainData = trainData.reset_index(drop=True)

testData_a = pd.read_csv('data/round1_ijcai_18_test_a_20180301.txt',sep=" ")
testData_b = pd.read_csv('data/round1_ijcai_18_test_b_20180418.txt',sep=" ")

testData=pd.concat([testData_a,testData_b])
#%%
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
#  每小时的购买转换率
d=trainData[['hour','is_trade']]
d['cnt']=1
d=d.groupby('hour').agg('sum').reset_index()
d['rate']=d['is_trade']/d['cnt']
trainData_t = pd.merge(trainData,d,on='hour',how='left')

g = sns.FacetGrid(trainData_t, col="day")
g.map(sns.barplot, 'hour', 'rate')
sns.factorplot(x='hour', y='rate', col='day', data=trainData_t)

plt.figure()
sns.barplot(x='day', y='rate', data=trainData_t)
###########
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


#%%%


dataset1=trainData[trainData['day']==23]
dataset2=trainData[trainData['day']==24]
#dataset3=test_a
feature1=trainData[trainData['day']==18]
feature2=trainData[trainData['day']==19]
feature3=trainData[trainData['day']==20]
feature4=trainData[trainData['day']==21]
feature5=trainData[trainData['day']==22]
feature6=trainData[trainData['day']==23]
feature7=trainData[trainData['day']==24]
feature1_2_3_4_5=pd.concat([feature1,feature2,feature3,feature4,feature4],ignore_index=True)
feature2_3_4_5_6=pd.concat([feature2,feature3,feature4,feature5,feature6],ignore_index=True)
feature3_4_5_6_7=pd.concat([feature3,feature4,feature5,feature6,feature7],ignore_index=True)
label=dataset1
feature=feature1_2_3_4_5
# 当前 用户 浏览的商品的价格等级 是否高于 用户曾购买的平均价格等级 能用 要处理 Nan后能用
d1=feature[['user_id','item_price_level','is_trade']]
d1=d1[d1['is_trade']==1]
d1=d1.drop('is_trade',axis=1)
d1=round(d1.groupby('user_id').agg('mean').reset_index())

d1=d1.rename(columns={'item_price_level':'item_price_level_mean'})
d2=label[['user_id','item_price_level','is_trade']]
d2=pd.merge(d2,d1,on='user_id',how='left')
d2['bigger']=(d2['item_price_level']<=d2['item_price_level_mean']).astype('int')

d2=d2.fillna(100)
d2['bigger'][d2[d2['item_price_level_mean']==100].index]=100

plt.figure()
sns.barplot(x="bigger", y="is_trade", data=d2);

# 当前 用户 浏览的 item_sales_level  不处理缺失能用
d1=feature[['user_id','item_sales_level','is_trade']]
d1=d1[d1['is_trade']==1]
d1=d1.drop('is_trade',axis=1)
d1=round(d1.groupby('user_id').agg('mean').reset_index())

d1=d1.rename(columns={'item_sales_level':'item_sales_level_mean'})
d2=label[['user_id','item_sales_level','is_trade']]
d2=pd.merge(d2,d1,on='user_id',how='left')
d2['bigger']=(d2['item_sales_level']>=d2['item_sales_level_mean']).astype('int')

plt.figure()
sns.barplot(x="bigger", y="is_trade", data=d2);

#当前 用户 浏览的 item_collected_level  不处理缺失能用
d1=feature[['user_id','item_collected_level','is_trade']]
d1=d1[d1['is_trade']==1]
d1=d1.drop('is_trade',axis=1)
d1=round(d1.groupby('user_id').agg('mean').reset_index())

d1=d1.rename(columns={'item_collected_level':'item_sales_level_mean'})
d2=label[['user_id','item_collected_level','is_trade']]
d2=pd.merge(d2,d1,on='user_id',how='left')
d2['bigger']=(d2['item_collected_level']>=d2['item_sales_level_mean']).astype('int')

plt.figure()
sns.barplot(x="bigger", y="is_trade", data=d2);

# 当前用户浏览的商店评价 等级 是shop_score_service

d1=feature[['user_id','shop_score_service','is_trade']]
d1=d1[d1['is_trade']==1]
d1=d1.drop('is_trade',axis=1)
d1=d1.groupby('user_id').agg('mean').reset_index()
d1=d1.rename(columns={'shop_score_service':'shop_review_num_level_mean'})
d2=label[['user_id','shop_score_service','is_trade']]
d2=pd.merge(d2,d1,on='user_id',how='left')
d2['bigger']=(d2['shop_score_service']>=d2['shop_review_num_level_mean']).astype('int')
plt.figure()
sns.barplot(x="bigger", y="is_trade", data=d2);

#不同城市的成交转换率
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
trainData['item_city_id'] = le.fit_transform(trainData['item_city_id'])
d1=trainData[['item_city_id','is_trade']]
d1['cnt']=1
d1=d1.groupby('item_city_id').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['item_city_id'] ,y=d1['rate']) 

# 不同商品 评价等级的成交转换率
d1=trainData[['item_price_level','is_trade']]
d1['cnt']=1
d1=d1.groupby('item_price_level').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['item_price_level'] ,y=d1['rate']) 

# 不同 商品 销量等级的成交转换率
d1=trainData[['item_sales_level','is_trade']]
d1['cnt']=1
d1=d1.groupby('item_sales_level').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['item_sales_level'] ,y=d1['rate']) 
# 不同 商品 收藏次数 成交转换率

d1=trainData[['item_collected_level','is_trade']]
d1['cnt']=1
d1=d1.groupby('item_collected_level').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['item_collected_level'] ,y=d1['rate']) 

# 不同 商品 item_pv_level 成交转换率   没什么用

d1=trainData[['item_pv_level','is_trade']]
d1['cnt']=1
d1=d1.groupby('item_pv_level').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['item_pv_level'] ,y=d1['rate']) 

#context_page_id

d1=trainData[['context_page_id','is_trade']]
d1['cnt']=1
d1=d1.groupby('context_page_id').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['context_page_id'] ,y=d1['rate']) 

#shop_review_num_level

d1=trainData[['shop_review_num_level','is_trade']]
d1['cnt']=1
d1=d1.groupby('shop_review_num_level').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['shop_review_num_level'] ,y=d1['rate']) 
#shop_star_level  影响不大
d1=trainData[['shop_star_level','is_trade']]
d1['cnt']=1
d1=d1.groupby('shop_star_level').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['shop_star_level'] ,y=d1['rate']) 
# 不同性别 user_gender_id
d1=trainData[['user_gender_id','is_trade']]
d1['cnt']=1
d1=d1.groupby('user_gender_id').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['user_gender_id'] ,y=d1['rate']) 

#user_age_level

d1=trainData[['user_age_level','is_trade']]
d1['cnt']=1
d1=d1.groupby('user_age_level').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['user_age_level'] ,y=d1['rate']) 
sns.countplot(trainData.user_age_level)

#user_occupation_id

d1=trainData[['user_occupation_id','is_trade']]
d1['cnt']=1
d1=d1.groupby('user_occupation_id').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['user_occupation_id'] ,y=d1['rate']) 
sns.countplot(trainData.user_occupation_id)


#user_star_level  可以 分段

d1=trainData[['user_star_level','is_trade']]
d1['cnt']=1
d1=d1.groupby('user_star_level').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['user_star_level'] ,y=d1['rate']) 
sns.countplot(trainData.user_star_level)

#shop_review_positive_rate  不能分段 注意  画起来时间

d1=trainData[['shop_review_positive_rate','is_trade']]
d1['cnt']=1
d1=d1.groupby('shop_review_positive_rate').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['shop_review_positive_rate'] ,y=d1['rate']) 
#shop_score_service
d1=trainData[['shop_score_service','is_trade']]
d1['cnt']=1
d1=d1.groupby('shop_score_service').agg('sum').reset_index()
d1['rate']=d1['is_trade']/d1['cnt']

sns.pointplot(x=d1['shop_score_service'] ,y=d1['rate']) 

# 历史上 该小时 发生购买不同 用户  /所有的 发生购买的不同 用户
d1=trainData[['hour','user_id','is_trade']]

d2=d1[d1['is_trade']==1]
d2=d2[['user_id','is_trade']]
d2=d2.drop_duplicates()
cnt=d2.iloc[:,0].size

d1=d1.drop_duplicates()
d1=d1[d1['is_trade']==1]
d1=d1.groupby('hour').size().reset_index()
d1=d1.rename(columns={0:'dif_user'})
d1['rate']=d1['dif_user']/cnt
trainData_t=pd.merge(trainData,d1,on='hour',how='left')

#sns.pointplot(x=d1['hour'] ,y=d1['rate']) 
g = sns.FacetGrid(trainData_t, col="day")
g.map(sns.barplot, 'hour', 'rate')
sns.factorplot(x='hour', y='rate', col='day', data=trainData_t)

##  每小时的购买转换率
d=trainData[['hour','is_trade']]
d['cnt']=1
d=d.groupby('hour').agg('sum').reset_index()
d['rate']=d['is_trade']/d['cnt']
trainData_t = pd.merge(trainData,d,on='hour',how='left')

g = sns.FacetGrid(trainData_t, col="day")
g.map(sns.barplot, 'hour', 'rate')
sns.factorplot(x='hour', y='rate', col='day', data=trainData_t)

plt.figure()
sns.barplot(x='day', y='rate', data=trainData_t)
#每天 是否不同 有多大不同
d=trainData[['hour','is_trade','day']]
d['cnt']=1
d=d.groupby(['hour','day']).agg('sum').reset_index()
d['rate']=d['is_trade']/d['cnt']
trainData_t = pd.merge(trainData,d,on=['hour','day'],how='left')

g = sns.FacetGrid(trainData_t, col="day")
g.map(sns.barplot, 'hour', 'rate')
sns.factorplot(x='hour', y='rate', col='day', data=trainData_t)

plt.figure()
sns.barplot(x='day', y='rate', data=trainData_t)
#历史上 小时和user_gender组合
d=trainData[['user_gender_id','hour','is_trade']]
d['cnt']=1
d=d.groupby(['user_gender_id','hour']).agg('sum').reset_index()
d['rate']=d['is_trade']/d['cnt']

g = sns.FacetGrid(d, col="user_gender_id")
g.map(sns.barplot, 'hour', 'rate')
sns.factorplot(x='hour', y='rate', col='user_gender_id', data=d)

 
plt.figure()
sns.pointplot(x=d[d['user_gender_id']==0]['hour'], y=d[d['user_gender_id']==0]['rate'], normed=True, color="#FF0000", alpha=.9)  
sns.pointplot(x=d[d['user_gender_id']==1]['hour'], y=d[d['user_gender_id']==1]['rate'], normed=True, color="#C1F320", alpha=.5) 
sns.pointplot(x=d[d['user_gender_id']==2]['hour'], y=d[d['user_gender_id']==2]['rate'], normed=True, color="#FFF0F5", alpha=.5) 
sns.pointplot(x=d[d['user_gender_id']==-1]['hour'], y=d[d['user_gender_id']==-1]['rate'], normed=True, color="#FFFF00", alpha=.5) 












