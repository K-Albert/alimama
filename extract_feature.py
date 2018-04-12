# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:24:55 2018

@author: lenovo
"""

import time
import pandas as pd
import os 
import numpy as np
os.getcwd() #get current working directory
os.chdir('F:\\006@天池\\0003@阿里妈妈')#change working directory
#%%
train = pd.read_csv('data/round1_ijcai_18_train_20180301.txt',sep=" ")
train = train.drop_duplicates(['instance_id'])
train = train.reset_index(drop=True)

test_a = pd.read_csv('data/round1_ijcai_18_test_a_20180301.txt',sep=" ")
def time2cov(time_):
    '''
    时间是根据天数推移，所以日期为脱敏，但是时间本身不脱敏
    :param time_:
    :return:
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))
train['real_time'] = pd.to_datetime(train['context_timestamp'].apply(time2cov))
train['real_hour'] = train['real_time'].dt.hour
train['real_day'] = train['real_time'].dt.day

test_a['real_time'] = pd.to_datetime(test_a['context_timestamp'].apply(time2cov))
test_a['real_hour'] = test_a['real_time'].dt.hour
test_a['real_day'] = test_a['real_time'].dt.day
#18 19 20 21 22
feature1=train[train['real_day']<23]
dataset1=train[train['real_day']==23]
#19 20 21 22 23
feature2=train[(train['real_day']<24)&(train['real_day']>18)]
dataset2=train[train['real_day']==24]
#20 21 22 23 24
feature3=train[(train['real_day']<25)&(train['real_day']>19)]
dataset3=test_a
#%% 为xgboost编写自定义损失函数
def c_log_loss(y_t,y_p):
    tmp = np.array(y_t) * np.log(np.array(y_p)) + (1 - np.array(y_t)) * np.log(1 - np.array(y_p))
    return -np.sum(tmp)/len(y_t),False
#%%
#注意存在三级category但是比较少
#所有商品一级分类相同
def splitItemCategory_second(s):
    s=str(s)
    s=s.split(';')
    return s[1]
def itemPropertyCnt(s):
    s=str(s)
    s=s.split(';')
    return s.size

"""
商品的二级类目
该二级类目下商品的 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级
该广告商品出现过多少次
该广告商品成交过多少次 item_id
该广告商品 收藏次数等级/展示次数等级
该广告商品 销量等级/展示次数等级
该广告商品 销量等级/收藏等级 如果被除数为0，则用0.1替代？
不同价格等级的成交率
不同销量等级的成交率
不同收藏等级的成交率
销量等级、收藏等级、价格等级的和
不同城市的成交率
不同品牌的成交率
商品有几个属性值
不同属性个数 的成交率
不同属性个数 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级
#某属性的成交率（很麻烦）
"""
feature= feature1   
def extract_merchant_feature(feature):
    merchant=feature[['instance_id','item_id','item_category_list','item_property_list','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','is_trade']]
    merchant['second_category']=merchant['item_category_list'].apply(splitItemCategory_second)   
    d=merchant[['second_category','item_price_level','item_sales_level','item_collected_level','item_pv_level','is_trade']]
    d=d[d['is_trade']==1]
    








