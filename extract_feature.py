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
    return len(s)   
#%%
"""
feature窗提取商品特征
该二级类目下商品的 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级（几个数据集的二级类目相同）
该二级类目下成功销售的商品的 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级（item_sales_level有缺失值 用0替代的  会影响最小值）
该广告商品出现过多少次（几个数据集item_id有没有重合）(feature3（8999）和feature2(9004)有8462重叠，feature1(9118)和feature2有8523重叠)
该广告商品成交过多少次 item_id

二级类目下商品数目
二级类目下商品销售成功
二级类目下商品转换率
二级类目下不同的商品数目

不同价格等级的成交率
不同销量等级的成交率
不同收藏等级的成交率
不同展示次数等级的成交率

不同城市的成交率
不同品牌的成交率

商品有几个属性值
不同属性个数 的成交率
(可拓展，价格+属性个数等等)

#销量等级、收藏等级、价格等级的和
该广告商品 收藏次数等级/展示次数等级
该广告商品 销量等级/展示次数等级
该广告商品 销量等级/收藏等级 如果被除数为0，则用0.1替代？
#某属性的成交率（很麻烦）
"""
#feature= feature1   
def extract_merchant_feature(feature):
    merchant=feature[['instance_id','item_id','item_category_list','item_property_list','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','is_trade']]
    merchant['second_category']=merchant['item_category_list'].apply(splitItemCategory_second)  
    merchant['item_property_cnt']=merchant.item_property_list.apply(itemPropertyCnt)     
#‘second_caregory’该二级类目下商品的 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级（几个数据集的二级类目相同）
    d=merchant[['second_category','item_price_level','item_sales_level','item_collected_level','item_pv_level']]
    d.replace(-1,0,inplace=True)
    d=d.groupby('second_category').agg(['mean','max','min']).reset_index()
    d.columns=['second_category','mean_item_price_level','max_item_price_level','min_item_price_level','mean_item_sales_level','max_item_sales_level','min_item_sales_level','mean_item_collected_level','max_item_collected_level','min_item_collected_level','mean_item_pv_level','max_item_pv_level','min_item_pv_level']
#‘second_caregory’该二级类目下成功销售的商品的 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级（item_sales_level有缺失值 用0替代的  会影响最小值）   
    d1=merchant[['second_category','item_price_level','item_sales_level','item_collected_level','item_pv_level','is_trade']]
    d1=d1[d1['is_trade']==1]
    d1=d1.drop('is_trade',axis=1)
    d1.replace(-1,0,inplace=True)
    d1=d1.groupby('second_category').agg(['mean','max','min']).reset_index()
    d1.columns=['second_category','trans_mean_item_price_level','trans_max_item_price_level','trans_min_item_price_level','trans_mean_item_sales_level','trans_max_item_sales_level','trans_min_item_sales_level','trans_mean_item_collected_level','trans_max_item_collected_level','trans_min_item_collected_level','trans_mean_item_pv_level','trans_max_item_pv_level','trans_min_item_pv_level']
#‘second_caregory’二和一的比
    d3=d[['second_category']]
    d3['mean_item_price_level_rate']=d1['trans_mean_item_price_level']/d['mean_item_price_level']
    d3['mean_item_sales_level_rate']=d1['trans_mean_item_sales_level']/d['mean_item_sales_level']
    d3['mean_item_collected_level_rate']=d1['trans_mean_item_collected_level']/d['mean_item_collected_level']
    d3['mean_item_pv_level_rate']=d1['trans_mean_item_pv_level']/d['mean_item_pv_level']
#‘second_caregory’二级类目下商品数目、二级类目下不同的商品数目
    d7=merchant[['second_category','item_id']]
    d8=d7.drop_duplicates()
    d8=d8.groupby('second_category').size().reset_index()
    d7=d7.groupby('second_category').size().reset_index()
    d7.rename(columns={0:'second_category_item_id_cnt'},inplace=True)
    d8.rename(columns={0:'second_category_dif_item_id_cnt'},inplace=True)
#‘second_caregory’二级类目下商品销售成功
    d9=merchant[['second_category','is_trade']]
    d9=d9.groupby('second_category').agg('sum').reset_index()
    d9.rename(columns={'is_trade':'second_category_trans_item_id_cnt'},inplace=True)
#‘second_caregory’二级类目下商品转换率
    d10=d9[['second_category']]
    d10['second_category_trans_item_id_rate']=d9['second_category_trans_item_id_cnt']/d7['second_category_item_id_cnt']   
#'item_id'该广告商品出现过多少次
    d4=merchant[['item_id']]
    d4=d4.groupby('item_id').size().reset_index()
    d4.rename(columns={0:'item_id_cnt'},inplace=True)
#'item_id'该广告商品成交多少次    
    d5=merchant[['item_id','is_trade']]
    d5=d5.groupby('item_id').agg('sum').reset_index()
    d5.rename(columns={'is_trade':'trans_item_id_cnt'},inplace=True)
#‘item_id’该商品的转换率
    d6=d5[['item_id']]
    d6['trans_item_id_rate']=d5['trans_item_id_cnt']/d4['item_id_cnt']
#'item_price_level'不同价格等级的数目
    d11=merchant[['item_price_level']]
    d11=d11.groupby('item_price_level').size().reset_index()
    d11.rename(columns={0:'dif_item_price_level_cnt'},inplace=True)    
#'item_price_level'不同价格等级的成交数目    
    d12=merchant[['item_price_level','is_trade']]
    d12=d12.groupby('item_price_level').agg('sum').reset_index()
    d12.rename(columns={'is_trade':'dif_trans_item_price_level_cnt'},inplace=True)       
#'item_price_level'不同价格等级的成交率
    d13=d12[['item_price_level']]
    d13['dif_trans_item_price_level_rate']=d12['dif_trans_item_price_level_cnt']/d11['dif_item_price_level_cnt']
#item_sales_level不同销量等级的数目！！！！！！！！！！！注意这里有缺失值
    d14=merchant[['item_sales_level']]
    d14=d14.groupby('item_sales_level').size().reset_index()
    d14.rename(columns={0:'dif_item_sales_level_cnt'},inplace=True)    
#item_sales_level不同销量等级的成交数目    
    d15=merchant[['item_sales_level','is_trade']]
    d15=d15.groupby('item_sales_level').agg('sum').reset_index()
    d15.rename(columns={'is_trade':'dif_trans_item_sales_level_cnt'},inplace=True)       
#item_sales_level不同销量等级的成交率
    d16=d15[['item_sales_level']]
    d16['dif_trans_item_sales_level_rate']=d15['dif_trans_item_sales_level_cnt']/d14['dif_item_sales_level_cnt']    
#'item_collected_level'不同收藏等级的数目
    d17=merchant[['item_collected_level']]
    d17=d17.groupby('item_collected_level').size().reset_index()
    d17.rename(columns={0:'dif_item_collected_level_cnt'},inplace=True)    
#'item_collected_level'不同收藏等级的成交数目    
    d18=merchant[['item_collected_level','is_trade']]
    d18=d18.groupby('item_collected_level').agg('sum').reset_index()
    d18.rename(columns={'is_trade':'dif_trans_item_collected_level_cnt'},inplace=True)       
#'item_collected_level'不同收藏等级的成交率
    d19=d18[['item_collected_level']]
    d19['dif_trans_item_collected_level_rate']=d18['dif_trans_item_collected_level_cnt']/d17['dif_item_collected_level_cnt']    
#'item_pv_level'不同展示次数等级的数目
    d20=merchant[['item_pv_level']]
    d20=d20.groupby('item_pv_level').size().reset_index()
    d20.rename(columns={0:'dif_item_pv_level_cnt'},inplace=True)    
#'item_pv_level'不同展示次数等级的成交数目    
    d21=merchant[['item_pv_level','is_trade']]
    d21=d21.groupby('item_pv_level').agg('sum').reset_index()
    d21.rename(columns={'is_trade':'dif_trans_item_pv_level_cnt'},inplace=True)       
#'item_pv_level'不同展示次数等级的成交率！！！！！！注意这里 基数很小会导致比列很大 存在偶然性
    d22=d21[['item_pv_level']]
    d22['dif_trans_item_pv_level_rate']=d21['dif_trans_item_pv_level_cnt']/d20['dif_item_pv_level_cnt']  
#item_city_id不同城市的出现次数
    d23=merchant[['item_city_id']]
    d23=d23.groupby('item_city_id').size().reset_index()
    d23.rename(columns={0:'item_city_id_cnt'},inplace=True)
#item_city_id不同城市的成交次数    
    d24=merchant[['item_city_id','is_trade']]
    d24=d24.groupby('item_city_id').agg('sum').reset_index()
    d24.rename(columns={'is_trade':'trans_item_city_id_cnt'},inplace=True)
#item_city_id不同城市的转换率   
    d25=d24[['item_city_id']]
    d25['trans_item_city_id_rate']=d24['trans_item_city_id_cnt']/d23['item_city_id_cnt']
#item_brand_id 不同品牌出现次数
    d26=merchant[['item_brand_id']]
    d26=d26.groupby('item_brand_id').size().reset_index()
    d26.rename(columns={0:'item_brand_id_cnt'},inplace=True)
#item_brand_id 不同品牌成交次数
    d27=merchant[['item_brand_id','is_trade']]
    d27=d27.groupby('item_brand_id').agg('sum').reset_index()
    d27.rename(columns={'is_trade':'trans_item_brand_id_cnt'},inplace=True)
#item_brand_id   不同品牌转换率
    d28=d27[['item_brand_id']]
    d28['trans_item_brand_id_rate']=d27['trans_item_brand_id_cnt']/d26['item_brand_id_cnt']    
#‘item_property_cnt’不同属性个数 的样例个数
    d29=merchant[['item_property_cnt','is_trade']]  
    d29=d29.groupby('item_property_cnt').size().reset_index()
    d29.rename(columns={0:'item_property_instance_cnt'},inplace=True)    
#‘item_property_cnt’不同属性个数 的成交个数   
    d30=merchant[['item_property_cnt','is_trade']]  
    d30=d30.groupby('item_property_cnt').agg('sum').reset_index()
    d30.rename(columns={'is_trade':'trans_item_property_instance_cnt'},inplace=True)
#‘item_property_cnt’不同属性个数 的转换率    
    d31=d30[['item_property_cnt']]
    d31['trans_item_property_instance_rate']=d30['trans_item_property_instance_cnt']/d29['item_property_instance_cnt']
#instance_id 该广告商品 收藏次数等级/展示次数等级
    d32=merchant[['instance_id','item_collected_level','item_pv_level']]
    d32['item_collected_pv_rate']=d32['item_collected_level']/d32['item_pv_level']
    d32=d32[['instance_id','item_collected_pv_rate']]
#instance_id 该广告商品 销量等级/展示次数等级
    d33=merchant[['instance_id','item_sales_level','item_pv_level']]
    d33['item_sale_pv_rate']=d33['item_sales_level']/d33['item_pv_level']
    d33=d33[['instance_id','item_sale_pv_rate']]
#instance_id 该广告商品 销量等级/收藏等级 如果被除数为0，则用0.1替代？    
    d34=merchant[['instance_id','item_sales_level','item_collected_level']]
    d34['item_sale_collected_rate']=d34['item_sales_level']/d34['item_collected_level']
    d34=d34[['instance_id','item_sale_collected_rate']]
#instance_id 该广告商品销量等级/价格等级
    d35=merchant[['instance_id','item_sales_level','item_price_level']]
    d35['item_sale_price_rate']=d35['item_sales_level']/d35['item_price_level']
    d35=d35[['instance_id','item_sale_price_rate']]    
#instance_id 该广告商品收藏等级/价格等级
    d36=merchant[['instance_id','item_collected_level','item_price_level']]
    d36['item_collected_price_rate']=d36['item_collected_level']/d36['item_price_level']
    d36=d36[['instance_id','item_collected_price_rate']] 
#组合起来
    merchant=pd.merge(merchant,d,on='second_category',how='left')
    merchant=pd.merge(merchant,d1,on='second_category',how='left')
    merchant=pd.merge(merchant,d3,on='second_category',how='left')
    merchant=pd.merge(merchant,d7,on='second_category',how='left')
    merchant=pd.merge(merchant,d8,on='second_category',how='left')
    merchant=pd.merge(merchant,d9,on='second_category',how='left')
    merchant=pd.merge(merchant,d10,on='second_category',how='left')

    merchant=pd.merge(merchant,d4,on='item_id',how='left')
    merchant=pd.merge(merchant,d5,on='item_id',how='left')
    merchant=pd.merge(merchant,d6,on='item_id',how='left')
    
    merchant=pd.merge(merchant,d11,on='item_price_level',how='left')
    merchant=pd.merge(merchant,d12,on='item_price_level',how='left')
    merchant=pd.merge(merchant,d13,on='item_price_level',how='left')
    
    merchant=pd.merge(merchant,d14,on='item_sales_level',how='left')
    merchant=pd.merge(merchant,d15,on='item_sales_level',how='left')
    merchant=pd.merge(merchant,d16,on='item_sales_level',how='left')
    
    merchant=pd.merge(merchant,d17,on='item_collected_level',how='left')
    merchant=pd.merge(merchant,d18,on='item_collected_level',how='left')
    merchant=pd.merge(merchant,d19,on='item_collected_level',how='left')
    
    merchant=pd.merge(merchant,d20,on='item_pv_level',how='left')
    merchant=pd.merge(merchant,d21,on='item_pv_level',how='left')
    merchant=pd.merge(merchant,d22,on='item_pv_level',how='left')
    
    merchant=pd.merge(merchant,d23,on='item_city_id',how='left')
    merchant=pd.merge(merchant,d24,on='item_city_id',how='left')
    merchant=pd.merge(merchant,d25,on='item_city_id',how='left')
    
    merchant=pd.merge(merchant,d26,on='item_brand_id',how='left')
    merchant=pd.merge(merchant,d27,on='item_brand_id',how='left')
    merchant=pd.merge(merchant,d28,on='item_brand_id',how='left')
    
    merchant=pd.merge(merchant,d29,on='item_property_cnt',how='left')
    merchant=pd.merge(merchant,d30,on='item_property_cnt',how='left')
    merchant=pd.merge(merchant,d31,on='item_property_cnt',how='left')
    
    merchant=pd.merge(merchant,d32,on='instance_id',how='left')
    merchant=pd.merge(merchant,d33,on='instance_id',how='left')
    merchant=pd.merge(merchant,d34,on='instance_id',how='left')
    merchant=pd.merge(merchant,d35,on='instance_id',how='left')
    merchant=pd.merge(merchant,d36,on='instance_id',how='left')
    return merchant
#%%
merchant_feature1=extract_merchant_feature(feature1)
merchant_feature2=extract_merchant_feature(feature2)
merchant_feature3=extract_merchant_feature(feature3)
 #%%   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
