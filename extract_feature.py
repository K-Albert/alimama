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

train['real_time'] = pd.to_datetime(train['context_timestamp'].apply(time2cov))
train['real_hour'] = train['real_time'].dt.hour
train['real_day'] = train['real_time'].dt.day
train['perdict_category']=train['predict_category_property'].apply(splitPredicCategory)
train['perdict_property']=train['predict_category_property'].apply(splitPredicProperty)

test_a['real_time'] = pd.to_datetime(test_a['context_timestamp'].apply(time2cov))
test_a['real_hour'] = test_a['real_time'].dt.hour
test_a['real_day'] = test_a['real_time'].dt.day
test_a['perdict_category']=test_a['predict_category_property'].apply(splitPredicCategory)
test_a['perdict_property']=test_a['predict_category_property'].apply(splitPredicProperty)
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
#浏览过该商品的不同用户数量
#消费过该商品的不同用户数量
"""
#feature= feature1   
def extract_merchant_feature(feature):
    merchant=feature[['instance_id','item_id','item_category_list','item_property_list','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','user_id','perdict_category','perdict_property','predict_category_property','is_trade']]
    merchant['second_category']=merchant['item_category_list'].apply(splitItemCategory_second)  
    merchant['item_property_cnt']=merchant.item_property_list.apply(itemPropertyCnt)     
#‘second_caregory’该二级类目下商品的 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级（几个数据集的二级类目相同）
    d=merchant[['second_category','item_price_level','item_sales_level','item_collected_level','item_pv_level']]
    d.replace(-1,0,inplace=True)
    d=round(d.groupby('second_category').agg(['mean','max','min']).reset_index())
    d.columns=['second_category','mean_item_price_level','max_item_price_level','min_item_price_level','mean_item_sales_level','max_item_sales_level','min_item_sales_level','mean_item_collected_level','max_item_collected_level','min_item_collected_level','mean_item_pv_level','max_item_pv_level','min_item_pv_level']
#‘second_caregory’该二级类目下成功销售的商品的 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级（item_sales_level有缺失值 用0替代的  会影响最小值）   
    d1=merchant[['second_category','item_price_level','item_sales_level','item_collected_level','item_pv_level','is_trade']]
    d1=d1[d1['is_trade']==1]
    d1=d1.drop('is_trade',axis=1)
    d1.replace(-1,0,inplace=True)
    d1=round(d1.groupby('second_category').agg(['mean','max','min']).reset_index())
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
#消费过该商家的不同用户个数
    d37=merchant[['item_id','user_id']]
    d37=d37.drop_duplicates()[['item_id']]
    d37=d37.groupby('item_id').size.reset_index()
    d37.rename(columns={0:'merchant_dif_user_buy'},inplace=True)
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
    merchant=pd.merge(merchant,d37,on='item_id',how='left')
#    merchant=pd.merge(merchant,d38,on='instance_id',how='left')    
    merchant=merchant.drop(['instance_id','item_category_list','item_property_list','item_brand_id','item_city_id','is_trade','user_id','predict_category','predict_property','predict_category_property'],axis=1)
    merchant=merchant.drop_duplicates()
    return merchant
#%%
merchant_feature1=extract_merchant_feature(feature1)
merchant_feature2=extract_merchant_feature(feature2)
merchant_feature3=extract_merchant_feature(feature3)
#%%   user
"""
提取用户信息
根据用户编号来：
该用户浏览过的商品数量
该用户成交了的商品数量
该用户的商品成交率

该用户浏览过的不同商品数量（item_id）
该用户消费过的不同商品数量
该用户消费过的不同商品数量/该用户浏览过的不同商品数量

该用户浏览不同品牌商品数量
该用户消费过的不同品牌数量

该用户浏览不同城市商品数量
该用户消费过的不同城市数量

该用户浏览不同价格等级商品数量
该用户消费的价格的平均等级

该用户浏览不同销量等级商品数量
该用户消费的销量的平均等级

该用户浏览不同收藏等级商品数量
该用户消费的商品的收藏的平均的等级

#该用户浏览不同展示次数商品数量

该用户浏览过的商店数量
该用户成交过的商店数量

该用户浏览不同的店铺数量

该用户消费的店铺的平均评价等级
该用户消费的店铺的平均好评等级
该用户消费的店铺的平均星级
该用户消费的店铺的平均服务态度
该用户消费的店铺的平均物流服务
该用户消费的店铺的平均描述相符度

#该用户浏览的店铺的平均好评率
#该用户浏览的店铺的平均服务态度
#该用户浏览的店铺的平均物流服务
#该用户浏览的店铺的平均描述相符度 

不同性别用户
{
成交率
消费商品平均价格等级
平均销量等级
平均收藏次数等级
平均展示次数等级
消费店铺平均
评价数量等级
好评率
星级
服务态度评分
物流服务评分
描述相符评分 注意除了成交率外 其余没有太大差别没有选用
}
不同年龄用户
不同职业用户
不同星级用户

用户活跃时间 

"""    
#feature=feature1
def extract_user_feature(feature):
    user=feature[['instance_id','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','item_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description','is_trade','real_hour']]
#'user_id' 该用户浏览过的商品数量   
    d=user[['user_id']]
    d=d.groupby('user_id').size().reset_index()
    d.rename(columns={0:'user_look_cnt'},inplace=True)    
#'user_id' 该用户成交了的商品数量
    d1=user[['user_id','is_trade']]
    d1=d1.groupby('user_id').agg('sum').reset_index()
    d1.rename(columns={'is_trade':'user_buy_cnt'},inplace=True)    
#'user_id' 用户成交率
    d2=d1[['user_id']]
    d2['user_buy_rate']=d1['user_buy_cnt']/d['user_look_cnt']       
#该用户浏览过的不同商品种类（item_id）
    d3=user[['user_id','item_id']]
    d3=d3.drop_duplicates()
    d3=d3.groupby('user_id').size().reset_index()
    d3.rename(columns={0:'user_look_dif_merchant_cnt'},inplace=True)         
#该用户消费过的不同商品种类
    d4=user[['user_id','item_id','is_trade']]
    d4=d4.drop_duplicates()
    d4=d4[['user_id','is_trade']]
    d4=d4.groupby('user_id').agg('sum').reset_index()
    d4.rename(columns={'is_trade':'user_buy_dif_merchant_cnt'},inplace=True) 
#该用户消费过的不同商品种类/该用户浏览过的不同商品种类    
    d5=d4[['user_id']]
    d5['user_buy_dif_merchant_rate']=d4['user_buy_dif_merchant_cnt']/d3['user_look_dif_merchant_cnt']   
#该用户浏览不同品牌商品数量
    d6=user[['user_id','item_brand_id']]
    d6=d6.drop_duplicates()
    d6=d6.groupby('user_id').size().reset_index()
    d6.rename(columns={0:'user_look_dif_merchant_brand_cnt'},inplace=True)     
#该用户消费不同品牌商品数量
    d7=user[['user_id','item_brand_id','is_trade']]
    d7=d7.drop_duplicates()
    d7=d7[['user_id','is_trade']]
    d7=d7.groupby('user_id').agg('sum').reset_index()
    d7.rename(columns={'is_trade':'user_buy_dif_merchant_brand_cnt'},inplace=True) 
#该用户不同品牌转换比
    d8=d7[['user_id']]
    d8['user_buy_dif_merchant_brand_rate']=d7['user_buy_dif_merchant_brand_cnt']/d6['user_look_dif_merchant_brand_cnt']      
#该用户浏览不同城市商品 城市种类
    d9=user[['user_id','item_city_id']]
    d9=d9.drop_duplicates()
    d9=d9.groupby('user_id').size().reset_index()
    d9.rename(columns={0:'user_look_dif_merchant_city_cnt'},inplace=True) 
#该用户 购买不同城市种类
    d10=user[['user_id','item_city_id','is_trade']]
    d10=d10.drop_duplicates()
    d10=d10[['user_id','is_trade']]
    d10=d10.groupby('user_id').agg('sum').reset_index()
    d10.rename(columns={'is_trade':'user_buy_dif_merchant_city_cnt'},inplace=True) 
#该用户 购买城市 转换比    
    d11=d10[['user_id']]
    d11['user_buy_dif_merchant_city_rate']=d10['user_buy_dif_merchant_city_cnt']/d9['user_look_dif_merchant_city_cnt']          
#该用户浏览不同价格等级商品数量
    d12=user[['user_id','item_price_level']]
    d12=d12.drop_duplicates()
    d12=d12.groupby('user_id').size().reset_index()
    d12.rename(columns={0:'user_look_dif_merchant_price_cnt'},inplace=True)         
#该用户消费的价格的不同等级种类
    d13=user[['user_id','item_price_level','is_trade']]
    d13=d13.drop_duplicates()
    d13=d13[['user_id','is_trade']]
    d13=d13.groupby('user_id').agg('sum').reset_index()
    d13.rename(columns={'is_trade':'user_buy_dif_merchant_price_cnt'},inplace=True) 
#该用户浏览过 商品 平均价格等级
    d14=user[['user_id','item_price_level']]
    d14=round(d14.groupby('user_id').agg('mean').reset_index())
    d14.columns=['user_id','user_look_mean_merchant_price_level']   
#该用户消费的 商品 平均价格等级
    d15=user[['user_id','item_price_level','is_trade']]
    d15=d15[d15['is_trade']==1]
    d15=d15[['user_id','item_price_level']]
    d15=round(d15.groupby('user_id').agg('mean').reset_index())
    d15.columns=['user_id','user_buy_mean_merchant_price_level'] 
    d15=pd.merge(d14[['user_id']],d15,on='user_id',how='left')  
    d15=d15.fillna(value=-1)    
#该用户浏览不同销量等级商品数量
    d16=user[['user_id','item_sales_level']]
    d16=d16.drop_duplicates()
    d16=d16.groupby('user_id').size().reset_index()
    d16.rename(columns={0:'user_look_dif_merchant_sales_cnt'},inplace=True)    
#该用户消费的销量的不同等级种类    
    d17=user[['user_id','item_sales_level','is_trade']]
    d17=d17.drop_duplicates()
    d17=d17[['user_id','is_trade']]
    d17=d17.groupby('user_id').agg('sum').reset_index()
    d17.rename(columns={'is_trade':'user_buy_dif_merchant_sales_cnt'},inplace=True)    
#该用户浏览过 商品 平均销售等级
    d18=user[['user_id','item_sales_level']]
    d18=round(d18.groupby('user_id').agg('mean').reset_index())
    d18.columns=['user_id','user_look_mean_merchant_sales_level']   
#该用户消费的 商品 平均销售等级
    d19=user[['user_id','item_price_level','is_trade']]
    d19=d19[d19['is_trade']==1]
    d19=d19[['user_id','item_price_level']]
    d19=round(d19.groupby('user_id').agg('mean').reset_index())
    d19.columns=['user_id','user_buy_mean_merchant_sales_level'] 
    d19=pd.merge(d18[['user_id']],d19,on='user_id',how='left')  
    d19=d19.fillna(value=-1)    
#该用户浏览不同收藏等级商品数量
    d20=user[['user_id','item_collected_level']]
    d20=d20.drop_duplicates()
    d20=d20.groupby('user_id').size().reset_index()
    d20.rename(columns={0:'user_look_dif_merchant_collected_cnt'},inplace=True)    
#该用户消费的收藏的不同等级种类    
    d21=user[['user_id','item_collected_level','is_trade']]
    d21=d21.drop_duplicates()
    d21=d21[['user_id','is_trade']]
    d21=d21.groupby('user_id').agg('sum').reset_index()
    d21.rename(columns={'is_trade':'user_buy_dif_merchant_collected_cnt'},inplace=True)    
#该用户浏览过 商品 平均收藏等级
    d22=user[['user_id','item_collected_level']]
    d22=round(d22.groupby('user_id').agg('mean').reset_index())
    d22.columns=['user_id','user_look_mean_merchant_collected_level']   
#该用户消费的 商品 平均收藏等级
    d23=user[['user_id','item_collected_level','is_trade']]
    d23=d23[d23['is_trade']==1]
    d23=d23[['user_id','item_collected_level']]
    d23=round(d23.groupby('user_id').agg('mean').reset_index())
    d23.columns=['user_id','user_buy_mean_merchant_collected_level'] 
    d23=pd.merge(d22[['user_id']],d19,on='user_id',how='left')  
    d23=d23.fillna(value=-1) 
#该用户浏览过的商店数量
    d24=user[['user_id','shop_id']]
    d24=d24.groupby('user_id').size().reset_index()
    d24.rename(columns={0:'user_look_shop_cnt'},inplace=True)     
#用户浏览过的不同 商店种类
    d25=user[['user_id','shop_id']]
    d25=d25.drop_duplicates()
    d25=d25.groupby('user_id').size().reset_index()
    d25.rename(columns={0:'user_look_dif_shop_cnt'},inplace=True)
#用户成交过的不同 商店种类    
    d26=user[['user_id','shop_id','is_trade']]
    d26=d26.drop_duplicates()
    d26=d26[['user_id','is_trade']]
    d26=d26.groupby('user_id').agg('sum').reset_index()
    d26.rename(columns={'is_trade':'user_buy_dif_shop_cnt'},inplace=True)    
#该用户消费过的不同商店种类/该用户浏览过的不同商店种类    
    d27=d26[['user_id']]
    d27['user_buy_dif_shop_rate']=d26['user_buy_dif_shop_cnt']/d25['user_look_dif_shop_cnt']        
#该用户浏览不同评价等级
    d28=user[['user_id','shop_review_num_level']]
    d28=d28.drop_duplicates()
    d28=d28.groupby('user_id').size().reset_index()
    d28.rename(columns={0:'user_look_dif_shop_review_num_level_cnt'},inplace=True)    
#该用户消费的店铺 不同评价等级种类   
    d29=user[['user_id','shop_review_num_level','is_trade']]
    d29=d29.drop_duplicates()
    d29=d29[['user_id','is_trade']]
    d29=d29.groupby('user_id').agg('sum').reset_index()
    d29.rename(columns={'is_trade':'user_buy_dif_shop_review_num_level_cnt'},inplace=True)    
#该用户浏览过 店铺 平均评价数量等级
    d30=user[['user_id','shop_review_num_level']]
    d30=round(d30.groupby('user_id').agg('mean').reset_index())
    d30.columns=['user_id','user_look_mean_shop_review_num_level']   
#该用户消费的 商品 平均评价数量等级
    d31=user[['user_id','shop_review_num_level','is_trade']]
    d31=d31[d31['is_trade']==1]
    d31=d31[['user_id','shop_review_num_level']]
    d31=round(d31.groupby('user_id').agg('mean').reset_index())
    d31.columns=['user_id','user_buy_mean_shop_review_num_level'] 
    d31=pd.merge(d30[['user_id']],d31,on='user_id',how='left')  
    d31=d31.fillna(value=-1)    
#该用户浏览不同商店 星级
    d32=user[['user_id','shop_star_level']]
    d32=d32.drop_duplicates()
    d32=d32.groupby('user_id').size().reset_index()
    d32.rename(columns={0:'user_look_dif_shop_star_level_cnt'},inplace=True)    
#该用户消费的店铺 不同星级种类  
    d33=user[['user_id','shop_star_level','is_trade']]
    d33=d33.drop_duplicates()
    d33=d33[['user_id','is_trade']]
    d33=d33.groupby('user_id').agg('sum').reset_index()
    d33.rename(columns={'is_trade':'user_buy_dif_shop_star_level_cnt'},inplace=True)    
#该用户浏览过 店铺 平均星级
    d34=user[['user_id','shop_star_level']]
    d34=round(d34.groupby('user_id').agg('mean').reset_index())
    d34.columns=['user_id','user_look_mean_shop_star_level']   
#该用户消费的 店铺 平均星级
    d35=user[['user_id','shop_star_level','is_trade']]
    d35=d35[d35['is_trade']==1]
    d35=d35[['user_id','shop_star_level']]
    d35=round(d35.groupby('user_id').agg('mean').reset_index())
    d35.columns=['user_id','user_buy_mean_shop_star_level'] 
    d35=pd.merge(d34[['user_id']],d35,on='user_id',how='left')  
    d35=d35.fillna(value=-1)
#该用户消费的店铺的平均好评率
    d36=user[['user_id','shop_review_positive_rate','is_trade']]
    d36=d36[d36['is_trade']==1]
    d36=d36[['user_id','shop_review_positive_rate']]
    d36=d36.groupby('user_id').agg('mean').reset_index()
    d36.columns=['user_id','user_buy_mean_shop_review_positive_rate'] 
    d36=pd.merge(d34[['user_id']],d36,on='user_id',how='left')  
    d36=d36.fillna(value=-1)   
#该用户消费的店铺的平均服务态度
    d37=user[['user_id','shop_score_service','is_trade']]
    d37=d37[d37['is_trade']==1]
    d37=d37[['user_id','shop_score_service']]
    d37=d37.groupby('user_id').agg('mean').reset_index()
    d37.columns=['user_id','user_buy_mean_shop_score_service'] 
    d37=pd.merge(d34[['user_id']],d37,on='user_id',how='left')  
    d37=d37.fillna(value=-1)      
#该用户消费的店铺的平均物流服务
    d38=user[['user_id','shop_score_delivery','is_trade']]
    d38=d38[d38['is_trade']==1]
    d38=d38[['user_id','shop_score_delivery']]
    d38=d38.groupby('user_id').agg('mean').reset_index()
    d38.columns=['user_id','user_buy_mean_shop_score_delivery'] 
    d38=pd.merge(d34[['user_id']],d38,on='user_id',how='left')  
    d38=d38.fillna(value=-1)      
#该用户消费的店铺的平均描述相符度    
    d39=user[['user_id','shop_score_description','is_trade']]
    d39=d39[d39['is_trade']==1]
    d39=d39[['user_id','shop_score_description']]
    d39=d39.groupby('user_id').agg('mean').reset_index()
    d39.columns=['user_id','user_buy_mean_shop_score_description'] 
    d39=pd.merge(d34[['user_id']],d39,on='user_id',how='left')  
    d39=d39.fillna(value=-1)      
# user_gender_id 不同性别用户 成交率
    d40=user[['user_gender_id','is_trade']]
    d40['cnt']=1
    d40=d40.groupby('user_gender_id').agg('sum').reset_index()
    d40['user_gender_buy_rate']=d40['is_trade']/d40['cnt']
    d40=d40[['user_gender_id','user_gender_buy_rate']]
# user_age_level 不同年龄用户  成交率
    d41=user[['user_age_level','is_trade']]
    d41['cnt']=1
    d41=d41.groupby('user_age_level').agg('sum').reset_index()
    d41['user_age_buy_rate']=d41['is_trade']/d41['cnt']
    d41=d41[['user_age_level','user_age_buy_rate']]    
    
    d42=user[['user_age_level','is_trade','shop_score_description']]
    d42=d42[d42['is_trade']==1][['user_age_level','shop_score_description']]
    d42=d42.groupby('user_age_level').agg('mean').reset_index()
    d42.columns=['user_age_level','user_age_buy_mean_shop_score_description']  
#user_occupation_id 不同职业用户   
    d43=user[['user_occupation_id','is_trade']]
    d43['cnt']=1
    d43=d43.groupby('user_occupation_id').agg('sum').reset_index()
    d43['user_occupation_buy_rate']=d43['is_trade']/d43['cnt']
    d43=d43[['user_occupation_id','user_occupation_buy_rate']] 
#user_star_level 不同星级用户成交率    
    d44=user[['user_star_level','is_trade']]
    d44['cnt']=1
    d44=d44.groupby('user_star_level').agg('sum').reset_index()
    d44['user_star_buy_rate']=d44['is_trade']/d44['cnt']
    d44=d44[['user_star_level','user_star_buy_rate']] 
    def sortFrequest(x):
       x=x.groupby('real_hour').size().reset_index() 
       x.rename(columns={0:'cnt'},inplace=True)
       x.sort_values(by = ['cnt'],axis = 0,ascending = False).reset_index(drop=True) 
       max_val=x['real_hour'][0]
       return max_val
        
    d45=user[['user_id','real_hour']]
    d45=d45.groupby('user_id').apply(sortFrequest).reset_index()
    d45.rename(columns={0:'user_buy_frequest_hour'},inplace=True)
    
    user=user.drop(['real_hour','instance_id','item_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description','is_trade'],axis=1)
    user=pd.merge(user,d,on='user_id',how='left')
    user=pd.merge(user,d1,on='user_id',how='left')
    user=pd.merge(user,d2,on='user_id',how='left')
    user=pd.merge(user,d3,on='user_id',how='left')
    user=pd.merge(user,d4,on='user_id',how='left')
    user=pd.merge(user,d5,on='user_id',how='left')
    user=pd.merge(user,d6,on='user_id',how='left')
    user=pd.merge(user,d7,on='user_id',how='left')
    user=pd.merge(user,d8,on='user_id',how='left')
    user=pd.merge(user,d9,on='user_id',how='left')
    user=pd.merge(user,d10,on='user_id',how='left')
    user=pd.merge(user,d11,on='user_id',how='left')
    user=pd.merge(user,d12,on='user_id',how='left')
    user=pd.merge(user,d13,on='user_id',how='left')
    user=pd.merge(user,d14,on='user_id',how='left')
    user=pd.merge(user,d15,on='user_id',how='left')
    user=pd.merge(user,d16,on='user_id',how='left')
    user=pd.merge(user,d17,on='user_id',how='left')
    user=pd.merge(user,d18,on='user_id',how='left')
    user=pd.merge(user,d19,on='user_id',how='left')
    user=pd.merge(user,d20,on='user_id',how='left')
    user=pd.merge(user,d21,on='user_id',how='left')
    user=pd.merge(user,d22,on='user_id',how='left')
    user=pd.merge(user,d23,on='user_id',how='left')
    user=pd.merge(user,d24,on='user_id',how='left')
    user=pd.merge(user,d25,on='user_id',how='left')
    user=pd.merge(user,d26,on='user_id',how='left')
    user=pd.merge(user,d27,on='user_id',how='left')
    user=pd.merge(user,d28,on='user_id',how='left')
    user=pd.merge(user,d29,on='user_id',how='left')
    user=pd.merge(user,d30,on='user_id',how='left')
    user=pd.merge(user,d31,on='user_id',how='left')
    user=pd.merge(user,d32,on='user_id',how='left')
    user=pd.merge(user,d33,on='user_id',how='left')
    user=pd.merge(user,d34,on='user_id',how='left')
    user=pd.merge(user,d35,on='user_id',how='left')
    user=pd.merge(user,d36,on='user_id',how='left')
    user=pd.merge(user,d37,on='user_id',how='left')
    user=pd.merge(user,d38,on='user_id',how='left')
    user=pd.merge(user,d39,on='user_id',how='left')
   
    user=pd.merge(user,d40,on='user_gender_id',how='left')
    user=pd.merge(user,d41,on='user_age_level',how='left')
    user=pd.merge(user,d42,on='user_age_level',how='left')
    user=pd.merge(user,d43,on='user_occupation_id',how='left')
    user=pd.merge(user,d44,on='user_star_level',how='left')
    user=pd.merge(user,d45,on='user_id',how='left')
    return user
#%%
user1=  extract_user_feature(feature1)  
user2=  extract_user_feature(feature2)    
user3=  extract_user_feature(feature3)    
user1.to_csv('data/user1.csv',index=None)
#%%    shop
"""
店铺特征：
店铺被浏览册数
店铺被消费 次数
店铺消费 次数/店铺浏览次数
店铺 浏览的 不同用户数
店铺 消费的 不同用户
店铺 商品 种类数
店铺 消费的商品 种类数

该店铺的商品平均商品价格等级
该店铺的商品平均销量等级
该店铺的商品平均收藏次数

不同评价数量等级的店铺 的 消费率
不同星级店铺的 消费率

店铺的评价数量等级
店铺的星级编号
店铺的好评率
店铺的服务态度评分
店铺的物流服务评分
店铺的描述相符评分
"""
#feature=feature1
def extract_shop_feature(feature):
    shop=feature[['instance_id','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','item_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description','is_trade']]
#店铺被浏览次数
    d=shop[['shop_id']]
    d=d.groupby('shop_id').size().reset_index()
    d.rename(columns={0:'shop_look_cnt'},inplace=True)
#店铺被消费次数
    d1=shop[['shop_id','is_trade']]
    d1=d1.groupby('shop_id').agg('sum').reset_index()
    d1.rename(columns={'is_trade':'shop_buy_cnt'},inplace=True)    
#店铺消费 次数/店铺浏览次数
    d2=d1[['shop_id']]
    d2['shop_buy_rate']=d1['shop_buy_cnt']/d['shop_look_cnt']    
#该店铺浏览的用户数
    d3=shop[['shop_id','user_id']]
    d3=d3.drop_duplicates()
    d3=d3.groupby('shop_id').size().reset_index()
    d3.rename(columns={0:'shop_user_look_cnt'},inplace=True)    
#该店铺消费的不同用户数
    d4=shop[['shop_id','user_id','is_trade']]
    d4=d4.drop_duplicates()
    d4=d4[['shop_id','is_trade']]
    d4=d4.groupby('shop_id').agg('sum').reset_index()
    d4.rename(columns={'is_trade':'shop_user_buy_cnt'},inplace=True)
#  该店铺消费的不同用户数/ 该店铺浏览的用户数
    d5=d4[['shop_id']]
    d5['shop_user_buy_rate']=d4['shop_user_buy_cnt']/d3['shop_user_look_cnt']
# 该店铺不同商品种类数
    d6=shop[['shop_id','item_id']]
    d6=d6.drop_duplicates()
    d6=d6.groupby('shop_id').size().reset_index()
    d6.rename(columns={0:'shop_item_cnt'},inplace=True)    
#该店铺被消费的商品种类数    
    d7=shop[['shop_id','item_id','is_trade']]
    d7=d7.drop_duplicates()
    d7=d7[['shop_id','is_trade']]
    d7=d7.groupby('shop_id').agg('sum').reset_index()
    d7.rename(columns={'is_trade':'shop_item_buy_cnt'},inplace=True)    
#   该店铺被消费的商品种类数 / 该店铺不同商品种类数
    d8=d4[['shop_id']]
    d8['shop_item_buy_rate']=d7['shop_item_buy_cnt']/d6['shop_item_cnt']    
# 该店铺的商品平均商品价格等级
    d9=shop[['shop_id','item_price_level']]
    d9=round(d9.groupby('shop_id').agg('mean').reset_index())
    d9.rename(columns={'item_price_level':'shop_item_mean_item_price_level'},inplace=True)        
#该店铺销售的商品的平均价格等级    
    d10=shop[['shop_id','item_price_level','is_trade']]
    d10=d10[d10['is_trade']==1][['shop_id','item_price_level']]    
    d10=round(d10.groupby('shop_id').agg('mean').reset_index())
    d10.rename(columns={'item_price_level':'shop_item_buy_mean_item_price_level'},inplace=True)    
    d10=pd.merge(d9[['shop_id']],d10,on='shop_id',how='left')  
    d10=d10.fillna(value=-1)   
#该店铺的商品平均销量等级
    d11=shop[['shop_id','item_sales_level']]
    d11=round(d11.groupby('shop_id').agg('mean').reset_index())
    d11.rename(columns={'item_sales_level':'shop_item_mean_item_sales_level'},inplace=True)
#该店铺销售的商品平均销量等级
    d12=shop[['shop_id','item_sales_level','is_trade']]
    d12=d12[d12['is_trade']==1][['shop_id','item_sales_level']]    
    d12=round(d12.groupby('shop_id').agg('mean').reset_index())
    d12.rename(columns={'item_sales_level':'shop_item_buy_mean_item_sales_level'},inplace=True)    
    d12=pd.merge(d9[['shop_id']],d12,on='shop_id',how='left')  
    d12=d12.fillna(value=-1)       
#该店铺的商品平均收藏次数    
    d13=shop[['shop_id','item_collected_level']]
    d13=round(d13.groupby('shop_id').agg('mean').reset_index())
    d13.rename(columns={'item_collected_level':'shop_item_mean_item_collected_level'},inplace=True)
#该店铺销售的商品平均收藏次数
    d14=shop[['shop_id','item_collected_level','is_trade']]
    d14=d14[d14['is_trade']==1][['shop_id','item_collected_level']]    
    d14=round(d14.groupby('shop_id').agg('mean').reset_index())
    d14.rename(columns={'item_collected_level':'shop_item_buy_mean_item_collected_level'},inplace=True)    
    d14=pd.merge(d9[['shop_id']],d14,on='shop_id',how='left')  
    d14=d14.fillna(value=-1)     
#该店铺的商品被展示次数
    d15=shop[['shop_id','item_pv_level']]
    d15=round(d15.groupby('shop_id').agg('mean').reset_index())
    d15.rename(columns={'item_pv_level':'shop_item_mean_item_pv_level'},inplace=True)
#该店铺销售的商品被展示次数
    d16=shop[['shop_id','item_pv_level','is_trade']]
    d16=d16[d16['is_trade']==1][['shop_id','item_pv_level']]    
    d16=round(d16.groupby('shop_id').agg('mean').reset_index())
    d16.rename(columns={'item_pv_level':'shop_item_buy_mean_item_pv_level'},inplace=True)    
    d16=pd.merge(d9[['shop_id']],d16,on='shop_id',how='left')  
    d16=d16.fillna(value=-1) 
#shop_review_num_level 不同评价数量等级的店铺 的 消费率
    d17=shop[['shop_review_num_level','is_trade']]
    d17['cnt']=1
    d17=d17.groupby('shop_review_num_level').agg('sum').reset_index()
    d17['shop_review_num_level_buy_rate']=d17['is_trade']/d17['cnt']
    d17=d17[['shop_review_num_level','shop_review_num_level_buy_rate']]
# shop_star_level 不同星级店铺的 消费率
    d18=shop[['shop_star_level','is_trade']]
    d18['cnt']=1
    d18=d18.groupby('shop_star_level').agg('sum').reset_index()
    d18['shop_star_level_buy_rate']=d18['is_trade']/d18['cnt']
    d18=d18[['shop_star_level','shop_star_level_buy_rate']]
#好评数 = 好评率*评价数量等级
    d19=shop[['shop_id','shop_review_positive_rate','shop_review_num_level']]
    d19['shop_positive_num']=d19['shop_review_positive_rate']*d19['shop_review_num_level']
    d19=d19[['shop_id','shop_positive_num']]
    
    shop=shop.drop(['instance_id','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','item_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','is_trade'],axis=1)
    shop=pd.merge(shop,d,on='shop_id',how='left')
    shop=pd.merge(shop,d1,on='shop_id',how='left')
    shop=pd.merge(shop,d2,on='shop_id',how='left')
    shop=pd.merge(shop,d3,on='shop_id',how='left')
    shop=pd.merge(shop,d4,on='shop_id',how='left')
    shop=pd.merge(shop,d5,on='shop_id',how='left')
    shop=pd.merge(shop,d6,on='shop_id',how='left')
    shop=pd.merge(shop,d7,on='shop_id',how='left')
    shop=pd.merge(shop,d8,on='shop_id',how='left')
    shop=pd.merge(shop,d9,on='shop_id',how='left')
    shop=pd.merge(shop,d10,on='shop_id',how='left')
    shop=pd.merge(shop,d11,on='shop_id',how='left')
    shop=pd.merge(shop,d12,on='shop_id',how='left')
    shop=pd.merge(shop,d13,on='shop_id',how='left')
    shop=pd.merge(shop,d14,on='shop_id',how='left')
    shop=pd.merge(shop,d15,on='shop_id',how='left')
    shop=pd.merge(shop,d16,on='shop_id',how='left')
    
    shop=pd.merge(shop,d17,on='shop_review_num_level',how='left')
    shop=pd.merge(shop,d18,on='shop_star_level',how='left')
    shop['shop_positive_num']=shop['shop_review_positive_rate']*shop['shop_review_num_level']
    #shop=pd.merge(shop,d19,on='shop_id',how='left')
    return shop
#%%
shop_feature1=extract_shop_feature(feature1)
shop_feature2=extract_shop_feature(feature2)
shop_feature3=extract_shop_feature(feature3)
#%%
"""
从label窗提取的特征
当天用户：
当天用户浏览的商品数量
当天用户浏览的不同的商品数量
当天用户浏览的不同商家数量

当天用户浏览的商品平均-----
#当天用户浏览的商家平均————
当天商铺被浏览次数

当天商品被浏览次数

用户在不同页的浏览次数

计算属性正确率
"""
#dataset=dataset1
def extract_other_feature(dataset):
    other=dataset[['instance_id','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','item_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description','real_hour','context_page_id','perdict_category','item_category_list','perdict_property','item_property_list']]
#当天用户浏览的商品数量
    d=other[['user_id']]
    d=d.groupby('user_id').size().reset_index()
    d.rename(columns={0:'label_user_look_cnt'},inplace=True)    
#当天用户浏览的不同的商品数量
    d1=other[['user_id','item_id']]
    d1=d1.drop_duplicates()
    d1=d1[['user_id']]
    d1=d1.groupby('user_id').size().reset_index()
    d1.rename(columns={0:'label_user_look_dif_item'},inplace=True)
#当天用户浏览的不同的商家种类
    d2=other[['user_id','shop_id']]
    d2=d2.drop_duplicates()
    d2=d2[['user_id']].groupby('user_id').size().reset_index()
    d2.rename(columns={0:'label_look_dif_shop'},inplace=True) 
#用户当天浏览的商品的平均价格等级
    d3=other[['user_id','item_price_level']]
    d3=round(d3.groupby('user_id').agg('mean').reset_index())
    d3.rename(columns={'item_price_level':'label_user_look_mean_item_price_level'})
#用户当天浏览的商品的平均销量等级
    d4=other[['user_id','item_sales_level']]
    d4=round(d4.groupby('user_id').agg('mean').reset_index())
    d4.rename(columns={'item_sales_level':'label_user_look_mean_item_sales_level'})
#用户当天浏览的商品的平均收藏次数
    d5=other[['user_id','item_collected_level']]
    d5=round(d5.groupby('user_id').agg('mean').reset_index())
    d5.rename(columns={'item_collected_level':'label_user_look_mean_item_collected_level'})
#当天店铺被浏览册数  
    d6=other[['shop_id']]
    d6=d6.groupby('shop_id').size().reset_index()
    d6.rename(columns={0:'label_shop_look_cnt'},inplace=True) 
#当天商品被浏览次数      
    d7=other[['item_id']]
    d7=d7.groupby('item_id').size().reset_index()
    d7.rename(columns={0:'label_item_look_cnt'},inplace=True) 
#不同页被浏览次数 
    d8=other[['context_page_id']]
    d8=d8.groupby('context_page_id').size().reset_index()
    d8.rename(columns={0:'label_context_page_id_look_cnt'},inplace=True)
#类别预测与商品的吻合度
    d9=other[['instance_id','perdict_category','item_category_list']]
    d9['label_predict_category_right_ratio']=d9['item_category_list'].astype('str')+':'+d9['perdict_category'].astype('str')
    d9['label_predict_category_right_ratio']=d9['label_predict_category_right_ratio'].apply(predictCategoryRight)
    d9=d9[['instance_id','label_predict_category_right_ratio']]
#属性预测吻合度
    d10=other[['instance_id','perdict_property','item_property_list']]
    d10['label_predict_property_right_ratio']=d10['item_property_list'].astype('str')+':'+d10['perdict_property'].astype('str')
    d10['label_predict_property_right_ratio']=d10['label_predict_property_right_ratio'].apply(predictPropertyRight)
    d10=d10[['instance_id','label_predict_property_right_ratio']]    
#属性符合个数
    d11=other[['instance_id','perdict_property','item_property_list']]
    d11['label_predict_property_right_num']=d11['item_property_list'].astype('str')+':'+d11['perdict_property'].astype('str')
    d11['label_predict_property_right_num']=d11['label_predict_property_right_num'].apply(predictPropertyRightNum)
    d11=d11[['instance_id','label_predict_property_right_num']]   
   
    other=pd.merge(other,d,on='user_id',how='left')
    other=pd.merge(other,d1,on='user_id',how='left')
    other=pd.merge(other,d2,on='user_id',how='left')
    other=pd.merge(other,d3,on='user_id',how='left')
    other=pd.merge(other,d4,on='user_id',how='left')
    other=pd.merge(other,d5,on='user_id',how='left')
    other=pd.merge(other,d6,on='shop_id',how='left')
    other=pd.merge(other,d7,on='item_id',how='left')
    other=pd.merge(other,d8,on='context_page_id',how='left')
    other=pd.merge(other,d9,on='instance_id',how='left')
    other=pd.merge(other,d10,on='instance_id',how='left')
    other=other.drop(['instance_id','item_brand_id','item_city_id','perdict_category','item_category_list','perdict_property','item_property_list'],axis=1)
    return other
#%%
other1= extract_other_feature(dataset1)
other2= extract_other_feature(dataset2)
other3= extract_other_feature(dataset3)
#%%
"""
用户——商品
用户——商品城市 购买转化率
用户——商品品牌 购买转化率
用户——商品价格等级 购买转化率
用户——商品销量等级 购买转化率
用户——商品收藏此书 购买转化率
用户——商品展示次数 购买转化率
"""
def extract_user_merchant_feature(dataset):
#%%
"""
用户——商店
用户——商店评价数量等级 购买转化率
用户——商店星级  购买转化率
用户——商店 好评数（需要计算） 购买转化率
"""
def extract_user_shop_feature(dataset):
#%%
"""
上下文信息
预测的类目属性列表中正确率
正确个数
不同正确个数的购买转化率
不同页的购买转化率
"""
def extract_context_feature(dataset):


