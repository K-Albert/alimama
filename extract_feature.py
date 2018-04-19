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
test_b = pd.read_csv('data/round1_ijcai_18_test_b_20180418.txt',sep=" ")
test=pd.concat([test_a,test_b]).reset_index(drop=True)
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

test['real_time'] = pd.to_datetime(test['context_timestamp'].apply(time2cov))
test['real_hour'] = test['real_time'].dt.hour
test['real_day'] = test['real_time'].dt.day
test['perdict_category']=test['predict_category_property'].apply(splitPredicCategory)
test['perdict_property']=test['predict_category_property'].apply(splitPredicProperty)
#18 19 20 21 22
#feature1=train[train['real_day']<23]
dataset1=train[train['real_day']==23]
#19 20 21 22 23
#feature2=train[(train['real_day']<24)&(train['real_day']>18)]
dataset2=train[train['real_day']==24]
#20 21 22 23 24
#feature3=train[(train['real_day']<25)&(train['real_day']>19)]
dataset3=test
feature1=train[train['real_day']==18]
feature2=train[train['real_day']==19]
feature3=train[train['real_day']==20]
feature4=train[train['real_day']==21]
feature5=train[train['real_day']==22]
feature6=train[train['real_day']==23]
feature7=train[train['real_day']==24]
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

#%%   user
feature1_2_3_4_5=pd.concat([feature1,feature2,feature3,feature4,feature4],ignore_index=True)
feature2_3_4_5_6=pd.concat([feature2,feature3,feature4,feature5,feature6],ignore_index=True)
feature3_4_5_6_7=pd.concat([feature3,feature4,feature5,feature6,feature7],ignore_index=True)
#%%
"""
感觉  用户应该是提取N天的
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
#用户在 一小时中的 第一次浏览时 购买 占所有购买的比率
#feature=feature1
def extract_user_feature(dataset,feature):
    label=dataset[['instance_id','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level']]
    user=feature[['instance_id','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','item_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description','is_trade','real_hour','real_time']]
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

    d74=user[['user_gender_id','is_trade','item_price_level']]
    d74=d74[d74['is_trade']==1][['user_gender_id','item_price_level']]
    d74=round(d74.groupby('user_gender_id').agg('mean').reset_index())
    d74.columns=['user_gender_id','user_gender_buy_mean_item_price_level'] 
    
    d75=user[['user_gender_id','is_trade','item_sales_level']]
    d75=d75[d75['is_trade']==1][['user_gender_id','item_sales_level']]
    d75=round(d75.groupby('user_gender_id').agg('mean').reset_index())
    d75.columns=['user_gender_id','user_gender_buy_mean_item_sales_level'] 

    d76=user[['user_gender_id','is_trade','item_collected_level']]
    d76=d76[d76['is_trade']==1][['user_gender_id','item_collected_level']]
    d76=round(d76.groupby('user_gender_id').agg('mean').reset_index())
    d76.columns=['user_gender_id','user_gender_buy_mean_item_collected_level'] 

    d77=user[['user_gender_id','is_trade','item_pv_level']]
    d77=d77[d77['is_trade']==1][['user_gender_id','item_pv_level']]
    d77=round(d77.groupby('user_gender_id').agg('mean').reset_index())
    d77.columns=['user_gender_id','user_gender_buy_mean_item_pv_level']   
    
    d78=user[['user_gender_id','is_trade','shop_review_num_level']]
    d78=d78[d78['is_trade']==1][['user_gender_id','shop_review_num_level']]
    d78=d78.groupby('user_gender_id').agg('mean').reset_index()
    d78.columns=['user_gender_id','user_gender_buy_mean_shop_review_num_level']    

    d79=user[['user_gender_id','is_trade','shop_review_num_level']]
    d79=d79[d79['is_trade']==1][['user_gender_id','shop_review_num_level']]
    d79=d79.groupby('user_gender_id').agg('mean').reset_index()
    d79.columns=['user_gender_id','user_gender_buy_mean_shop_review_num_level']    
    
    d80=user[['user_gender_id','is_trade','shop_review_positive_rate']]
    d80=d80[d80['is_trade']==1][['user_gender_id','shop_review_positive_rate']]
    d80=d80.groupby('user_gender_id').agg('mean').reset_index()
    d80.columns=['user_gender_id','user_gender_buy_mean_shop_review_positive_rate']    
    
    d81=user[['user_gender_id','is_trade','shop_star_level']]
    d81=d81[d81['is_trade']==1][['user_gender_id','shop_star_level']]
    d81=d81.groupby('user_gender_id').agg('mean').reset_index()
    d81.columns=['user_gender_id','user_gender_buy_mean_shop_star_level']    
    
    d82=user[['user_gender_id','is_trade','shop_score_service']]
    d82=d82[d82['is_trade']==1][['user_gender_id','shop_score_service']]
    d82=d82.groupby('user_gender_id').agg('mean').reset_index()
    d82.columns=['user_gender_id','user_gender_buy_mean_shop_score_service']    
    
    d83=user[['user_gender_id','is_trade','shop_score_delivery']]
    d83=d83[d83['is_trade']==1][['user_gender_id','shop_score_delivery']]
    d83=d83.groupby('user_gender_id').agg('mean').reset_index()
    d83.columns=['user_gender_id','user_gender_buy_mean_shop_score_delivery']    
    
    d84=user[['user_gender_id','is_trade','shop_score_description']]
    d84=d84[d84['is_trade']==1][['user_gender_id','shop_score_description']]
    d84=d84.groupby('user_gender_id').agg('mean').reset_index()
    d84.columns=['user_gender_id','user_gender_buy_mean_shop_score_description']      
 

    
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
    
    d45=user[['user_age_level','is_trade','shop_review_num_level']]
    d45=d45[d45['is_trade']==1][['user_age_level','shop_review_num_level']]
    d45=d45.groupby('user_age_level').agg('mean').reset_index()
    d45.columns=['user_age_level','user_age_buy_mean_shop_review_num_level']     

    d46=user[['user_age_level','is_trade','shop_review_positive_rate']]
    d46=d46[d46['is_trade']==1][['user_age_level','shop_review_positive_rate']]
    d46=d46.groupby('user_age_level').agg('mean').reset_index()
    d46.columns=['user_age_level','user_age_buy_mean_shop_review_positive_rate']     

    d47=user[['user_age_level','is_trade','shop_star_level']]
    d47=d47[d47['is_trade']==1][['user_age_level','shop_star_level']]
    d47=d47.groupby('user_age_level').agg('mean').reset_index()
    d47.columns=['user_age_level','user_age_buy_mean_shop_star_level']   

    d48=user[['user_age_level','is_trade','shop_score_service']]
    d48=d48[d48['is_trade']==1][['user_age_level','shop_score_service']]
    d48=d48.groupby('user_age_level').agg('mean').reset_index()
    d48.columns=['user_age_level','user_age_buy_mean_shop_score_service']  

    d49=user[['user_age_level','is_trade','shop_score_service']]
    d49=d49[d49['is_trade']==1][['user_age_level','shop_score_service']]
    d49=d49.groupby('user_age_level').agg('mean').reset_index()
    d49.columns=['user_age_level','user_age_buy_mean_shop_score_service']  

    d50=user[['user_age_level','is_trade','shop_score_delivery']]
    d50=d50[d50['is_trade']==1][['user_age_level','shop_score_delivery']]
    d50=d50.groupby('user_age_level').agg('mean').reset_index()
    d50.columns=['user_age_level','user_age_buy_mean_shop_score_delivery']   

    d51=user[['user_age_level','is_trade','item_price_level']]
    d51=d51[d51['is_trade']==1][['user_age_level','item_price_level']]
    d51=round(d51.groupby('user_age_level').agg('mean').reset_index())
    d51.columns=['user_age_level','user_age_buy_mean_item_price_level']

    d52=user[['user_age_level','is_trade','item_sales_level']]
    d52=d52[d52['is_trade']==1][['user_age_level','item_sales_level']]
    d52=round(d52.groupby('user_age_level').agg('mean').reset_index())
    d52.columns=['user_age_level','user_age_buy_mean_item_sales_level']  

    d53=user[['user_age_level','is_trade','item_collected_level']]
    d53=d53[d53['is_trade']==1][['user_age_level','item_collected_level']]
    d53=round(d53.groupby('user_age_level').agg('mean').reset_index())
    d53.columns=['user_age_level','user_age_buy_mean_item_collected_level']  

    d54=user[['user_age_level','is_trade','item_pv_level']]
    d54=d54[d54['is_trade']==1][['user_age_level','item_pv_level']]
    d54=round(d54.groupby('user_age_level').agg('mean').reset_index())
    d54.columns=['user_age_level','user_age_buy_mean_item_pv_level']  
       
#user_occupation_id 不同职业用户   
    d43=user[['user_occupation_id','is_trade']]
    d43['cnt']=1
    d43=d43.groupby('user_occupation_id').agg('sum').reset_index()
    d43['user_occupation_buy_rate']=d43['is_trade']/d43['cnt']
    d43=d43[['user_occupation_id','user_occupation_buy_rate']] 

    d64=user[['user_occupation_id','is_trade','item_price_level']]
    d64=d64[d64['is_trade']==1][['user_occupation_id','item_price_level']]
    d64=round(d64.groupby('user_occupation_id').agg('mean').reset_index())
    d64.columns=['user_occupation_id','user_occupation_buy_mean_item_price_level'] 

    d65=user[['user_occupation_id','is_trade','item_sales_level']]
    d65=d65[d65['is_trade']==1][['user_occupation_id','item_sales_level']]
    d65=round(d65.groupby('user_occupation_id').agg('mean').reset_index())
    d65.columns=['user_occupation_id','user_occupation_buy_mean_item_sales_level'] 
    
    d66=user[['user_occupation_id','is_trade','item_collected_level']]
    d66=d66[d66['is_trade']==1][['user_occupation_id','item_collected_level']]
    d66=round(d66.groupby('user_occupation_id').agg('mean').reset_index())
    d66.columns=['user_occupation_id','user_occupation_buy_mean_item_collected_level'] 
    
    d67=user[['user_occupation_id','is_trade','item_pv_level']]
    d67=d67[d67['is_trade']==1][['user_occupation_id','item_pv_level']]
    d67=round(d67.groupby('user_occupation_id').agg('mean').reset_index())
    d67.columns=['user_occupation_id','user_occupation_buy_mean_item_pv_level'] 
    
    d68=user[['user_occupation_id','is_trade','shop_review_num_level']]
    d68=d68[d68['is_trade']==1][['user_occupation_id','shop_review_num_level']]
    d68=d68.groupby('user_occupation_id').agg('mean').reset_index()
    d68.columns=['user_occupation_id','user_occupation_buy_mean_shop_review_num_level'] 
    
    d69=user[['user_occupation_id','is_trade','shop_review_positive_rate']]
    d69=d69[d69['is_trade']==1][['user_occupation_id','shop_review_positive_rate']]
    d69=d69.groupby('user_occupation_id').agg('mean').reset_index()
    d69.columns=['user_occupation_id','user_occupation_buy_mean_shop_review_positive_rate'] 
    
    d70=user[['user_occupation_id','is_trade','shop_star_level']]
    d70=d70[d70['is_trade']==1][['user_occupation_id','shop_star_level']]
    d70=d70.groupby('user_occupation_id').agg('mean').reset_index()
    d70.columns=['user_occupation_id','user_occupation_buy_mean_shop_star_level'] 
    
    d71=user[['user_occupation_id','is_trade','shop_score_service']]
    d71=d71[d71['is_trade']==1][['user_occupation_id','shop_score_service']]
    d71=d71.groupby('user_occupation_id').agg('mean').reset_index()
    d71.columns=['user_occupation_id','user_occupation_buy_mean_shop_score_service'] 
    
    d72=user[['user_occupation_id','is_trade','shop_score_delivery']]
    d72=d72[d72['is_trade']==1][['user_occupation_id','shop_score_delivery']]
    d72=d72.groupby('user_occupation_id').agg('mean').reset_index()
    d72.columns=['user_occupation_id','user_occupation_buy_mean_shop_score_delivery'] 
    
    d73=user[['user_occupation_id','is_trade','shop_score_description']]
    d73=d73[d73['is_trade']==1][['user_occupation_id','shop_score_description']]
    d73=d73.groupby('user_occupation_id').agg('mean').reset_index()
    d73.columns=['user_occupation_id','user_occupation_buy_mean_shop_score_description'] 
#user_star_level 不同星级用户成交率    
    d44=user[['user_star_level','is_trade']]
    d44['cnt']=1
    d44=d44.groupby('user_star_level').agg('sum').reset_index()
    d44['user_star_buy_rate']=d44['is_trade']/d44['cnt']
    d44=d44[['user_star_level','user_star_buy_rate']] 

    d55=user[['user_star_level','is_trade','item_price_level']]
    d55=d55[d55['is_trade']==1][['user_star_level','item_price_level']]
    d55=round(d55.groupby('user_star_level').agg('mean').reset_index())
    d55.columns=['user_star_level','user_star_buy_mean_item_price_level']     
    
    d56=user[['user_star_level','is_trade','item_sales_level']]
    d56=d56[d56['is_trade']==1][['user_star_level','item_sales_level']]
    d56=round(d56.groupby('user_star_level').agg('mean').reset_index())
    d56.columns=['user_star_level','user_star_buy_mean_item_sales_level'] 
    
    d57=user[['user_star_level','is_trade','item_collected_level']]
    d57=d57[d57['is_trade']==1][['user_star_level','item_collected_level']]
    d57=round(d57.groupby('user_star_level').agg('mean').reset_index())
    d57.columns=['user_star_level','user_star_buy_mean_item_collected_level']   

    d58=user[['user_star_level','is_trade','item_pv_level']]
    d58=d58[d58['is_trade']==1][['user_star_level','item_pv_level']]
    d58=round(d58.groupby('user_star_level').agg('mean').reset_index())
    d58.columns=['user_star_level','user_star_buy_mean_item_pv_level']   

    d59=user[['user_star_level','is_trade','shop_review_num_level']]
    d59=d59[d59['is_trade']==1][['user_star_level','shop_review_num_level']]
    d59=d59.groupby('user_star_level').agg('mean').reset_index()
    d59.columns=['user_star_level','user_star_buy_mean_shop_review_num_level'] 

    d60=user[['user_star_level','is_trade','shop_review_positive_rate']]
    d60=d60[d60['is_trade']==1][['user_star_level','shop_review_positive_rate']]
    d60=d60.groupby('user_star_level').agg('mean').reset_index()
    d60.columns=['user_star_level','user_star_buy_mean_shop_review_positive_rate'] 

    d61=user[['user_star_level','is_trade','shop_star_level']]
    d61=d61[d61['is_trade']==1][['user_star_level','shop_star_level']]
    d61=d61.groupby('user_star_level').agg('mean').reset_index()
    d61.columns=['user_star_level','user_star_buy_mean_shop_star_level'] 

    d61=user[['user_star_level','is_trade','shop_score_service']]
    d61=d61[d61['is_trade']==1][['user_star_level','shop_score_service']]
    d61=d61.groupby('user_star_level').agg('mean').reset_index()
    d61.columns=['user_star_level','user_star_buy_mean_shop_score_service'] 

    d62=user[['user_star_level','is_trade','shop_score_delivery']]
    d62=d62[d62['is_trade']==1][['user_star_level','shop_score_delivery']]
    d62=d62.groupby('user_star_level').agg('mean').reset_index()
    d62.columns=['user_star_level','user_star_buy_mean_shop_score_delivery'] 

    d63=user[['user_star_level','is_trade','shop_score_description']]
    d63=d63[d63['is_trade']==1][['user_star_level','shop_score_description']]
    d63=d63.groupby('user_star_level').agg('mean').reset_index()
    d63.columns=['user_star_level','user_star_buy_mean_shop_score_description']         
#    def sortFrequest(x):
#       x=x.groupby('real_hour').size().reset_index() 
#       x.rename(columns={0:'cnt'},inplace=True)
#       x.sort_values(by = ['cnt'],axis = 0,ascending = False).reset_index(drop=True) 
#       max_val=x['real_hour'][0]
#       return max_val
#        
#    d45=user[['user_id','real_hour']]
#    d45=d45.groupby('user_id').apply(sortFrequest).reset_index()
#    d45.rename(columns={0:'user_buy_frequest_hour'},inplace=True)
    
    label=pd.merge(label,d,on='user_id',how='left')
    label=pd.merge(label,d1,on='user_id',how='left')
    label=pd.merge(label,d2,on='user_id',how='left')
    label=pd.merge(label,d3,on='user_id',how='left')
    label=pd.merge(label,d4,on='user_id',how='left')
    label=pd.merge(label,d5,on='user_id',how='left')
    label=pd.merge(label,d6,on='user_id',how='left')
    label=pd.merge(label,d7,on='user_id',how='left')
    label=pd.merge(label,d8,on='user_id',how='left')
    label=pd.merge(label,d9,on='user_id',how='left')
    label=pd.merge(label,d10,on='user_id',how='left')
    label=pd.merge(label,d11,on='user_id',how='left')
    label=pd.merge(label,d12,on='user_id',how='left')
    label=pd.merge(label,d13,on='user_id',how='left')
    label=pd.merge(label,d14,on='user_id',how='left')
    label=pd.merge(label,d15,on='user_id',how='left')
    label=pd.merge(label,d16,on='user_id',how='left')
    label=pd.merge(label,d17,on='user_id',how='left')
    label=pd.merge(label,d18,on='user_id',how='left')
    label=pd.merge(label,d19,on='user_id',how='left')
    label=pd.merge(label,d20,on='user_id',how='left')
    label=pd.merge(label,d21,on='user_id',how='left')
    label=pd.merge(label,d22,on='user_id',how='left')
    label=pd.merge(label,d23,on='user_id',how='left')
    label=pd.merge(label,d24,on='user_id',how='left')
    label=pd.merge(label,d25,on='user_id',how='left')
    label=pd.merge(label,d26,on='user_id',how='left')
    label=pd.merge(label,d27,on='user_id',how='left')
    label=pd.merge(label,d28,on='user_id',how='left')
    label=pd.merge(label,d29,on='user_id',how='left')
    label=pd.merge(label,d30,on='user_id',how='left')
    label=pd.merge(label,d31,on='user_id',how='left')
    label=pd.merge(label,d32,on='user_id',how='left')
    label=pd.merge(label,d33,on='user_id',how='left')
    label=pd.merge(label,d34,on='user_id',how='left')
    label=pd.merge(label,d35,on='user_id',how='left')
    label=pd.merge(label,d36,on='user_id',how='left')
    label=pd.merge(label,d37,on='user_id',how='left')
    label=pd.merge(label,d38,on='user_id',how='left')
    label=pd.merge(label,d39,on='user_id',how='left')
   
    label=pd.merge(label,d40,on='user_gender_id',how='left')
    label=pd.merge(label,d41,on='user_age_level',how='left')
    label=pd.merge(label,d42,on='user_age_level',how='left')
    label=pd.merge(label,d43,on='user_occupation_id',how='left')
    label=pd.merge(label,d44,on='user_star_level',how='left')   
#    label=pd.merge(label,d45,on='user_age_level',how='left')
#    label=pd.merge(label,d46,on='user_age_level',how='left')
#    label=pd.merge(label,d47,on='user_age_level',how='left')
#    label=pd.merge(label,d48,on='user_age_level',how='left')
#    label=pd.merge(label,d49,on='user_age_level',how='left')
#    label=pd.merge(label,d50,on='user_age_level',how='left')
#    label=pd.merge(label,d51,on='user_age_level',how='left')
#    label=pd.merge(label,d52,on='user_age_level',how='left')
#    label=pd.merge(label,d53,on='user_age_level',how='left')
#    label=pd.merge(label,d54,on='user_age_level',how='left')
#    label=pd.merge(label,d55,on='user_star_level',how='left')   
#    label=pd.merge(label,d56,on='user_star_level',how='left')   
#    label=pd.merge(label,d57,on='user_star_level',how='left')   
#    label=pd.merge(label,d58,on='user_star_level',how='left')   
#    label=pd.merge(label,d59,on='user_star_level',how='left')   
#    label=pd.merge(label,d60,on='user_star_level',how='left')   
#    label=pd.merge(label,d61,on='user_star_level',how='left')   
#    label=pd.merge(label,d62,on='user_star_level',how='left')   
#    label=pd.merge(label,d63,on='user_star_level',how='left')   

#    label=pd.merge(label,d64,on='user_occupation_id',how='left')
#    label=pd.merge(label,d65,on='user_occupation_id',how='left')
#    label=pd.merge(label,d66,on='user_occupation_id',how='left')
#    label=pd.merge(label,d67,on='user_occupation_id',how='left')
#    label=pd.merge(label,d68,on='user_occupation_id',how='left')
#    label=pd.merge(label,d69,on='user_occupation_id',how='left')
#    label=pd.merge(label,d70,on='user_occupation_id',how='left')
#    label=pd.merge(label,d71,on='user_occupation_id',how='left')
#    label=pd.merge(label,d72,on='user_occupation_id',how='left')
#    label=pd.merge(label,d73,on='user_occupation_id',how='left')
#
#    label=pd.merge(label,d74,on='user_gender_id',how='left')
#    label=pd.merge(label,d75,on='user_gender_id',how='left')
#    label=pd.merge(label,d76,on='user_gender_id',how='left')
#    label=pd.merge(label,d77,on='user_gender_id',how='left')
#    label=pd.merge(label,d78,on='user_gender_id',how='left')
#    label=pd.merge(label,d79,on='user_gender_id',how='left')
#    label=pd.merge(label,d80,on='user_gender_id',how='left')
#    label=pd.merge(label,d81,on='user_gender_id',how='left')
#    label=pd.merge(label,d82,on='user_gender_id',how='left')
#    label=pd.merge(label,d83,on='user_gender_id',how='left')
#    label=pd.merge(label,d84,on='user_gender_id',how='left')


#    label=pd.merge(label,d44,on='user_star_level',how='left')
    return label


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
def extract_shop_feature(dataset,feature):
    label=dataset[['instance_id','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']]
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
    
    label=pd.merge(label,d,on='shop_id',how='left')
    label=pd.merge(label,d1,on='shop_id',how='left')
    label=pd.merge(label,d2,on='shop_id',how='left')
    label=pd.merge(label,d3,on='shop_id',how='left')
    label=pd.merge(label,d4,on='shop_id',how='left')
    label=pd.merge(label,d5,on='shop_id',how='left')
    label=pd.merge(label,d6,on='shop_id',how='left')
    label=pd.merge(label,d7,on='shop_id',how='left')
    label=pd.merge(label,d8,on='shop_id',how='left')
    label=pd.merge(label,d9,on='shop_id',how='left')
    label=pd.merge(label,d10,on='shop_id',how='left')
    label=pd.merge(label,d11,on='shop_id',how='left')
    label=pd.merge(label,d12,on='shop_id',how='left')
    label=pd.merge(label,d13,on='shop_id',how='left')
    label=pd.merge(label,d14,on='shop_id',how='left')
    label=pd.merge(label,d15,on='shop_id',how='left')
    label=pd.merge(label,d16,on='shop_id',how='left')
    
    label=pd.merge(label,d17,on='shop_review_num_level',how='left')
    label=pd.merge(label,d18,on='shop_star_level',how='left')
    label['shop_positive_num']=label['shop_review_positive_rate']*label['shop_review_num_level']

    return label


#%% 
"""
每天item 会变 所以按照天提取
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
消费过该商品的不同用户数量
#商品价格等级有没有下降
"""
#label窗中出现的到 feature窗中的特征
def extract_label_merchant_feature(dataset,feature):
    label=dataset[['instance_id','item_id','item_property_list','item_brand_id','item_category_list','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level']]
    merchant=feature[['item_id','item_category_list','item_property_list','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','user_id','perdict_category','perdict_property','predict_category_property','is_trade']]
    label['second_category']=label['item_category_list'].apply(splitItemCategory_second) 
    label['item_property_cnt']=label.item_property_list.apply(itemPropertyCnt)  
    label['third_category']=label.item_category_list.apply(lambda x:-1 if(len(x.split(';'))<3) else int(x.split(';')[2]))  
    label=label.drop(['item_property_list','item_category_list'],axis=1)
    
    merchant['second_category']=merchant['item_category_list'].apply(splitItemCategory_second)  
    merchant['item_property_cnt']=merchant.item_property_list.apply(itemPropertyCnt)  
#    merchant['third_category']=merchant.item_category_list.apply(lambda x:-1 if(len(x.split(';'))<3) else int(x.split(';')[2]))  
#‘second_caregory’该二级类目下商品的 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级（几个数据集的二级类目相同）
    d=merchant[['second_category','item_price_level','item_sales_level','item_collected_level','item_pv_level']]
    d.replace(-1,0,inplace=True)
    d=round(d.groupby('second_category').agg(['mean']).reset_index())
    d.columns=['second_category','mean_item_price_level','mean_item_sales_level','mean_item_collected_level','mean_item_pv_level']
#‘second_caregory’该二级类目下成功销售的商品的 平均、最大、最小价格等级，销量等级，收藏次数等级，展示次数等级（item_sales_level有缺失值 用0替代的  会影响最小值）   
    d1=merchant[['second_category','item_price_level','item_sales_level','item_collected_level','item_pv_level','is_trade']]
    d1=d1[d1['is_trade']==1]
    d1=d1.drop('is_trade',axis=1)
    d1.replace(-1,0,inplace=True)
    d1=round(d1.groupby('second_category').agg(['mean']).reset_index())
    d1.columns=['second_category','trans_mean_item_price_level','trans_mean_item_sales_level','trans_mean_item_collected_level','trans_mean_item_pv_level']
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
#消费过该商品的不同用户个数
    d37=merchant[['item_id','user_id']]
    d37=d37.drop_duplicates()[['item_id']]
    d37=d37.groupby('item_id').size().reset_index()
    d37.rename(columns={0:'merchant_dif_user_buy'},inplace=True) 
    
#组合起来
    label=pd.merge(label,d,on='second_category',how='left')
    label=pd.merge(label,d1,on='second_category',how='left')
    label=pd.merge(label,d3,on='second_category',how='left')
    label=pd.merge(label,d7,on='second_category',how='left')
    label=pd.merge(label,d8,on='second_category',how='left')
    label=pd.merge(label,d9,on='second_category',how='left')
    label=pd.merge(label,d10,on='second_category',how='left')

    label=pd.merge(label,d4,on='item_id',how='left')
    label=pd.merge(label,d5,on='item_id',how='left')
    label=pd.merge(label,d6,on='item_id',how='left')
    
    label=pd.merge(label,d11,on='item_price_level',how='left')
    label=pd.merge(label,d12,on='item_price_level',how='left')
    label=pd.merge(label,d13,on='item_price_level',how='left')
    
    label=pd.merge(label,d14,on='item_sales_level',how='left')
    label=pd.merge(label,d15,on='item_sales_level',how='left')
    label=pd.merge(label,d16,on='item_sales_level',how='left')
    
    label=pd.merge(label,d17,on='item_collected_level',how='left')
    label=pd.merge(label,d18,on='item_collected_level',how='left')
    label=pd.merge(label,d19,on='item_collected_level',how='left')
    
    label=pd.merge(label,d20,on='item_pv_level',how='left')
    label=pd.merge(label,d21,on='item_pv_level',how='left')
    label=pd.merge(label,d22,on='item_pv_level',how='left')
    
    label=pd.merge(label,d23,on='item_city_id',how='left')
    label=pd.merge(label,d24,on='item_city_id',how='left')
    label=pd.merge(label,d25,on='item_city_id',how='left')
    
    label=pd.merge(label,d26,on='item_brand_id',how='left')
    label=pd.merge(label,d27,on='item_brand_id',how='left')
    label=pd.merge(label,d28,on='item_brand_id',how='left')
    
    label=pd.merge(label,d29,on='item_property_cnt',how='left')
    label=pd.merge(label,d30,on='item_property_cnt',how='left')
    label=pd.merge(label,d31,on='item_property_cnt',how='left')
    
    label=pd.merge(label,d37,on='item_id',how='left')
    label=label.drop(['item_brand_id','item_city_id'],axis=1)
    return label


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
def extract_other_feature(feature,dataset):
    other=dataset
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
    d3.rename(columns={'item_price_level':'label_user_look_mean_item_price_level'},inplace=True)
#用户当天浏览的商品的平均销量等级
    d4=other[['user_id','item_sales_level']]
    d4=round(d4.groupby('user_id').agg('mean').reset_index())
    d4.rename(columns={'item_sales_level':'label_user_look_mean_item_sales_level'},inplace=True)
#用户当天浏览的商品的平均收藏次数
    d5=other[['user_id','item_collected_level']]
    d5=round(d5.groupby('user_id').agg('mean').reset_index())
    d5.rename(columns={'item_collected_level':'label_user_look_mean_item_collected_level'},inplace=True)
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
#当天用户浏览的商家的平均评价数量等级
    d12=other[['shop_id','shop_review_num_level']]
    d12=round(d12.groupby('shop_id').agg('mean').reset_index())
    d12.rename(columns={'shop_review_num_level':'label_user_look_mean_shop_review_num_level'},inplace=True)    
#当天用户浏览的商家的平均好评率
    d13=other[['shop_id','shop_review_positive_rate']]
    d13=d13.groupby('shop_id').agg('mean').reset_index()
    d13.rename(columns={'shop_review_positive_rate':'label_user_look_mean_shop_review_positive_rate'},inplace=True)      
#当天用户浏览的商家平均星级
    d14=other[['shop_id','shop_star_level']]
    d14=round(d14.groupby('shop_id').agg('mean').reset_index())
    d14.rename(columns={'shop_star_level':'label_user_look_mean_shop_star_level'},inplace=True)          
#当天用户浏览的商家的平均服务态度
    d15=other[['shop_id','shop_score_service']]
    d15=d15.groupby('shop_id').agg('mean').reset_index()
    d15.rename(columns={'shop_score_service':'label_user_look_mean_shop_score_service'},inplace=True)          
#当天用户浏览的商家的平均物流服务
    d16=other[['shop_id','shop_score_delivery']]
    d16=d16.groupby('shop_id').agg('mean').reset_index()
    d16.rename(columns={'shop_score_delivery':'label_user_look_mean_shop_score_delivery'},inplace=True)    
#当天用户浏览的商家 的描述相符评分
    d17=other[['shop_id','shop_score_description']]
    d17=d17.groupby('shop_id').agg('mean').reset_index()
    d17.rename(columns={'shop_score_description':'label_user_look_mean_shop_score_description'},inplace=True)     
#是不是用户 该 小时的 的最后一次  浏览 user_id  real_time 去merge
#    d18=other[['user_id','real_hour','real_time']]
#    d18=d18.drop_duplicates()
#    d19=d18.groupby(['user_id','real_hour']).size().reset_index()
#    d19=d19.rename(columns={0:'cnt'})
#    d19=d19[d19['cnt']!=1] 
#    d19=d19.drop_duplicates()
#    d18=pd.merge(d19,d18,on=['user_id','real_hour'],how='inner')
#    
#    d20=d18.groupby(['user_id','real_hour']).agg('max').reset_index()
#    d20=d20.rename(columns={'real_time':'real_time_max'})    
#    d18=pd.merge(d18,d20,on=['user_id','real_hour'],how='left')
#    d18['label_is_latest_time']=(d18['real_time']==d18['real_time_max']).astype('int')
#    d18=d18[['user_id','real_time','label_is_latest_time']]
#    d18=d18.drop_duplicates()
# 现在 是不是 用户当天 当小时最后一次电击
    d18=other[['user_id','real_hour','real_time']]
    d18=d18.drop_duplicates()
    d19=d18.groupby(['user_id','real_hour']).size().reset_index()
    d19=d19.rename(columns={0:'cnt'})
    d19=d19[d19['cnt']!=1] 
    d19=d19.drop_duplicates()
    d18=pd.merge(d19,d18,on=['user_id','real_hour'],how='inner')
    
    d20=d18.groupby(['user_id','real_hour']).agg('max').reset_index()
    d20=d20.rename(columns={'real_time':'real_time_max'})    
    d18=pd.merge(d18,d20,on=['user_id','real_hour'],how='left')
    d18['label_is_latest_time']=(d18['real_time']==d18['real_time_max']).astype('int')
    d18=d18[['user_id','real_time','label_is_latest_time']]
    d18=d18.drop_duplicates()    
#用户 + 在这个时间点之前(并不局限于该小时) 有没有浏览过某品牌user_id user_id item_brand_id
    d19=other[['user_id','real_time','item_brand_id']]
    d20=d19.groupby(['user_id','item_brand_id']).agg('min').reset_index()
    d20=d20.rename(columns={'real_time':'real_time_min'})
    
    d19=pd.merge(d19,d20,on=['user_id','item_brand_id'],how='left')
    d19['label_item_brand_has_ever']=(d19['real_time']>d19['real_time_min']).astype('int')
    d19=d19[['user_id','item_brand_id','label_item_brand_has_ever','real_time']]
    d19=d19.drop_duplicates()

#用户 + 在这个时间点之前 有没有浏览过某城市    
    d20=other[['user_id','real_time','item_city_id']]
    d21=d20.groupby(['user_id','item_city_id']).agg('min').reset_index()
    d21=d21.rename(columns={'real_time':'real_time_min'})
    
    d20=pd.merge(d20,d21,on=['user_id','item_city_id'],how='left')
    d20['label_item_city_has_ever']=(d20['real_time']>d20['real_time_min']).astype('int')
    d20=d20[['user_id','item_city_id','label_item_city_has_ever','real_time']]
    d20=d20.drop_duplicates()

#这个 小时里 浏览的 这个 item——price 是最低的吗
    d21=other[['user_id','real_hour','item_price_level']]
    d22=d21.groupby(['user_id','real_hour']).agg('min').reset_index()
    d22=d22.rename(columns={'item_price_level':'item_price_level_min'})
    
    d21=pd.merge(d21,d22,on=['user_id','real_hour'],how='left')
    d21['label_item_price_is_lowest']=(~(d21['item_price_level']>d21['item_price_level_min'])).astype('int')
    d21=d21[['user_id','real_hour','label_item_price_is_lowest','item_price_level']]
    d21=d21.drop_duplicates()
    
#用户 在这个小时里 浏览的 shop_review_num_level 是最高的吗
    d22=other[['user_id','real_hour','shop_review_num_level']]
    d23=d22.groupby(['user_id','real_hour']).agg('max').reset_index()
    d23=d23.rename(columns={'shop_review_num_level':'shop_review_num_level_max'})
    
    d22=pd.merge(d22,d23,on=['user_id','real_hour'],how='left')
    d22['label_shop_review_num_level_is_highest']=(d22['shop_review_num_level']==d22['shop_review_num_level_max']).astype('int')
    d22=d22[['user_id','real_hour','label_shop_review_num_level_is_highest','shop_review_num_level']]    
    d22=d22.drop_duplicates()

#    商品价格等级 在n天内 有没有下降过
    d23=feature[['item_id','item_price_level','real_day']]
    d23=d23.drop_duplicates(['item_id','item_price_level'])
    
    t=d23.groupby('item_id').size().reset_index()
    t=t.rename(columns={0:'cnt'})
    t=t[t['cnt']>1]
    t=d23[d23['item_id'].isin(t['item_id'])]
    t1=t.groupby('item_id').agg({'real_day':'max'}).reset_index()
    t1=t1.rename(columns={'real_day':'real_day_max'})
    t1=pd.merge(t,t1,on='item_id',how='left')
    t1['is_del']=(t1['real_day']!=t1['real_day_max']).astype('int')
    t1=pd.merge(d23,t1,on=['real_day','item_id','item_price_level'],how='left')    
    t1=t1.drop('real_day_max',axis=1)
    t1=t1.fillna(0)
    d23=t1[t1['is_del']==0]
    d23=d23.drop(['is_del','real_day'],axis=1)
    
    d23=d23.rename(columns={'item_price_level':'item_price_level_before'})
    d24=other[['item_id','item_price_level']]
    d24=d24.drop_duplicates()
    d24=d24.rename(columns={'item_price_level':'item_price_level_now'})
    d25=pd.merge(d24,d23,on='item_id',how='inner')
    d25['label_item_price_down']=(d25['item_price_level_now']<d25['item_price_level_before']).astype('int')
    d23=d25[['item_id','label_item_price_down']]
    d23=d23.drop_duplicates()
#    商品 价格等级 与前一天相比 有没有上升
    d24=feature[['item_id','item_price_level','real_day']]
    d24=d24.drop_duplicates(['item_id','item_price_level'])

    t=d24.groupby('item_id').size().reset_index()
    t=t.rename(columns={0:'cnt'})
    t=t[t['cnt']>1]
    t=d24[d24['item_id'].isin(t['item_id'])]
    t1=t.groupby('item_id').agg({'real_day':'max'}).reset_index()
    t1=t1.rename(columns={'real_day':'real_day_max'})
    t1=pd.merge(t,t1,on='item_id',how='left')
    t1['is_del']=(t1['real_day']!=t1['real_day_max']).astype('int')
    t1=pd.merge(d24,t1,on=['real_day','item_id','item_price_level'],how='left')    
    t1=t1.drop('real_day_max',axis=1)
    t1=t1.fillna(0)
    d24=t1[t1['is_del']==0]
    d24=d24.drop(['is_del','real_day'],axis=1)    
      
    d24=d24.rename(columns={'item_price_level':'item_price_level_before'})
    d25=other[['item_id','item_price_level']]
    d25=d25.drop_duplicates()
    d25=d25.rename(columns={'item_price_level':'item_price_level_now'})
    d26=pd.merge(d25,d24,on='item_id',how='inner')
    d26['label_item_price_up']=(d26['item_price_level_now']>d26['item_price_level_before']).astype('int')
    d24=d26[['item_id','label_item_price_up']]
    d24=d24.drop_duplicates()
#  商品 销量等级、展示次数等级、收藏次数等级
#  历史上是否 浏览过该item
    d25=feature[['user_id','item_id']]
    d25=d25.drop_duplicates()
    d26=other[['user_id','item_id']]
    d26=d26.drop_duplicates()
    d26=pd.merge(d26,d25,on=['item_id','user_id'],how='inner')  
    d26['label_item_has_before']=1
    d26=pd.merge(other[['user_id','item_id']],d26,on=['user_id','item_id'],how='left')
    d26=d26.fillna(0)
    d26=d26.drop_duplicates()
#  历史上 是否 浏览过 某品牌
    d27=feature[['user_id','item_brand_id']]
    d27=d27.drop_duplicates()
    d28=other[['user_id','item_brand_id']]
    d28=d28.drop_duplicates()    
    d28=pd.merge(d28,d27,on=['item_brand_id','user_id'],how='inner')  
    d28['label_item_brand_has_before']=1
    d28=pd.merge(other[['user_id','item_brand_id']],d28,on=['user_id','item_brand_id'],how='left')
    d28=d28.fillna(0)
    d28=d28.drop_duplicates() 
# 用户  小时  是一天中 后一个小时吗  !!!!!!!!!!!!!!!这里 还要再考虑一下  有没有前提条件 是 当天分两个小时
    d29=other[['user_id','real_hour']]
    d30=d29.groupby('user_id').agg('max').reset_index()
    d30=d30.rename(columns={'real_hour':'real_hour_max'})
    
    d29=pd.merge(d29,d30,on='user_id',how='left')
    d29['label_is_lastest_hour_of_day']=(d29['real_hour']==d29['real_hour_max']).astype('int')
    d29=d29[['user_id','real_hour','label_is_lastest_hour_of_day']]
    d29=d29.drop_duplicates()
    d29=other[['user_id','real_hour']]
    d30=d29.groupby('user_id').agg('max').reset_index()
    d30=d30.rename(columns={'real_hour':'real_hour_max'})
    
    d29=pd.merge(d29,d30,on='user_id',how='left')
    d29['label_is_lastest_hour_of_day']=(d29['real_hour']==d29['real_hour_max']).astype('int')
    d29=d29[['user_id','real_hour','label_is_lastest_hour_of_day']]
    d29=d29.drop_duplicates()    
#用户当天  分了好几个小时 浏览   当前是后一个小时
#    d30=other[['user_id','real_hour']]
#    d30=d30.drop_duplicates()
#    d31=d30.groupby(['user_id']).size().reset_index()
#    d31=d31.rename(columns={0:'cnt'})
#    d31=d31[d31['cnt']>1]   
#    d31=pd.merge(d31,d30,on='user_id',how='inner')
#    d31=d31[['user_id','real_hour']]
#    d31=d31.groupby('user_id').agg('max').reset_index()
#    d31=d31.rename(columns={'real_hour':'real_hour_max'})    
#    d30=pd.merge(d30,d31,on=['user_id'],how='left')
#    d30['label_is_lastest_hour_of_day_when_more_than2h']=(d30['real_hour']==d30['real_hour_max']).astype('int')    
#    d30=d30[['user_id','real_hour','label_is_lastest_hour_of_day_when_more_than2h']]
#    d30=d30.drop_duplicates()    
# 历史上是否浏览 过某城市
    d31=feature[['user_id','item_city_id']]
    d31=d31.drop_duplicates()    
    d32=other[['user_id','item_city_id']]
    d32=d32.drop_duplicates()        
    d32=pd.merge(d32,d31,on=['item_city_id','user_id'],how='inner')  
    d32['label_item_city_has_before']=1
    d32=pd.merge(other[['user_id','item_city_id']],d32,on=['user_id','item_city_id'],how='left')
    d32=d32.fillna(0)
    d32=d32.drop_duplicates()     
#历史上 用户 浏览某品牌/历史上所有品牌
    d33=other[['user_id','item_brand_id']]    
    d34=feature[['user_id','item_brand_id']]
    d35=d34.groupby('user_id').size().reset_index()
    d35=d35.rename(columns={0:'user_cnt'})
    d36=d34.groupby(['user_id','item_brand_id']).size().reset_index()
    d36=d36.rename(columns={0:'user_brand_cnt'})
    
    d33=pd.merge(d33,d35,on='user_id',how='left')
    d33=pd.merge(d33,d36,on=['user_id','item_brand_id'],how='left')
    d33['label_user_his_brand_rate']=d33['user_brand_cnt']/d33['user_cnt']
    d33=d33[['user_id','item_brand_id','label_user_his_brand_rate']]
    d33=d33.fillna(0)
    d33=d33.drop_duplicates()
#历史上 用户  浏览某城市/所有城市  
    d34=other[['user_id','item_city_id']]    
    d35=feature[['user_id','item_city_id']]
    d36=d35.groupby('user_id').size().reset_index()
    d36=d36.rename(columns={0:'user_cnt'})
    d37=d35.groupby(['user_id','item_city_id']).size().reset_index()
    d37=d37.rename(columns={0:'user_city_cnt'})
    
    d34=pd.merge(d34,d36,on='user_id',how='left')
    d34=pd.merge(d34,d37,on=['user_id','item_city_id'],how='left')
    d34['label_user_his_city_rate']=d34['user_city_cnt']/d34['user_cnt']
    d34=d34[['user_id','item_city_id','label_user_his_city_rate']]
    d34=d34.fillna(0)
    d34=d34.drop_duplicates()    
 # 某店铺商品的平均价格等级 有没有下降 
#商品 销量等级有没有上升
 
    d35=feature[['item_id','item_sales_level','real_day']]
    d35=d35.drop_duplicates(['item_id','item_sales_level'])
    d35=d35.replace(-1,100)

    t=d35.groupby('item_id').size().reset_index()
    t=t.rename(columns={0:'cnt'})
    t=t[t['cnt']>1]
    t=d35[d35['item_id'].isin(t['item_id'])]
    t1=t.groupby('item_id').agg({'real_day':'max'}).reset_index()
    t1=t1.rename(columns={'real_day':'real_day_max'})
    t1=pd.merge(t,t1,on='item_id',how='left')
    t1['is_del']=(t1['real_day']!=t1['real_day_max']).astype('int')
    t1=pd.merge(d35,t1,on=['real_day','item_id','item_sales_level'],how='left')    
    t1=t1.drop('real_day_max',axis=1)
    t1=t1.fillna(0)
    d35=t1[t1['is_del']==0]
    d35=d35.drop(['is_del','real_day'],axis=1)  

    d35=d35.rename(columns={'item_sales_level':'item_sales_level_before'})    
    d36=other[['item_id','item_sales_level']]
    d36=d36.drop_duplicates()
    d36=d36.rename(columns={'item_sales_level':'item_sales_level_now'})
    
    d37=pd.merge(d36,d35,on='item_id',how='inner')
    d37['label_item_sales_up']=(d37['item_sales_level_now']>d37['item_sales_level_before']).astype('int')
    d35=d37[['item_id','label_item_sales_up']]
    d35=d35.drop_duplicates()   
#商品 收藏次数等级 有没有上升    
    d36=feature[['item_id','item_collected_level','real_day']]
    d36=d36.drop_duplicates()
    d36=d36.replace(-1,100)

    t=d36.groupby('item_id').size().reset_index()
    t=t.rename(columns={0:'cnt'})
    t=t[t['cnt']>1]
    t=d36[d36['item_id'].isin(t['item_id'])]
    t1=t.groupby('item_id').agg({'real_day':'max'}).reset_index()
    t1=t1.rename(columns={'real_day':'real_day_max'})
    t1=pd.merge(t,t1,on='item_id',how='left')
    t1['is_del']=(t1['real_day']!=t1['real_day_max']).astype('int')
    t1=pd.merge(d36,t1,on=['real_day','item_id','item_collected_level'],how='left')    
    t1=t1.drop('real_day_max',axis=1)
    t1=t1.fillna(0)
    d36=t1[t1['is_del']==0]
    d36=d36.drop(['is_del','real_day'],axis=1)  
    
    d36=d36.rename(columns={'item_collected_level':'item_collected_level_before'})    
    d37=other[['item_id','item_collected_level']]
    d37=d37.drop_duplicates()
    d37=d37.rename(columns={'item_collected_level':'item_collected_level_now'})
    
    d38=pd.merge(d37,d36,on='item_id',how='inner')
    d38['label_item_collected_up']=(d38['item_collected_level_now']>d38['item_collected_level_before']).astype('int')
    d36=d38[['item_id','label_item_collected_up']]
    d36=d36.drop_duplicates()       
    
#历史上  用户在 一个小时中最后一次  点击发生购买的几率 用户 在一小时中  最后一次 购买转换率
    d40=feature[['user_id','real_time','real_hour','is_trade']]
    d40['cnt']=1
    
    d41=d40.groupby('user_id').size().reset_index()
    d41=d41.rename(columns={0:'cnt'})
    d41=d41[d41['cnt']!=1]    

    d40=d40[d40['user_id'].isin(d41['user_id'])]
    
    d41=d40.groupby(['user_id','real_hour']).agg({'real_time':'max'}).reset_index()
    d41=d41.rename(columns={'real_time':'real_time_max'})
    
    d40=pd.merge(d40,d41,on=['user_id','real_hour'],how='left')
    d40['feature_is_latest_time']=(d40['real_time']==d40['real_time_max']).astype('int')
    d40['feature_hour_is_latest_time_buy']=(d40['is_trade']&d40['feature_is_latest_time']).astype('int')#一天中的最后一次 并 购买
      
    d41=d40.groupby('user_id').agg({'feature_hour_is_latest_time_buy':'sum','is_trade':'sum','cnt':'sum'}).reset_index()
    d41=d41[d41['is_trade']>0]
    d41=d41.rename(columns={'feature_hour_is_latest_time_buy':'feature_is_latest_time_buy_sum','is_trade':'buy_sum','cnt':'look_sum'})
    d41['user_per_hour_trans_desier']=d41['feature_is_latest_time_buy_sum']/d41['buy_sum']
    d41['user_per_hour_trans_rate']=d41['feature_is_latest_time_buy_sum']/d41['look_sum']
        
    d40=pd.merge(d40,d41,on=['user_id'],how='left')
    d40=d40[['user_id','user_per_hour_trans_rate','user_per_hour_trans_desier']]    
    d40=d40.drop_duplicates()
    d40=d40.fillna(0)
# 相乘法    
    d41=pd.merge(d18,d40,on='user_id',how='left')
    d41['label_user_per_hour_trans_rate']=d41['label_is_latest_time']*d41['user_per_hour_trans_rate']
    d41=d41.fillna(0)
#    d41=d41.drop('real_time',axis=1)
#  现在是这个用户这天第几次  点击
#    d42=other[['user_id','real_time']]
#    d42=d42.drop_duplicates()
#    d42['label_user_ith_click']=0
#    
#    d43=d42.groupby('user_id')
#    d44=d43.size().reset_index()
#    d44=d44.rename(columns={0:'Size'})
#    d44=d44.drop(d44[d44.Size==1].index)[['user_id']]
#    d44=d44.reset_index(drop=True)
#    
#    for index,row in d44.iterrows():
#        d45=d43.get_group(d44.user_id[index])
#        d45=d45.reset_index()
#        d45=d45.rename(columns={'index':'Index'})
#        d45=d45.sort_index(axis=0,ascending=True,by='real_time')
#        d45=d45.reset_index(drop=True)
#        d42.label_user_ith_click[d45.Index]=d45.index
#    d42.label_user_ith_click=d42.label_user_ith_click+1
#    cnt=other.iloc[:,0].size    
#    d42['label_user_ith_click_normalize']= d42.label_user_ith_click/cnt
    
#用户 + 在这个时间点之前 有没有浏览过某商pin  
    d50=other[['user_id','real_time','item_id']]
    d51=d50.groupby(['user_id','item_id']).agg('min').reset_index()
    d51=d51.rename(columns={'real_time':'real_time_min'})
    
    d50=pd.merge(d50,d51,on=['user_id','item_id'],how='left')
    d50['label_item_id_has_ever']=(d50['real_time']>d50['real_time_min']).astype('int')
    d50=d50[['user_id','item_id','label_item_id_has_ever','real_time']]
    d50=d50.drop_duplicates() 
#用户 + 在这个时间点之前 有没有浏览过某商dian   
    d52=other[['user_id','real_time','shop_id']]
    d53=d52.groupby(['user_id','shop_id']).agg('min').reset_index()
    d53=d53.rename(columns={'real_time':'real_time_min'})
    
    d52=pd.merge(d52,d53,on=['user_id','shop_id'],how='left')
    d52['label_shop_id_has_ever']=(d52['real_time']>d52['real_time_min']).astype('int')
    d52=d52[['user_id','shop_id','label_shop_id_has_ever','real_time']]
    d52=d52.drop_duplicates()    

#  历史上是否 浏览过该shop
    d54=feature[['user_id','shop_id']]
    d54=d54.drop_duplicates()
    d55=other[['user_id','shop_id']]
    d55=d55.drop_duplicates()
    d55=pd.merge(d55,d54,on=['shop_id','user_id'],how='inner')  
    d55['label_shop_id_has_before']=1
    d55=pd.merge(other[['user_id','shop_id']],d55,on=['user_id','shop_id'],how='left')
    d55=d55.fillna(0)
    d55=d55.drop_duplicates()
##一天只有 单个小时 也算
    d60=other[['user_id','real_hour','real_time']]
    d60=d60.drop_duplicates()
    d61=d60.groupby(['user_id','real_hour']).size().reset_index()
    d61=d61.rename(columns={0:'cnt'})
    d61=d61.drop_duplicates()
    d60=pd.merge(d61,d60,on=['user_id','real_hour'],how='inner')
    
    d62=d60.groupby(['user_id','real_hour']).agg('max').reset_index()
    d62=d62.rename(columns={'real_time':'real_time_max'})    
    d60=pd.merge(d60,d62,on=['user_id','real_hour'],how='left')
    d60['label_is_latest_time000']=(d60['real_time']==d60['real_time_max']).astype('int')
    d60=d60[['user_id','real_time','label_is_latest_time000']]
    d60=d60.drop_duplicates()    
#    d42=d42.drop('real_time',axis=1)
#  历史上 用户  不同点击顺序的购买率 
#    d43=feature[['user_id','real_time']]
#    d44=other[['user_id']]
#    d43=d43[d43['user_id'].isin(d44['user_id'])]
#    d43=d43.drop_duplicates()
#    
#    d43['label_his_user_ith_click']=0
#    
#    d44=d43.groupby('user_id')
#    d45=d44.size().reset_index()
#    d45=d45.rename(columns={0:'Size'})
#    d45=d45.drop(d45[d45.Size==1].index)[['user_id']]
#    d45=d45.reset_index(drop=True)
#    
#    for index,row in d45.iterrows():
#        d46=d44.get_group(d45.user_id[index])
#        d46=d46.reset_index()
#        d46=d46.rename(columns={'index':'Index'})
#        d46=d46.sort_index(axis=0,ascending=True,by='real_time')
#        d46=d46.reset_index(drop=True)
#        d43.label_his_user_ith_click[d46.Index]=d46.index
#    d43.label_his_user_ith_click=d43.label_his_user_ith_click+1
#    
#    d44=feature[['user_id','real_time','is_trade']]
#    d44=d44.drop_duplicates()
#    d45=other[['user_id']]
#    d44=d44[d44['user_id'].isin(d45['user_id'])]
#    d44=pd.merge(d44,d43,on=['user_id','real_time'],how='left')
#    
#    d45=d44.groupby(['user_id','label_his_user_ith_click']).size().reset_index()

# 相乘法    
    
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
    other=pd.merge(other,d11,on='instance_id',how='left')
    other=pd.merge(other,d12,on='shop_id',how='left')
    other=pd.merge(other,d13,on='shop_id',how='left')
    other=pd.merge(other,d14,on='shop_id',how='left')
    other=pd.merge(other,d15,on='shop_id',how='left')
    other=pd.merge(other,d16,on='shop_id',how='left')
    other=pd.merge(other,d17,on='shop_id',how='left')

#    other=pd.merge(other,d18,on=['user_id','real_time'],how='left')
    other=pd.merge(other,d19,on=['user_id','item_brand_id','real_time'],how='left')
    other=pd.merge(other,d20,on=['user_id','item_city_id','real_time'],how='left')
    other=pd.merge(other,d21,on=['user_id','real_hour','item_price_level'],how='left')
    other=pd.merge(other,d22,on=['user_id','real_hour','shop_review_num_level'],how='left')
    other=pd.merge(other,d23,on='item_id',how='left')
    other=pd.merge(other,d24,on='item_id',how='left')
    other=pd.merge(other,d26,on=['user_id','item_id'],how='left')
    other=pd.merge(other,d28,on=['user_id','item_brand_id'],how='left')
    other=pd.merge(other,d29,on=['user_id','real_hour'],how='left')
#    other=pd.merge(other,d30,on=['user_id','real_hour'],how='left')
    other=pd.merge(other,d32,on=['user_id','item_city_id'],how='left')
    other=pd.merge(other,d33,on=['user_id','item_brand_id'],how='left')
    other=pd.merge(other,d34,on=['user_id','item_city_id'],how='left')
    other=pd.merge(other,d35,on='item_id',how='left')
    other=pd.merge(other,d36,on='item_id',how='left')
    
    other=pd.merge(other,d41,on=['user_id','real_time'],how='left')
#    other=pd.merge(other,d42,on=['user_id','real_time'],how='left')

    other=pd.merge(other,d50,on=['user_id','item_id','real_time'],how='left')
    other=pd.merge(other,d52,on=['user_id','shop_id','real_time'],how='left')
    other=pd.merge(other,d55,on=['user_id','shop_id'],how='left')
    other=pd.merge(other,d60,on=['user_id','real_time'],how='left')

    
    other=other.drop(['perdict_category','context_id','perdict_property','predict_category_property','context_timestamp','real_time','real_day','real_hour','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description','item_property_list','item_brand_id','item_category_list','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','user_gender_id','user_age_level','user_occupation_id','user_star_level'],axis=1)
    return other


#%%
"""
上下文信息
预测的类目属性列表中正确率
正确个数
不同正确个数的购买转化率
不同页的购买转化率
"""
def extract_context_label_feature(feature):
    context=feature[['is_trade','context_page_id','perdict_property','item_property_list']]
    context['label_predict_property_right_num']=context['item_property_list'].astype('str')+':'+context['perdict_property'].astype('str')
    context['label_predict_property_right_num']=context['label_predict_property_right_num'].apply(predictPropertyRightNum)
    
    d=context[['label_predict_property_right_num','is_trade']]
    d['cnt']=1
    d=d.groupby('label_predict_property_right_num').agg('sum').reset_index()    
    d['label_predict_property_right_num_buy_rate']=d['is_trade']/d['cnt']
    d=d[['label_predict_property_right_num','label_predict_property_right_num_buy_rate']]  
    return d
def extract_context_page_feature(feature):
    context=feature[['is_trade','context_page_id','perdict_property','item_property_list']]
    d=context[['context_page_id','is_trade']]
    d['cnt']=1
    d=d.groupby('context_page_id').agg('sum').reset_index()    
    d['context_page_buy_rate']=d['is_trade']/d['cnt']
    d=d[['context_page_id','context_page_buy_rate']] 
    return d

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
def extract_user_merchant_feature(dataset,feature):
    label=dataset[['instance_id','user_id','item_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level']]
    user_merchant=feature[['user_id','item_id','is_trade','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level']]
#用户——商品城市 购买转化率
    d=user_merchant[['user_id','item_city_id','is_trade']]
    d['cnt']=1
    d=d.groupby(['user_id','item_city_id']).agg('sum').reset_index()    
    d['user_item_city_buy_rate']=d['is_trade']/d['cnt']
    d=d[['user_id','item_city_id','user_item_city_buy_rate']]
#用户-商品品牌 购买转化率
    d1=user_merchant[['user_id','item_brand_id','is_trade']]
    d1['cnt']=1
    d1=d1.groupby(['user_id','item_brand_id']).agg('sum').reset_index()    
    d1['user_item_brand_buy_rate']=d1['is_trade']/d1['cnt']
    d1=d1[['user_id','item_brand_id','user_item_brand_buy_rate']]    
#用户-商品价格等级 购买转换率
    d2=user_merchant[['user_id','item_price_level','is_trade']]
    d2['cnt']=1
    d2=d2.groupby(['user_id','item_price_level']).agg('sum').reset_index()    
    d2['user_item_price_level_buy_rate']=d2['is_trade']/d2['cnt']
    d2=d2[['user_id','item_price_level','user_item_price_level_buy_rate']]
#用户-商品收藏等级 购买转换率
    d3=user_merchant[['user_id','item_sales_level','is_trade']]
    d3['cnt']=1
    d3=d3.groupby(['user_id','item_sales_level']).agg('sum').reset_index()    
    d3['user_item_sales_level_buy_rate']=d3['is_trade']/d3['cnt']
    d3=d3[['user_id','item_sales_level','user_item_sales_level_buy_rate']]    
#用户-商品销量等级 购买转换率
    d4=user_merchant[['user_id','item_collected_level','is_trade']]
    d4['cnt']=1
    d4=d4.groupby(['user_id','item_collected_level']).agg('sum').reset_index()    
    d4['user_item_collected_level_buy_rate']=d4['is_trade']/d4['cnt']
    d4=d4[['user_id','item_collected_level','user_item_collected_level_buy_rate']]  
#用户-展示次数 购买转换率
    d5=user_merchant[['user_id','item_pv_level','is_trade']]
    d5['cnt']=1
    d5=d5.groupby(['user_id','item_pv_level']).agg('sum').reset_index()    
    d5['user_item_pv_level_buy_rate']=d5['is_trade']/d5['cnt']
    d5=d5[['user_id','item_pv_level','user_item_pv_level_buy_rate']]
    
    label=pd.merge(label,d,on=['user_id','item_city_id'],how='left')
    label=pd.merge(label,d1,on=['user_id','item_brand_id'],how='left')       
    label=pd.merge(label,d2,on=['user_id','item_price_level'],how='left')       
    label=pd.merge(label,d3,on=['user_id','item_sales_level'],how='left')       
    label=pd.merge(label,d4,on=['user_id','item_collected_level'],how='left')       
    label=pd.merge(label,d5,on=['user_id','item_pv_level'],how='left')
    label=label.drop(['user_id','item_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level'],axis=1)  
    return label     
      
#%%
"""
用户——商店
用户——商店评价数量等级 购买转化率
用户——商店星级  购买转化率
用户——商店 好评数（需要计算） 购买转化率
"""
def extract_user_shop_feature(dataset,feature):
    label=dataset[['instance_id','user_id','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level']]
    user_shop=feature[['user_id','shop_id','is_trade','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']]  
#用户——商店 购买转化率
    d=user_shop[['user_id','shop_id','is_trade']]
    d['cnt']=1
    d=d.groupby(['user_id','shop_id']).agg('sum').reset_index()    
    d['user_shop_buy_rate']=d['is_trade']/d['cnt']
    d=d[['user_id','shop_id','user_shop_buy_rate']]    
#用户——商店评价数量等级 购买转化率
    d1=user_shop[['user_id','shop_review_num_level','is_trade']]
    d1['cnt']=1
    d1=d1.groupby(['user_id','shop_review_num_level']).agg('sum').reset_index()    
    d1['user_shop_review_num_level_buy_rate']=d1['is_trade']/d1['cnt']
    d1=d1[['user_id','shop_review_num_level','user_shop_review_num_level_buy_rate']]     
 #用户——商店商店星级 购买转化率
    d2=user_shop[['user_id','shop_star_level','is_trade']]
    d2['cnt']=1
    d2=d2.groupby(['user_id','shop_star_level']).agg('sum').reset_index()    
    d2['user_shop_star_level_buy_rate']=d2['is_trade']/d2['cnt']
    d2=d2[['user_id','shop_star_level','user_shop_star_level_buy_rate']]        
    
    label=pd.merge(label,d,on=['user_id','shop_id'],how='left')
    label=pd.merge(label,d1,on=['user_id','shop_review_num_level'],how='left')
    label=pd.merge(label,d2,on=['user_id','shop_star_level'],how='left')
    label=label.drop(['user_id','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level'],axis=1)  
    return label
#%%
def extract_hour_relate_feature(dataset,feature):
    label=dataset[['instance_id','real_hour','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level']]
#    union=feature[['real_hour','is_trade','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level']]
# 历史上 每小时的购买转换率  用 real_hour merge   可否进行合并？ 
    d=feature[['real_hour','is_trade']]
    d['cnt']=1
    d=d.groupby('real_hour').agg('sum').reset_index()
    d['hour_trans_rate']=d['is_trade']/d['cnt']
    d=d[['real_hour','hour_trans_rate']]
#历史上 每小时 发生购买的不同 用户/ 所有发生购买的不同用户 用 real_hour 进行merge
    d1=feature[['real_hour','user_id','is_trade']]
    
    d2=d1[d1['is_trade']==1]
    d2=d2[['user_id','is_trade']]
    d2=d2.drop_duplicates()
    cnt=d2.iloc[:,0].size
    
    d1=d1.drop_duplicates()
    d1=d1[d1['is_trade']==1]
    d1=d1.groupby('real_hour').size().reset_index()
    d1=d1.rename(columns={0:'dif_user'})
    d1['hour_trans_dif_user_rate']=d1['dif_user']/cnt
    d1=d1[['real_hour','hour_trans_dif_user_rate']]
# 历史上 不同小时 user_gender 的 转换率  用  real_hour 和user_gender_id 进行merge
    d2=feature[['user_gender_id','real_hour','is_trade']]
    d2['cnt']=1
    d2=d2.groupby(['user_gender_id','real_hour']).agg('sum').reset_index()
    d2['hour_dif_user_gender_trans_rate']=d2['is_trade']/d2['cnt']
    d2=d2[['real_hour','user_gender_id','hour_dif_user_gender_trans_rate']]
# 历史上 不同小时  user_occpuation的转换率
    d3=feature[['user_occupation_id','real_hour','is_trade']]
    d3['cnt']=1
    d3=d3.groupby(['user_occupation_id','real_hour']).agg('sum').reset_index()
    d3['hour_dif_user_occupation_trans_rate']=d3['is_trade']/d3['cnt']
    d3=d3[['real_hour','user_occupation_id','hour_dif_user_occupation_trans_rate']]
# 历史上 不同小时  user_age 的转换率
    d4=feature[['user_age_level','real_hour','is_trade']]
    d4['cnt']=1
    d4=d4.groupby(['user_age_level','real_hour']).agg('sum').reset_index()
    d4['hour_dif_user_age_level_trans_rate']=d4['is_trade']/d4['cnt']
    d4=d4[['real_hour','user_age_level','hour_dif_user_age_level_trans_rate']]    
# 历史上 不同小时  user-star的转换率    
    d5=feature[['user_star_level','real_hour','is_trade']]
    d5['cnt']=1
    d5=d5.groupby(['user_star_level','real_hour']).agg('sum').reset_index()
    d5['hour_dif_user_star_level_trans_rate']=d5['is_trade']/d5['cnt']
    d5=d5[['real_hour','user_star_level','hour_dif_user_star_level_trans_rate']]

    label=pd.merge(label,d,on='real_hour',how='left')
    label=pd.merge(label,d1,on='real_hour',how='left')
    label=pd.merge(label,d2,on=['real_hour','user_gender_id'],how='left')
    label=pd.merge(label,d3,on=['real_hour','user_occupation_id'],how='left')
    label=pd.merge(label,d4,on=['real_hour','user_age_level'],how='left')
    label=pd.merge(label,d5,on=['real_hour','user_star_level'],how='left')
   
    label=label.drop(['real_hour','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level'],axis=1)
    return label 
#%%
"""
当前用户 浏览的 商品 价格等级 是否高于 用户曾购买的平均价格等级（需要处理nan）
当前用户 浏览的 Item_sales_level 是否高于 用户曾购买的平均价格等级（不需处理nan）
当前用户 浏览的 Item_collected_level是否 高于 用户曾购买的平均首层等级 （不处理缺失）
当前用户 浏览的 shop_score_Service(不处理缺失)

当天 一共有多少不同商品
当天 一共有多少不同用户出现
当天 一共有多少不同商店出现
当天 该小时一共发生多少次点击
当天 该用户一共发生多少次点击

点击*用户的购买转化率

用户上一次购买距离这一次的时间
用户上一次点击距离这一次时间
 
当前用户浏览的 商品价格等级 是否低于  当天用户浏览的平均价格等级
"""
#def get_time_gap_before(s):
#    date_received,dates = s.split('-')
#    dates = dates.split(':')
#    gaps = []
#    for d in dates:
#        this_gap = (date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
#        if this_gap>0:
#            gaps.append(this_gap)
#    if len(gaps)==0:
#        return np.nan
#    else:
#        return min(gaps)
def extract_label_relate_feature(dataset,feature):
    label=dataset[['instance_id','real_hour','item_id','item_category_list','context_timestamp','user_id','item_price_level','item_sales_level','item_collected_level','shop_score_service']]
#当前用户 浏览的 商品 价格等级 是否高于 用户曾购买的平均价格等级（需要处理nan）       
    feature['second_category']=feature['item_category_list'].apply(splitItemCategory_second)
    label['second_category']=label['item_category_list'].apply(splitItemCategory_second)
    
    d=feature[['user_id','item_price_level','is_trade']]
    d=d[d['is_trade']==1]
    d=d.drop('is_trade',axis=1)
    d=round(d.groupby('user_id').agg('mean').reset_index())
    d=d.rename(columns={'item_price_level':'item_price_level_mean'})
        
    d2=label[['user_id','item_price_level']]
    d2=pd.merge(d2,d,on='user_id',how='left')
    d2['label_now_item_price_level_bigger']=(d2['item_price_level']<=d2['item_price_level_mean']).astype('int')
    d2=d2.fillna(100)
    d2['label_now_item_price_level_bigger'][d2[d2['item_price_level_mean']==100].index]=-1
    d2=d2.replace(100,-1)        
    d=d2[['user_id','item_price_level','label_now_item_price_level_bigger']]   
    d=d.drop_duplicates()
#   当前用户 浏览的 Item_sales_level 是否高于 用户曾购买的平均价格等级（不需处理nan）  
    d1=feature[['user_id','item_sales_level','is_trade']]
    d1=d1[d1['is_trade']==1]
    d1=d1.drop('is_trade',axis=1)
    d1=round(d1.groupby('user_id').agg('mean').reset_index())
    d1=d1.rename(columns={'item_sales_level':'item_sales_level_mean'})
        
    d2=label[['user_id','item_sales_level']]
    d2=pd.merge(d2,d1,on='user_id',how='left')
    d2['label_now_item_sales_level_bigger']=(d2['item_sales_level']<=d2['item_sales_level_mean']).astype('int')       
    d1=d2[['user_id','item_sales_level','label_now_item_sales_level_bigger']]   
    d1=d1.drop_duplicates()        
# 当前用户 浏览的 Item_collected_level是否 高于 用户曾购买的平均首层等级 （不处理缺失）       
    d3=feature[['user_id','item_collected_level','is_trade']]
    d3=d3[d3['is_trade']==1]
    d3=d3.drop('is_trade',axis=1)
    d3=round(d3.groupby('user_id').agg('mean').reset_index())
    d3=d3.rename(columns={'item_collected_level':'item_collected_level_mean'})
        
    d2=label[['user_id','item_collected_level']]
    d2=pd.merge(d2,d3,on='user_id',how='left')
    d2['label_now_item_collected_level_bigger']=(d2['item_collected_level']<=d2['item_collected_level_mean']).astype('int')
    d3=d2[['user_id','item_collected_level','label_now_item_collected_level_bigger']]   
    d3=d3.drop_duplicates()    
#当前用户 浏览的 shop_score_Service
    d4=feature[['user_id','shop_score_service','is_trade']]
    d4=d4[d4['is_trade']==1]
    d4=d4.drop('is_trade',axis=1)
    d4=round(d4.groupby('user_id').agg('mean').reset_index())
    d4=d4.rename(columns={'shop_score_service':'shop_score_service_mean'})
        
    d2=label[['user_id','shop_score_service']]
    d2=pd.merge(d2,d4,on='user_id',how='left')
    d2['shop_score_service_bigger']=(d2['shop_score_service']<=d2['shop_score_service_mean']).astype('int')
    d4=d2[['user_id','shop_score_service','shop_score_service_bigger']]   
    d4=d4.drop_duplicates()
#当天一共有多少不同商品出现  这个特征用在训练集label 是不同天时
    d5=label[['item_id']]
    d5=d5.drop_duplicates()
    d5=d5.iloc[:,0].size    
#当天一共有多少用户活跃 这个特征用在训练集label 是不同天时
    d6=label[['user_id']]
    d6=d6.drop_duplicates()
    d6=d6.iloc[:,0].size
#当天 一共有多少 商店 这个特征用在训练集label 是不同天时
    d7=label[['user_id']]
    d7=d7.drop_duplicates()
    d7=d7.iloc[:,0].size
#当天 该小时一共发生多少次点击
    d8=label[['real_hour']]
    d8=d8.groupby('real_hour').size().reset_index()
    d8=d8.rename(columns={0:'label_now_day_hour_how_many_click'})
#当天 该用户 一共发生多少次点击
    d9=label[['user_id']]
    d9=d9.groupby('user_id').size().reset_index()
    d9=d9.rename(columns={0:'label_now_day_user_how_many_click'})
#  点击*历史上用户的购买转化率
    d10=feature[['user_id','is_trade']]
    d10['cnt']=1
    d10=d10.groupby('user_id').agg('sum').reset_index()
    d10['rate']=d10['is_trade']/d10['cnt']
    d10=pd.merge(d9,d10,on='user_id',how='left')
    d10['label_now_userclick_dot_rate']=d10['label_now_day_user_how_many_click']*d10['rate']
    d10=d10[['user_id','label_now_userclick_dot_rate','label_now_day_user_how_many_click']]
#用户上一次点击距离这一次时间 
#    d11=label[['user_id','context_timestamp']]
    
# 历史上用户上一次购买距离这一次的时间    
    
# 用户当前 浏览的商品 价格 是否  二级类目下 用户 浏览过/买过的平均价格等级   用sinstance_id
    d13=feature[['second_category','user_id','item_price_level','is_trade']]
    d14=d13[d13['is_trade']==1]
    d14=d14[['second_category','user_id','item_price_level']]
    d14=d14.groupby(['second_category','user_id']).agg('mean').reset_index()
    d14=d14.rename(columns={'item_price_level':'item_price_level_mean'})
    
    d15=label[['instance_id','second_category','user_id','item_price_level']]
        
    d13=pd.merge(d15,d14,on=['second_category','user_id'],how='left')
    d13['label_now_user_cat2_item_price_lower']=(d13['item_price_level']<=d13['item_price_level_mean']).astype('int')
    
    d13=d13.fillna(-1)
    d13['label_now_user_cat2_item_price_lower'][d13[d13['item_price_level_mean']==-1].index]=-1
    d13=d13[['instance_id','label_now_user_cat2_item_price_lower']]
    d13=d13.drop_duplicates()
         
    label['label_now_day_how_many_items']=d5
    label['label_now_day_how_many_users']=d6
    label['label_now_day_how_many_shop']=d7
    label=pd.merge(label,d,on=['item_price_level','user_id'],how='left')
    label=pd.merge(label,d1,on=['item_sales_level','user_id'],how='left')
    label=pd.merge(label,d3,on=['item_collected_level','user_id'],how='left')
    label=pd.merge(label,d4,on=['shop_score_service','user_id'],how='left')
    label=pd.merge(label,d8,on='real_hour',how='left')
    label=pd.merge(label,d10,on='user_id',how='left')
    label=pd.merge(label,d13,on='instance_id',how='left')

    #这个drop要注意改
    label=label.drop(['second_category','real_hour','item_id','item_category_list','context_timestamp','user_id','item_price_level','item_sales_level','item_collected_level','shop_score_service'],axis=1)
    return label  
#%%

feature1_2_3=pd.concat([feature1,feature2,feature3]).reset_index(drop=True)
feature2_3_4=pd.concat([feature2,feature3,feature4]).reset_index(drop=True)
feature3_4_5=pd.concat([feature3,feature4,feature5]).reset_index(drop=True)
feature4_5_6=pd.concat([feature4,feature5,feature6]).reset_index(drop=True)
feature5_6_7=pd.concat([feature5,feature6,feature7]).reset_index(drop=True)

dataset1=feature4
dataset2=feature5
dataset3=feature6
dataset4=feature7
dataset5=test
#%%
user1=  extract_user_feature(dataset1,feature1_2_3)  
user2=  extract_user_feature(dataset2,feature2_3_4)    
user3=  extract_user_feature(dataset3,feature3_4_5) 
user4=  extract_user_feature(dataset4,feature4_5_6) 
user5=  extract_user_feature(dataset5,feature5_6_7) 

user1=user1.drop('user_id',axis=1) 
user2=user2.drop('user_id',axis=1) 
user3=user3.drop('user_id',axis=1) 
user4=user4.drop('user_id',axis=1) 
user5=user5.drop('user_id',axis=1) 

#%%  
user1.to_csv('data/user1.csv',index=None)
user2.to_csv('data/user2.csv',index=None)
user3.to_csv('data/user3.csv',index=None)
user4.to_csv('data/user4.csv',index=None)
user5.to_csv('data/user5.csv',index=None)


#%%
shop_feature1=extract_shop_feature(dataset1,feature1_2_3)
shop_feature2=extract_shop_feature(dataset2,feature2_3_4)
shop_feature3=extract_shop_feature(dataset3,feature3_4_5)
shop_feature4=extract_shop_feature(dataset4,feature4_5_6)
shop_feature5=extract_shop_feature(dataset5,feature5_6_7)

shop_feature1=shop_feature1.drop('shop_id',axis=1)
shop_feature2=shop_feature2.drop('shop_id',axis=1)
shop_feature3=shop_feature3.drop('shop_id',axis=1)
shop_feature4=shop_feature4.drop('shop_id',axis=1)
shop_feature5=shop_feature5.drop('shop_id',axis=1)

#%%
shop_feature1.to_csv('data/shop_feature1.csv',index=None)
shop_feature2.to_csv('data/shop_feature2.csv',index=None)
shop_feature3.to_csv('data/shop_feature3.csv',index=None)
#%%
merchant_feature1=extract_label_merchant_feature(dataset1,feature1_2_3)
merchant_feature2=extract_label_merchant_feature(dataset2,feature2_3_4)
merchant_feature3=extract_label_merchant_feature(dataset3,feature3_4_5)
merchant_feature4=extract_label_merchant_feature(dataset4,feature4_5_6)
merchant_feature5=extract_label_merchant_feature(dataset5,feature5_6_7)

merchant_feature1=merchant_feature1.drop('item_id',axis=1)
merchant_feature2=merchant_feature2.drop('item_id',axis=1)
merchant_feature3=merchant_feature3.drop('item_id',axis=1)
merchant_feature4=merchant_feature4.drop('item_id',axis=1)
merchant_feature5=merchant_feature5.drop('item_id',axis=1)

#%%
merchant_feature1.to_csv('data/merchant_feature1.csv',index=None)
merchant_feature2.to_csv('data/merchant_feature2.csv',index=None)
merchant_feature3.to_csv('data/merchant_feature3.csv',index=None)
#%%
other1= extract_other_feature(feature1_2_3,dataset1)
other2= extract_other_feature(feature2_3_4,dataset2)
other3= extract_other_feature(feature3_4_5,dataset3)
other4= extract_other_feature(feature4_5_6,dataset4)
other5= extract_other_feature(feature5_6_7,dataset5)


other1=pd.merge(other1,dataset1[['instance_id','real_hour']],on='instance_id',how='left')
other2=pd.merge(other2,dataset2[['instance_id','real_hour']],on='instance_id',how='left')
other3=pd.merge(other3,dataset3[['instance_id','real_hour']],on='instance_id',how='left')
other4=pd.merge(other4,dataset4[['instance_id','real_hour']],on='instance_id',how='left')
other5=pd.merge(other5,dataset5[['instance_id','real_hour']],on='instance_id',how='left')

#%%
#注意这里  有问题  应该提取？？    一天还是N天  想提取N天的
context_label1= extract_context_label_feature(feature1_2_3) 
context_label2= extract_context_label_feature(feature2_3_4) 
context_label3= extract_context_label_feature(feature3_4_5) 
context_label4= extract_context_label_feature(feature4_5_6) 
context_label5= extract_context_label_feature(feature5_6_7) 

context_page1= extract_context_page_feature(feature1_2_3) 
context_page2= extract_context_page_feature(feature2_3_4) 
context_page3= extract_context_page_feature(feature3_4_5) 
context_page4= extract_context_page_feature(feature4_5_6) 
context_page5= extract_context_page_feature(feature5_6_7) 

other1=pd.merge(other1,context_label1,on='label_predict_property_right_num',how='left')
other2=pd.merge(other2,context_label2,on='label_predict_property_right_num',how='left')
other3=pd.merge(other3,context_label3,on='label_predict_property_right_num',how='left')
other4=pd.merge(other4,context_label4,on='label_predict_property_right_num',how='left')
other5=pd.merge(other5,context_label5,on='label_predict_property_right_num',how='left')

other1=pd.merge(other1,context_page1,on='context_page_id',how='left')
other2=pd.merge(other2,context_page2,on='context_page_id',how='left')
other3=pd.merge(other3,context_page3,on='context_page_id',how='left')
other4=pd.merge(other4,context_page4,on='context_page_id',how='left')
other5=pd.merge(other5,context_page5,on='context_page_id',how='left')

#%% 
other1.to_csv('data/other1.csv',index=None)
other2.to_csv('data/other2.csv',index=None)
other3.to_csv('data/other3.csv',index=None)
#%%
user_merchent1= extract_user_merchant_feature(dataset1,feature1_2_3)  
user_merchent2= extract_user_merchant_feature(dataset2,feature2_3_4)  
user_merchent3= extract_user_merchant_feature(dataset3,feature3_4_5)  
user_merchent4= extract_user_merchant_feature(dataset4,feature4_5_6)  
user_merchent5= extract_user_merchant_feature(dataset5,feature5_6_7)  

#%%
user_merchent1.to_csv('data/user_merchant1.csv',index=None) 
user_merchent2.to_csv('data/user_merchant2.csv',index=None)    
user_merchent3.to_csv('data/user_merchant3.csv',index=None) 
#%%    
user_shop1= extract_user_shop_feature(dataset1,feature1_2_3)  
user_shop2= extract_user_shop_feature(dataset2,feature2_3_4)  
user_shop3= extract_user_shop_feature(dataset3,feature3_4_5) 
user_shop4= extract_user_shop_feature(dataset4,feature4_5_6)  
user_shop5= extract_user_shop_feature(dataset5,feature5_6_7)  
 
#%%
user_shop1.to_csv('data/user_shop1.csv',index=None) 
user_shop2.to_csv('data/user_shop2.csv',index=None)    
user_shop3.to_csv('data/user_shop3.csv',index=None)       

#%%
hour_related_feature1=extract_hour_relate_feature(dataset1,feature1_2_3) 
hour_related_feature2=extract_hour_relate_feature(dataset2,feature2_3_4) 
hour_related_feature3=extract_hour_relate_feature(dataset3,feature3_4_5) 
hour_related_feature4=extract_hour_relate_feature(dataset4,feature4_5_6) 
hour_related_feature5=extract_hour_relate_feature(dataset5,feature5_6_7) 

#%%
hour_related_feature1.to_csv('data/hour_related_feature1.csv')
hour_related_feature2.to_csv('data/hour_related_feature2.csv')
hour_related_feature3.to_csv('data/hour_related_feature3.csv')     
#%%
label_relate_feature1=extract_label_relate_feature(dataset1,feature1_2_3)
label_relate_feature2=extract_label_relate_feature(dataset2,feature2_3_4)
label_relate_feature3=extract_label_relate_feature(dataset3,feature3_4_5)
label_relate_feature4=extract_label_relate_feature(dataset4,feature4_5_6)
label_relate_feature5=extract_label_relate_feature(dataset5,feature5_6_7)

#%%
label_relate_feature1.to_csv('data/label_relate_feature1.csv')
label_relate_feature2.to_csv('data/label_relate_feature2.csv')
label_relate_feature3.to_csv('data/label_relate_feature3.csv') 
#%%
#读取
other1=pd.read_csv('data/other1.csv')
other2=pd.read_csv('data/other2.csv')
other3=pd.read_csv('data/other3.csv')

merchant_feature1=pd.read_csv('data/merchant_feature1.csv')
merchant_feature2=pd.read_csv('data/merchant_feature2.csv')
merchant_feature3=pd.read_csv('data/merchant_feature3.csv')

shop_feature1=pd.read_csv('data/shop_feature1.csv')
shop_feature2=pd.read_csv('data/shop_feature2.csv')
shop_feature3=pd.read_csv('data/shop_feature3.csv')

user1=pd.read_csv('data/user1.csv')
user2=pd.read_csv('data/user2.csv')
user3=pd.read_csv('data/user3.csv') 

user_merchent1=pd.read_csv('data/user_merchant1.csv')   
user_merchent2=pd.read_csv('data/user_merchant2.csv')   
user_merchent3=pd.read_csv('data/user_merchant3.csv')   

user_shop1=pd.read_csv('data/user_shop1.csv')   
user_shop2=pd.read_csv('data/user_shop2.csv')   
user_shop3=pd.read_csv('data/user_shop3.csv') 

hour_related_feature1=pd.read_csv('data/hour_related_feature1.csv')
hour_related_feature2=pd.read_csv('data/hour_related_feature2.csv')
hour_related_feature3=pd.read_csv('data/hour_related_feature3.csv')

label_relate_feature1=pd.read_csv('data/label_relate_feature1.csv')
label_relate_feature2=pd.read_csv('data/label_relate_feature2.csv')
label_relate_feature3=pd.read_csv('data/label_relate_feature3.csv')
#%%

merchant_feature1=merchant_feature1.drop_duplicates()
merchant_feature2=merchant_feature2.drop_duplicates()
merchant_feature3=merchant_feature3.drop_duplicates()
merchant_feature4=merchant_feature4.drop_duplicates()
merchant_feature5=merchant_feature5.drop_duplicates()

shop_feature1=shop_feature1.drop_duplicates()
shop_feature2=shop_feature2.drop_duplicates()
shop_feature3=shop_feature3.drop_duplicates()
shop_feature4=shop_feature4.drop_duplicates()
shop_feature5=shop_feature5.drop_duplicates()

user1=user1.drop_duplicates()
user2=user2.drop_duplicates()
user3=user3.drop_duplicates()
user4=user4.drop_duplicates()
user5=user5.drop_duplicates()

user_shop1=user_shop1.drop_duplicates()
user_shop2=user_shop2.drop_duplicates()
user_shop3=user_shop3.drop_duplicates()
user_shop4=user_shop4.drop_duplicates()
user_shop5=user_shop5.drop_duplicates()

user_merchent1=user_merchent1.drop_duplicates()
user_merchent2=user_merchent2.drop_duplicates()
user_merchent3=user_merchent3.drop_duplicates()
user_merchent4=user_merchent4.drop_duplicates()
user_merchent5=user_merchent5.drop_duplicates()

#%%
dataset1= pd.merge(other1,merchant_feature1,on='instance_id',how='left',copy=False)
dataset1= pd.merge(dataset1,shop_feature1,on='instance_id',how='left',copy=False)
dataset1= pd.merge(dataset1,user1,on='instance_id',how='left',copy=False)
dataset1= pd.merge(dataset1,user_merchent1,on='instance_id',how='left',copy=False)
dataset1= pd.merge(dataset1,user_shop1,on='instance_id',how='left',copy=False)
dataset1= pd.merge(dataset1,hour_related_feature1,on='instance_id',how='left',copy=False)
dataset1= pd.merge(dataset1,label_relate_feature1,on='instance_id',how='left',copy=False)

dataset2= pd.merge(other2,merchant_feature2,on='instance_id',how='left')
dataset2= pd.merge(dataset2,shop_feature2,on='instance_id',how='left')
dataset2= pd.merge(dataset2,user2,on='instance_id',how='left')
dataset2= pd.merge(dataset2,user_merchent2,on='instance_id',how='left')
dataset2= pd.merge(dataset2,user_shop2,on='instance_id',how='left')
dataset2= pd.merge(dataset2,hour_related_feature1,on='instance_id',how='left',copy=False)
dataset2= pd.merge(dataset2,label_relate_feature2,on='instance_id',how='left',copy=False)

dataset3= pd.merge(other3,merchant_feature3,on='instance_id',how='left')
dataset3= pd.merge(dataset3,shop_feature3,on='instance_id',how='left')
dataset3= pd.merge(dataset3,user3,on='instance_id',how='left')
dataset3= pd.merge(dataset3,user_merchent3,on='instance_id',how='left')
dataset3= pd.merge(dataset3,user_shop3,on='instance_id',how='left')
dataset3= pd.merge(dataset3,hour_related_feature3,on='instance_id',how='left',copy=False)
dataset3= pd.merge(dataset3,label_relate_feature3,on='instance_id',how='left',copy=False)

dataset4= pd.merge(other4,merchant_feature4,on='instance_id',how='left')
dataset4= pd.merge(dataset4,shop_feature4,on='instance_id',how='left')
dataset4= pd.merge(dataset4,user4,on='instance_id',how='left')
dataset4= pd.merge(dataset4,user_merchent4,on='instance_id',how='left')
dataset4= pd.merge(dataset4,user_shop4,on='instance_id',how='left')
dataset4= pd.merge(dataset4,hour_related_feature4,on='instance_id',how='left',copy=False)
dataset4= pd.merge(dataset4,label_relate_feature4,on='instance_id',how='left',copy=False)

dataset5= pd.merge(other5,merchant_feature5,on='instance_id',how='left')
dataset5= pd.merge(dataset5,shop_feature5,on='instance_id',how='left')
dataset5= pd.merge(dataset5,user5,on='instance_id',how='left')
dataset5= pd.merge(dataset5,user_merchent5,on='instance_id',how='left')
dataset5= pd.merge(dataset5,user_shop5,on='instance_id',how='left')
dataset5= pd.merge(dataset5,hour_related_feature5,on='instance_id',how='left',copy=False)
dataset5= pd.merge(dataset5,label_relate_feature5,on='instance_id',how='left',copy=False)

dataset1=dataset1.fillna(value=-1)
dataset2=dataset2.fillna(value=-1)
dataset3=dataset3.fillna(value=-1)
dataset4=dataset4.fillna(value=-1)
dataset5=dataset5.fillna(value=-1)

dataset1=dataset1.drop(['item_id','shop_id','user_id'],axis=1)
dataset2=dataset2.drop(['item_id','shop_id','user_id'],axis=1)
dataset3=dataset3.drop(['item_id','shop_id','user_id'],axis=1)
dataset4=dataset4.drop(['item_id','shop_id','user_id'],axis=1)
dataset5=dataset5.drop(['item_id','shop_id','user_id'],axis=1)

#%%
dataset1.to_csv('data/dataset1.csv',index=None)
dataset2.to_csv('data/dataset2.csv',index=None)
dataset3.to_csv('data/dataset3.csv',index=None)
dataset4.to_csv('data/dataset4.csv',index=None)
dataset5.to_csv('data/dataset4.csv',index=None)




   