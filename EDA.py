# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 18:54:49 2018

@author: surface
"""
import matplotlib.pyplot as plt
import seaborn as sns
featureUsed=['label_user_per_hour_trans_rate','user_per_hour_trans_rate','user_per_hour_trans_desier','label_is_latest_time','is_trade']
corrmat=other[featureUsed].corr()
plt.figure()
sns.heatmap(corrmat)