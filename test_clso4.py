# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:22:03 2022

@author: ma
"""

# Importing the required libraries
import xgboost as xgb
import pandas as pd
import scipy.stats as stats
# First XGBoost model for Pima Indians dataset
import numpy as np
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.metrics import r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
import random
# Reading the csv file and putting it into 'df' object
df = pd.read_csv('mix_clso4_all2.csv')
#df = pd.read_csv('pore_na2so4.csv')
df.head()

# Putting feature variable to X
X = df.drop('clso4',axis=1)
# Putting response variable to y
y = df['clso4']
#X = df.drop('Na2SO4',axis=1)
### Putting response variable to y
#y = df['Na2SO4']
r_score1,r_score2,r_score3=[],[],[]
random2r=[]
rmse_score=[]
rmse_score1,rmse_score2,rmse_score3=[],[],[]
result,result_all=[],[]
#choice=[]
#for i in range(40,180,2):
#    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, train_size=0.8, random_state=i)#89
#    X_train_all.shape, X_test.shape
#    
#    
## now lets split the data into train and test
##Splitting the data into train and test
##=========================================================================================
#    for j in range(40,200,2):
#        X_train, X_ver, y_train, y_ver = train_test_split(X_train_all, y_train_all, train_size=0.8, random_state=j)
#        X_train.shape, y_ver.shape
#        
#        for k in range(5,7,1):
#            k=k/100
#            
#            for l in range(4,7,1):
#                
#                model = xgb.XGBRegressor(n_estimators=180, 
#                         learning_rate=k, 
#                         max_depth=l, 
#                         silent=True, 
#                         objective='reg:squarederror',
#                         random_state=7)
#                model.fit(X_train, y_train)
#                 # 对测试集进行预测
#                y_pred = model.predict(X_ver)
#                ## evaluate predictions
#                pearson_r=stats.pearsonr(y_ver,y_pred)
#                R2=metrics.r2_score(y_ver,y_pred)
#                RMSE=metrics.mean_squared_error(y_ver,y_pred)**0.5
#                print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(pearson_r[0],RMSE))
#                print ('r2_score: %.2f' %R2)
#                result=[]
#                result.append(i)
#                result.append(j)
#                result.append(k)
#                result.append(l)
#                result.append(R2)
#                result.append(RMSE)
#                result_all.append(result)
###########################################################################
#
#
#for i in result_all:
#    if i[-2]>0.8:
#        
#        choice.append(i)
#
#                
#######################################################################            
#test_r,test_rmse=[],[]
#for i in choice:
#    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, train_size=0.8, random_state=i[0])#89
#    X_train_all.shape, X_test.shape
#    X_train, X_ver, y_train, y_ver = train_test_split(X_train_all, y_train_all, train_size=0.8, random_state=i[1])
#    X_train.shape, y_ver.shape
#    model = xgb.XGBRegressor(n_estimators=180, 
#                         learning_rate=i[2], 
#                         max_depth=i[3], 
#                         silent=True, 
#                         objective='reg:squarederror',
#                         random_state=7)  #reg:gamma squarederror
#    model.fit(X_train, y_train)      
#    y_pred = model.predict(X_test)
#    random_forest_error=y_pred-y_test    
#    # evaluate predictions
#    from sklearn import metrics
#    print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test,y_pred))
#    pearson_r=stats.pearsonr(y_test,y_pred)
#    R2=metrics.r2_score(y_test,y_pred)
#    RMSE=metrics.mean_squared_error(y_test,y_pred)**0.5
#    test_r.append(R2)
#    test_rmse.append(RMSE)


######################################################################################
##66 116
#
#X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, train_size=0.8, random_state=136)#nacl_117,na2so4_57,71,mgso4_31,mgcl2_141
#X_train_all.shape, X_test.shape
#X_train, X_ver, y_train, y_ver = train_test_split(X_train_all, y_train_all, train_size=0.8, random_state=108)#nacl_45,na2so4_193,33,mgso4_169,mgcl2_45
#X_train.shape, y_ver.shape
#model = xgb.XGBRegressor(n_estimators=180, 
#                         learning_rate=0.06,#0nacl_0.07,na2so4_0.05,0.05,mgso4_0.06,mgcl2_0.05
#                         max_depth=6, #nacl_5,na2so4_7,7,mgso4_5,mgcl2_5
#                         silent=True, 
#                         objective='reg:squarederror',
#                         random_state=7,
#                         gamma=0,
#                         importance_type='total_gain')  #reg:gamma squarederror
#model.fit(X_train, y_train)
##################################################################################################
#random_forest_predict1=model.predict(X_ver)
#random_forest_error1=random_forest_predict1-y_ver
## Verify the accuracy
#from sklearn import metrics
#print('Mean Absolute Error: ', metrics.mean_absolute_error(y_ver,random_forest_predict1))
#pearson_r1=stats.pearsonr(y_ver,random_forest_predict1)
#R21=metrics.r2_score(y_ver,random_forest_predict1)
#RMSE1=metrics.mean_squared_error(y_ver,random_forest_predict1)**0.5
#
##Draw test plot
#font = {"color": "darkred",
#        "size": 18,
#        "family" : "times new roman"}
#font1 = {"color": "black",
#        "size": 12,
#        "family" : "times new roman"}
#
#Text='r='+str(round(pearson_r1[0],2))
#plt.figure(3)
#plt.clf()
#ax=plt.axes(aspect='equal')
#plt.scatter(y_ver,random_forest_predict1,color='red')
#plt.xlabel('True Values',fontdict=font)
#plt.ylabel('Predictions',fontdict=font)
#Lims=[0,110]
#plt.xlim(Lims)
#plt.ylim(Lims)
#plt.tick_params(labelsize=10)
#plt.plot(Lims,Lims,color='black')
#plt.grid(False)
#plt.title('ion',fontdict=font)
#plt.text(2,10,Text,fontdict=font1)
#plt.savefig('figure3.png', dpi=100,bbox_inches='tight') 
#
#################################################################################################
## 对测试集进行预测
#y_pred = model.predict(X_test)
#random_forest_error=y_pred-y_test
## evaluate predictions
#from sklearn import metrics
#print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test,y_pred))
#pearson_r=stats.pearsonr(y_test,y_pred)
#R2=metrics.r2_score(y_test,y_pred)
#RMSE=metrics.mean_squared_error(y_test,y_pred)**0.5
#print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(pearson_r[0],RMSE))
#print ('r2_score: %.2f' %R2)
#rmse_score.append(RMSE)    
##Draw test plot
#font = {"color": "darkred",
#        "size": 18,
#        "family" : "times new roman"}
#font1 = {"color": "black",
#        "size": 12,
#        "family" : "times new roman"}
#
#Text='r='+str(round(pearson_r[0],2))
#plt.figure(1)
#plt.clf()
#ax=plt.axes(aspect='equal')
#plt.scatter(y_test,y_pred,color='red')
#plt.xlabel('True Values',fontdict=font)
#plt.ylabel('Predictions',fontdict=font)
#Lims=[0,110]
#plt.xlim(Lims)
#plt.ylim(Lims)
#plt.tick_params(labelsize=10)
#plt.plot(Lims,Lims,color='black')
#plt.grid(False)
#plt.title('ion',fontdict=font)
#plt.text(2,10,Text,fontdict=font1)
#plt.savefig('figure1.png', dpi=100,bbox_inches='tight')   
#
#
#plt.figure(2)
#plt.clf()
#plt.hist(random_forest_error,bins=30)
#plt.xlabel('Prediction Error',fontdict=font)
#plt.ylabel('Count',fontdict=font)
#plt.grid(False)
#plt.title('ion',fontdict=font)
#plt.savefig('figure2.png', dpi=100,bbox_inches='tight')
#print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(pearson_r[0],RMSE))
#
## 显示重要特征
#plot_importance(model,importance_type=('total_gain'))
#pyplot.show()
#feature_importance = model.feature_importances_.tolist()
##Calculate the importance of variables

################################################################################################重复做实验
choice_repeat=[]
for i in range(0,300,1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=i)#89
    X_train.shape, X_test.shape
    
    
# now lets split the data into train and test
#Splitting the data into train and test
#=========================================================================================


    model1 = xgb.XGBRegressor(n_estimators=180, 
                         learning_rate=0.06, 
                         max_depth=6, 
                         silent=True, 
                         objective='reg:squarederror',
                         random_state=7,
                         importance_type='total_gain')
    model1.fit(X_train, y_train)
                 # 对测试集进行预测
    y_pred = model1.predict(X_test)
#################################################################################################
    random_forest_predict1=model1.predict(X_train)
    random_forest_error1=random_forest_predict1-y_train
    # Verify the accuracy
    from sklearn import metrics
    mae1=metrics.mean_absolute_error(y_train,random_forest_predict1)
    pearson_r1=stats.pearsonr(y_train,random_forest_predict1)
    R21=metrics.r2_score(y_train,random_forest_predict1)
    RMSE1=metrics.mean_squared_error(y_train,random_forest_predict1)**0.5
###########################################################################################################
        ## evaluate predictions
    pearson_r=stats.pearsonr(y_test,y_pred)
    mae=metrics.mean_absolute_error(y_test,y_pred)
    R2=metrics.r2_score(y_test,y_pred)
    RMSE=metrics.mean_squared_error(y_test,y_pred)**0.5
    print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(pearson_r[0],RMSE))
    print ('r2_score: %.2f' %R2)
    result=[]
    
    result.append(R2)
    result.append(i)
    result.append(mae)
    result.append(RMSE)
    
    feature_importance1 = model1.feature_importances_.tolist()
    for j in feature_importance1:
        result.append(j)
        
    result.append(R21)
    result.append(mae1)
    result.append(RMSE1)
    choice_repeat.append(result)
    

resule_repeat_choice=sorted(choice_repeat,reverse=True)    
sum_mae,sum_r2,sum_rmse,sum_rd,sum_zeta,sum_bar,sum_con_cl,sum_con_so4,sum_mol_ratio,sum1_mae,sum1_r2,sum1_rmse=[],[],[],[],[],[],[],[],[],[],[],[]

for i in resule_repeat_choice[:50]:
    sum_mae.append(i[2])
    sum_r2.append(i[0])
    sum_rmse.append(i[3])
    sum_rd.append(i[4])
    sum_zeta.append(i[5])
    sum_bar.append(i[6])
    sum_con_cl.append(i[7])
    sum_con_so4.append(i[8])
    sum_mol_ratio.append(i[9])
    sum1_r2.append(i[10])
    sum1_mae.append(i[11])
    sum1_rmse.append(i[12])
  
all_mae=np.array(sum_mae)
mean_mae=np.mean(all_mae)
std_mae=np.std(all_mae,ddof = 1)

all_r2=np.array(sum_r2)
mean_r2=np.mean(all_r2)
std_r2=np.std(all_r2,ddof = 1)

all_rmse=np.array(sum_rmse)
mean_rmse=np.mean(all_rmse)
std_rmse=np.std(all_rmse,ddof = 1)

all_rd=np.array(sum_rd)
mean_rd=np.mean(all_rd)
std_rd=np.std(all_rd,ddof = 1)

all_zeta=np.array(sum_zeta)
mean_zeta=np.mean(all_zeta)
std_zeta=np.std(all_zeta,ddof = 1)

all_bar=np.array(sum_bar)
mean_bar=np.mean(all_bar)
std_bar=np.std(all_bar,ddof = 1)

all_con_cl=np.array(sum_con_cl)
mean_con_cl=np.mean(all_con_cl)
std_con_cl=np.std(all_con_cl,ddof = 1)

all_con_so4=np.array(sum_con_so4)
mean_con_so4=np.mean(all_con_so4)
std_con_so4=np.std(all_con_so4,ddof = 1)

all_ra=np.array(sum_mol_ratio)
mean_ra=np.mean(all_ra)
std_cra=np.std(all_ra,ddof = 1)

all_mae1=np.array(sum1_mae)
mean_mae1=np.mean(all_mae1)
std_mae1=np.std(all_mae1,ddof = 1)

all_r21=np.array(sum1_r2)
mean_r21=np.mean(all_r21)
std_r21=np.std(all_r21,ddof = 1)

all_rmse1=np.array(sum1_rmse)
mean_rmse1=np.mean(all_rmse1)
std_rmse1=np.std(all_rmse1,ddof = 1)