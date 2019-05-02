# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:47:01 2019

@author: dr36495
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## read csv file
## setting index column to 0 so that the first column can be the index column
KNN_df= pd.read_csv("C:/Users/DR36495/Documents/MyGit/KNN_Project/14-K-Nearest-Neighbors/KNN_Project_Data",index_col=0)
print(KNN_df.head())

 plot some graphs for missing data
sns.heatmap(KNN_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
##conclusion no missing data

## creating pairplot  indicated by the target colmn
sns.pairplot(KNN_df,hue='TARGET CLASS',palette='coolwarm')
## standardize the variables
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(KNN_df.drop('TARGET CLASS',axis=1))
scaled_features=scaler.transform(KNN_df.drop('TARGET CLASS',axis=1))
df_features=pd.DataFrame(scaled_features,columns=KNN_df.columns[:-1])
print(df_features)

## Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df_features,KNN_df['TARGET CLASS'],test_size=0.30)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
#
##evaluation
from sklearn.metrics import classification_report,confusion_matrix
#
print(classification_report(y_test,pred),confusion_matrix(y_test,pred))

error_rate=[]

for i in range(1,40):
    knn_i=KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(X_train,y_train)
    pred_i=knn_i.predict(X_test)
    #print(i,pred_i,y_test)
    error_rate.append(np.mean(pred_i!=y_test))
print(error_rate)
    
plt.figure(figsize=(10,8))
plt.plot (range(1,40),error_rate)
plt.xlabel('K')
plt.ylabel('Error Rate')



# for k =20
knn_20=KNeighborsClassifier(n_neighbors=20)
knn_20.fit(X_train,y_train)
pred_20=knn_20.predict(X_test)
    
#Evaluation
print(classification_report(y_test,pred_20),confusion_matrix(y_test,pred_20))


knn_20=KNeighborsClassifier(n_neighbors=30)
knn_20.fit(X_train,y_train)
pred_20=knn_20.predict(X_test)
    
#Evaluation
print(classification_report(y_test,pred_20),confusion_matrix(y_test,pred_20))
    
    

