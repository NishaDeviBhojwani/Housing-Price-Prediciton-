# -*- coding: utf-8 -*-
"""
Created on Sun May  5 20:13:36 2019

@author: Nisha Bhojwani
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor



#data model randomForest
def RandomForestegression(df):
    labels = df["Price"]
    train1 = df.drop(['Price'],axis=1)
    x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =42)
    rf = RandomForestRegressor()
    rf.fit(x_train,y_train)
    print("----------------Data Modeling random forest  ---------- ")
    print ("ÄCcuracy Using Random Forest Regression: ",rf.score(x_test,y_test))


# data model gradient boost
def GradientBoost(df):
    labels = df["Price"]
    train1 = df.drop(['Price'],axis=1)
    x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =42)
    clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,learning_rate = 0.1, loss = 'ls')
    clf.fit(x_train, y_train)
    print("----------------Data Modeling Gradient Boost ---------- ")
    print("ÄCcuracy Using Gradient Boost: ",clf.score(x_test,y_test))


   #outliers remove 
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

#Area size Sq. ft
def CleanAreaCountSize(df):
    df['AreaCount'] = df['AreaCount'].str.replace(",","")
    df['AreaCount'] = pd.to_numeric(df['AreaCount'],errors='coerce')
    return df

# currency data transform
def CleanCurrency(df):
    df['Currency'] = df['Currency'].str.upper()
    df['Currency'] = df['Currency'].str.replace("PKR","1")
    df['Currency'] = pd.to_numeric(df['Currency'],errors='coerce')
    return df

#null values remove
def RemoveNullValue(df):
    df = df.dropna()
    return df

# drop dulicates
def DropDuplicates(df):
    df.drop_duplicates(keep = False, inplace = True)
    df = df.sort_values(by='Price')
    df = df.reset_index(drop=True)
    return df


# replace Crore & lac with zeros & to numeric the price
# remove white space
def CleaningPriceData(df):
    df['Price'] = df['Price'].replace('\w*Crore','10000000',regex=True) 
    df['Price'] = df['Price'].replace('[lL]akh','100000',regex = True) 
    price = df['Price'].str.split()
    dfPrice = price.str.get(0)
    dfLabel = price.str.get(1)
    dfPrice = dfPrice.str.replace(" ","")
    dfLabel = dfLabel.str.replace(" ","")
    dfPrice = pd.to_numeric(dfPrice,errors='coerce')
    dfLabel = pd.to_numeric(dfLabel,errors='coerce')
    df['Price'] = dfPrice*dfLabel
    return df


""" -----------Data Visualization-----------------------------------------------"""

def DataVisualization(df):
    sns.kdeplot(df);
    plt.plot(df)
    plt.legend('ABCDEF', ncol=2, loc='upper left');
    df.boxplot()
    plt.hist(x = df['Price'])
    plt.xlabel('Price Histogram')
    plt.plot()
    sns.pairplot(df, size=2.5)
    plt.tight_layout()
    sns.set(font_scale=1.5)
    


# Data Cleaning 
df = pd.read_csv("new_data.csv")
print ("Number of Rows of each column before removing duplicates:");
print (df.shape);
print(df.shape)
df = df.drop(['Currency', 'Area','Unnamed: 0'],axis=1)
df = CleaningPriceData(df)
print ("Number of Rows of each column after cleaning price data:");
print (df.shape);
df = CleanAreaCountSize(df)
df = remove_outlier(df,'Price')
print ("Number of Rows of each column after cleaning area size:");    
print (df.shape);
df = DropDuplicates(df)
print ("Number of Rows of each column after removing duplicates:");
print (df.shape);
df = RemoveNullValue(df)
print ("Number of Rows of each column after removing null values:");
print (df.shape);
print("Dataset Head:")
print(df.head())
print(df.shape)

# Data Modeling 
GradientBoost(df)
RandomForestegression(df)






