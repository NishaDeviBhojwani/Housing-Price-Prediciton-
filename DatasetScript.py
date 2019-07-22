# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:53:20 2019

@author: Nisha Bhojwani
"""

import pandas as pd
import requests
from lxml import html



#size Area in sqft 
def GetArea(tree):
    Area = tree.xpath('//li[@class="d-inline-block pointer-events-auto area"]/text()')
    print("Area : ",Area.__len__())
    return Area


def GetAreaCount(tree):
    Area = tree.xpath('//li[@class="d-inline-block pointer-events-auto area"]/span[@class="count"]/text()')
    print("Area : ",Area.__len__())
    return Area


#number of washrooms    
def GetBathRooms(tree):
    Bath = tree.xpath('//li[@class="d-inline-block"]/span[@class="count"]/text()')
    print("Bath : ",Bath.__len__())
    return Bath


#number of bedrooms    
def GetBedRooms(tree):
    Bed = tree.xpath('//li[@class="d-inline-block bed"]/span[@class="count"]/text()')
    print("Bed : ",Bed.__len__())
    return Bed

# Location of House
def GetLocation(tree):
    location = tree.xpath('//div[@class="location text-truncate"]/text()')
    print("Location : ",location.__len__())
    return location

def GetCurrency(tree):
    currency = tree.xpath('//span[@class="unit d-inline-block"]/text()')
    print("Currency : ",currency.__len__())
    return currency

# Price Of house
def GetPrice(tree):
    price = tree.xpath('//span[@class="price-range d-inline-block text-capitalize"]/text()')
    return price


def TreeXpath(url):
    page = requests.get(url)
    tree = html.fromstring(page.content)    
    return tree

def DataFrameReturn(url):
    tree = TreeXpath(url)
    price2 = GetPrice(tree)
    currency2 = GetCurrency(tree)
    location2 = GetLocation(tree)
    Bed2 = GetBedRooms(tree)
    Bath2 = GetBathRooms(tree)
    Area2 = GetArea(tree)
    AreaCount = GetAreaCount(tree)
    df2 = pd.DataFrame(list(zip(*[location2,Bed2,Bath2,AreaCount,Area2,currency2,price2]))).add_prefix('col')
    df2.rename(columns={'col0':'Location','col1':'BedRooms','col2':'WashRooms','col3':'AreaCount','col4':'Area','col5':'Currency','col6':'Price'}, 
                     inplace=True)
    return df2


df1 = DataFrameReturn("https://www.prop.pk/karachi/houses-for-sale-2/")
for i in range(2,701):
    url = "https://www.prop.pk/karachi/houses-for-sale-2/"+i.__str__()+"/"
    df = DataFrameReturn(url)
    df1 = df1.append(df)


df1.to_csv('DATASET_PROP.csv', index=False)   
print("Successful")
df1['Location'], levels = pd.factorize(df1['Location'])
df1.to_csv('new_data.csv')