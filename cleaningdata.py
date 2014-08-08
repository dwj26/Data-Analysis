# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 13:58:35 2014

@author: Dan
"""

import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('C:/Users/Dan/Downloads/train.csv', header=0)
#Create new gender column where male is mapped to 1 and female to 0
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#create copy of age column
df['AgeFill'] = df['Age']
#define median ages as an array
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
 
median_ages
#fill the gaps of NaN with the median ages for that class/gender
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]
#create some new columns
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass
#drop the columns that aren't numbers
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df = df.drop(['Age'], axis=1)
#drop the columns with no values
df = df.dropna()
#create a numpy array
train_data = df.values
print train_data

