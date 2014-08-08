# -*- coding: utf-8 -*-
"""
Created on Thu Aug 07 18:50:34 2014

@author: Dan
"""


#Plots of the mortality rate of TB in 3 countries from WHO data
import csv as csv
import numpy as np
import pandas as pd
import pylab as P
import pylab
import matplotlib.pyplot as plt
# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('C:/Users/Dan/Downloads/TB_burden_countries_2014-08-07(1).csv', header=0)

csv_file_object = csv.reader(open('C:/Users/Dan/Downloads/TB_burden_countries_2014-08-07(1).csv', 'rb'))       # Load in the csv file
header = csv_file_object.next()                             # Skip the fist line as it is a header
data=[]                                                     # Create a variable to hold the data

for row in csv_file_object:                 # Skip through each row in the csv file
    data.append(row)                        # adding each row to the data variable
data = np.array(data)                       # Then convert from a list to an array

x = df['year'].head(22)
y = df['e_inc_tbhiv_100k'].head(22)
#head takes the first 22 values
plt.xlabel('Year')
plt.ylabel('Deaths per 100000')
plt.title("Afghanistan TB Mortality")
#label axes
plt.plot(x,y)
P.show()
#plot and show graph
x  = df['year']
x = x[253:275]
y = df['e_inc_tbhiv_100k']
y = y[253:275]
plt.xlabel('Year')
plt.ylabel('Deaths per 100000')
plt.title("Australia TB Mortality")
plt.plot(x,y)
P.show()
x  = df['year']
x = x[4581:4603]
y = df['e_inc_tbhiv_100k']
y = y[4581:4603]
plt.ylim(0.1,0.7)
plt.xlabel('Year')
plt.ylabel('Deaths per 100000')
plt.title("UK TB Mortality")
plt.plot(x,y)
P.show()
