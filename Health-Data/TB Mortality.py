# -*- coding: utf-8 -*-
"""
Created on Thu Aug 07 18:50:34 2014

@author: Dan
"""



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

#plot for Afghan, which is first 22 data points
x = df['year'].head(22)
y = df['e_inc_tbhiv_100k'].head(22)
pylab.plot(x,y,label = 'Afghanistan')
#plot for Australia between shown data points
x  = df['year']
x = x[253:275]
y = df['e_inc_tbhiv_100k']
y = y[253:275]
plt.plot(x,y, label = 'Australia')
#plot for UK between shown data points
x  = df['year']
x = x[4581:4603]
y = df['e_inc_tbhiv_100k']
y = y[4581:4603]
#make graph pretty
plt.ylim(0.1,1.0)
plt.xlim(1990,2011)
plt.xlabel('Year')
plt.ylabel('Deaths per 100000')
plt.title('TB Mortality')
plt.plot(x,y, label = 'UK')
pylab.legend(loc='upper left')  #locate legend in upper left out of way of graph
P.show()
