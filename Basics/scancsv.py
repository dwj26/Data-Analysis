
import csv as csv
import numpy as np

csv_file_object = csv.reader(open('C:/Users/Dan/Downloads/TB_burden_countries_2014-08-07(1).csv', 'rb'))       # Load in the csv file
header = csv_file_object.next()                             # Skip the fist line as it is a header
data=[]                                                     # Create a variable to hold the data

for row in csv_file_object:                 # Skip through each row in the csv file
    data.append(row)                        # adding each row to the data variable
data = np.array(data)                       # Then convert from a list to an array
print data
