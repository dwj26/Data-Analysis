from matplotlib import pyplot

df = pd.read_csv('C:/Users/Dan/Downloads/life_expectancies_usa.csv', header = 0)

x = df['Date']
y1 = df['MaleAge']
y2 = df['FemaleAge']
pyplot.plot(x,y1, 'o-', label = 'Male')
pyplot.plot(x,y2,'o-', label = 'Female')
pylab.legend(loc='upper left')
pyplot.xlim(1850,2000)

pyplot.xlabel('Year')
pyplot.ylabel('Life Expextancy in Years')
pyplot.show()
