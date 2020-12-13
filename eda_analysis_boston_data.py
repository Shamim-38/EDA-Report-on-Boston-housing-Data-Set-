import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import plotly.express as px
import seaborn as sns
import stemgraphic
from sklearn import datasets
boston= datasets.load_boston ()
####Now transform the data as a pandas’s DATAFRAME
import pandas as pd
df = pd.DataFrame(boston.data ,columns = boston.feature_names)
df['price']=boston.target


# List of data series
datarowsSeries =   [pd.Series([0.069,10,2.3,0,0.53,6.5,65.2,4.01,1,290,15,395,4.9,24.0],
index=df.columns ), pd.Series([0.69+12, 10+12, 2.3+.12, 0, 0.5+.12, 6.5+.12, 65.2+.12, 4.1+.12, 1, 290+12, 15+12, 395+12, 4.9+.12, 24.3+12],
index=df.columns ), pd.Series([0.68+12, 11+12, 2.4+.12, 0, 0.6+.12, 6.6+.12, 65.1+.12, 4.0+.12, 1, 291+12, 13+12, 390+12, 4.2+.12, 24.2+12],
index=df.columns ), pd.Series([0.67+12, 12+12, 2.5+.12, 0, 0.4+.12, 6.5+.12, 65.3+.12, 4.2+.12, 1, 292+12, 14+12, 392+12, 4.3+.12, 24.1+12],
index=df.columns ), pd.Series([0.66+12, 13+12, 2.4+.12, 0, 0.7+.12, 6.5+.12, 65.4+.12, 4.1+.12, 1, 293+12, 16+12, 391+12, 4.4+.12, 24.2+12],
index=df.columns ) ]

# Pass the list of data series to the append() to add multiple rows
new_data = df.append(datarowsSeries , ignore_index=True)
df = new_data

#-----------Data Dicription 29-51 line-----------------------------------
#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~To print the dataset description~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(boston['DESCR'])

#print(boston.keys())

#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~To print feature names of dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print(boston.feature_names)



#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~To print data shape of dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#print("Type of boston dataset:", boston.data.shape)
"""
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~Print datatypes of variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(df.dtypes)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~To print the missing data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(df.isnull().sum())



print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~To print the datatypes of variables and the missing data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(df.info())


#-----------for Table 2  code is 55-282 line-----------------------------------
df.boxplot()
plt.show()

plt.boxplot(df.CRIM)
plt.title('Boxplot for per capita crime rate by town', color='RED')
plt.xlabel('CRIM', color='RED')
plt.show()

plt.boxplot(df.ZN)
plt.title('Boxplot for the proportion of residential land zoned', color='RED')
plt.xlabel('ZN', color='RED')
plt.show()

plt.boxplot(df.INDUS)
plt.title('Boxplot for the proportion of non-retail business acres per town', color='RED')
plt.xlabel('INDUS', color='RED')
plt.show()

plt.boxplot(df.CHAS)
plt.title('Boxplot for the Charles River dummy variable', color='RED')
plt.xlabel('CHAS', color='RED')
plt.show()


plt.boxplot(df.NOX)
plt.title('Boxplot for the nitric oxides concentration ', color='RED')
plt.xlabel('NOX', color='RED')
plt.show()

plt.boxplot(df.RM)
plt.title('Boxplot for the average number of rooms per dwelling', color='RED')
plt.xlabel('RM', color='RED')
plt.show()

plt.boxplot(df.AGE)
plt.title('Boxplot for the proportion of owner-occupied units built prior to 1940', color='RED')
plt.xlabel('AGE', color='RED')
plt.show()

plt.boxplot(df.DIS)
plt.title('Boxplot for the weighted distances to five employment centres', color='RED')
plt.xlabel('DIS', color='RED')
plt.show()

plt.boxplot(df.RAD)
plt.title('Boxplot for the index of accessibility to radial highways', color='RED')
plt.xlabel('RAD', color='RED')
plt.show()

plt.boxplot(df.TAX)
plt.title('Boxplot for the full-value property-tax rate per $10,000', color='RED')
plt.xlabel('TAX', color='RED')
plt.show()

plt.boxplot(df.PTRATIO)
plt.title('Boxplot for the pupil-teacher ratio by town', color='RED')
plt.xlabel('PTRATIO', color='RED')
plt.show()

plt.boxplot(df.B)
plt.title('Boxplot for the 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town', color='RED')
plt.xlabel('B', color='RED')
plt.show()

plt.boxplot(df.LSTAT)
plt.title('Boxplot for the persentage lower status of the population', color='RED')
plt.xlabel('LSTAT', color='RED')
plt.show()

plt.boxplot(df.price)
plt.title('Median value of owner-occupied homes in USD 1000’s', color='RED')
plt.xlabel('price', color='RED')
plt.show()

sns.boxplot(y="LSTAT", data=df)
plt.show()



plt.hist(df.CRIM, bins=5)
plt.xlabel('Histrogram for per capita crime rate by town')
plt.show()

plt.hist(df.ZN, bins=5)
plt.xlabel('Histrogram for the proportion of residential land zoned for lots over 25,000 sq.ft.')
plt.show()

plt.hist(df.INDUS, bins=5)
plt.xlabel('Histrogram for the proportion of non-retail business acres per town')
plt.show()

plt.hist(df.CHAS, bins=5)
plt.xlabel('Histrogram for the Charles River dummy variable')
plt.show()

plt.hist(df.NOX, bins=5)
plt.xlabel('Histrogram for the nitric oxides concentration')
plt.show()

plt.hist(df.RM, bins=5)
plt.xlabel('Histrogram for the average number of rooms per dwelling')
plt.show()

plt.hist(df.AGE, bins=5)
plt.xlabel('Histrogram for the proportion of owner-occupied units built prior to 1940')
plt.show()

plt.hist(df.DIS, bins=5)
plt.xlabel('Histrogram for the weighted distances to five employment centres')
plt.show()

plt.hist(df.RAD, bins=5)
plt.xlabel('Histrogram for the index of accessibility to radial highways')
plt.show()

plt.hist(df.TAX, bins=5)
plt.xlabel('Histrogram for the full-value property-tax rate per $10,000')
plt.show()

plt.hist(df.PTRATIO, bins=5)
plt.xlabel('Histrogram for the pupil-teacher ratio by town')
plt.show()

plt.hist(df.B, bins=5)
plt.xlabel('Histrogram for the 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town')
plt.show()

plt.hist(df.LSTAT, bins=5)
plt.xlabel('Histrogram for the persentage lower status of the population')
plt.show()

plt.hist(df.price, bins=5)
plt.xlabel('Histrogram for the Median value of owner-occupied homes in $1000s')
plt.show()

"""
"""
fig, ax = stemgraphic.stem_graphic(df.CRIM, scale=10)
ax.set_title("Stem and Left Plot for CRIM")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.ZN, scale=10)
ax.set_title("Stem and Left Plot for ZN")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.INDUS, scale=10)
ax.set_title("Stem and Left Plot for INDUS")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.CHAS, scale=10)
ax.set_title("Stem and Left Plot for CHAS")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.NOX)
ax.set_title("Stem and Left Plot for NOX")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.RM)
ax.set_title("Stem and Left Plot for RM")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.AGE, scale=10)
ax.set_title("Stem and Left Plot for AGE")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.DIS, scale=10)
ax.set_title("Stem and Left Plot for DIS")
plt.show()



fig, ax = stemgraphic.stem_graphic(df.RAD, scale=10)
ax.set_title("Stem and Left Plot for RAD")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.TAX, scale=10)
ax.set_title("Stem and Left Plot for TAX")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.PTRATIO, scale=10)
ax.set_title("Stem and Left Plot for PTRATIO")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.B, scale=10)
ax.set_title("Stem and Left Plot for B")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.LSTAT, scale=10)
ax.set_title("Stem and Left Plot for LSTAT")
plt.show()

fig, ax = stemgraphic.stem_graphic(df.price, scale=10)
ax.set_title("Stem and Left Plot for PRICE")
plt.show()


fig = px.pie(df, values='pop', names='country',
             title='Population of American continent',
             hover_data=['lifeExp'], labels={'lifeExp':'life expectancy'})
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


#for pie chart
s = df.groupby('CHAS').size()
sns.set()
s.plot(kind='pie', title='Charles River dummy variable ', figsize=[8,8],
          autopct=lambda p: '{:.2f}%({:.0f})'.format(p,(p/100)*s.sum()))
plt.show()

#for bar chart
s = df.groupby('CHAS').size()
sns.set()
s.plot(kind='bar', title='Charles River dummy variable', stacked=True)
plt.show()

#for pie chart
s = df.groupby('RAD').size()
sns.set()
s.plot(kind='pie', title='Index of accessibility to radial highways', figsize=[8,8],
          autopct=lambda p: '{:.2f}%({:.0f})'.format(p,(p/100)*s.sum()))
plt.show()

#for bar chart
s = df.groupby('RAD').size()
sns.set()
s.plot(kind='bar', title='Index of accessibility to radial highways', stacked=True)
plt.show()
"""



#for bivariante 
"""
#Scatter Diagram
plt.scatter(df['CRIM'],df['price'], c = 'b')
plt.xlabel('CRIM')
plt.ylabel('price')
plt.title('CRIM vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['ZN'],df['price'], c = 'b')
plt.xlabel('ZN')
plt.ylabel('price')
plt.title('ZN vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['INDUS'],df['price'], c = 'b')
plt.xlabel('INDUS')
plt.ylabel('price')
plt.title('INDUS vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['NOX'],df['price'], c = 'b')
plt.xlabel('NOX')
plt.ylabel('price')
plt.title('NOX vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['RM'],df['price'], c = 'b')
plt.xlabel('RM')
plt.ylabel('price')
plt.title('RM vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['AGE'],df['price'], c = 'b')
plt.xlabel('AGE')
plt.ylabel('price')
plt.title('AGE vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['DIS'],df['price'], c = 'b')
plt.xlabel('DIS')
plt.ylabel('price')
plt.title('DIS vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['TAX'],df['price'], c = 'b')
plt.xlabel('TAX')
plt.ylabel('price')
plt.title('TAX vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['PTRATIO'],df['price'], c = 'b')
plt.xlabel('PTRATIO')
plt.ylabel('price')
plt.title('PTRATIO vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['B'],df['price'], c = 'b')
plt.xlabel('B')
plt.ylabel('price')
plt.title('B vs Price')
plt.show()

#Scatter Diagram
plt.scatter(df['LSTAT'],df['price'], c = 'b')
plt.xlabel('LSTAT')
plt.ylabel('price')
plt.title('LSTAT vs Price')
plt.show()

#Box plot for bivariate 
sns.boxplot(x='CHAS',y='price', data=df)
plt.show()

sns.boxplot(x='RAD',y='price', data=df)
plt.show()

#Figure 17
#Pairplots
sns.set_style('ticks');
sns.pairplot(df, hue = 'price')
plt.show()

sns.lmplot(x = 'CRIM', y = 'price', data = df)
plt.show()
"""

#Summary Measures
#print(df.describe())
#print(df.mode())
#print(df.median())
#print(df.corr())
print(df.cov())
















