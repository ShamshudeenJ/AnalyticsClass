# -*- coding: utf-8 -*-
"""AnalyticsClass.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l2Z-BCt3ruu65HKUwIYocB9SeO_WR4go
"""

import pandas as pd

url = "https://raw.githubusercontent.com/ShamshudeenJ/AnalyticsClass/master/roomSensorData.csv"
df = pd.read_csv(url)

print(df.info())
print(df.describe())
print(df.head())
#############################################################
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig = plt.figure(figsize=(18,6))
x = df['Occupancy'].value_counts()
print(x)
plt.pie(x,labels=[0,1],shadow=True,autopct='%1.1f%%')
plt.title('Occupancy')
plt.show()
#############################################################
fig = plt.figure(figsize=(18,6))

ax1 = fig.add_subplot(1,3,1)
x = []
for i in df.Occupancy.unique():
   x.append(df[df['Occupancy'] == i]['Temperature'])
ax1.boxplot(x)
ax1.set_xticklabels([0,1])
ax1.set_title('Temperature')

ax2 = fig.add_subplot(1,3,2)
x = []
for i in df.Occupancy.unique():
   x.append(df[df['Occupancy'] == i]['Humidity'])
ax2.boxplot(x)
ax2.set_xticklabels([0,1])
ax2.set_title('Humidity')

ax3 = fig.add_subplot(1,3,3)
x = []
for i in df.Occupancy.unique():
   x.append(df[df['Occupancy'] == i]['CO2'])
ax3.boxplot(x)
ax3.set_xticklabels([0,1])
ax3.set_title('CO2')

plt.show()
#############################################################
fig = plt.figure(figsize=(13,6))
x=df['Temperature']
y=df['Humidity']
z=df['CO2']
ax1 = fig.add_subplot(131)
ax1.scatter(x,y,c=df['Occupancy']*5)
ax1.set_title('Temperature Vs Humidity')

ax2 = fig.add_subplot(132)
ax2.scatter(x,z,c=df['Occupancy']*5)
ax2.set_title('Temperature Vs CO2')

ax3 = fig.add_subplot(133)
ax3.scatter(z,y,c=df['Occupancy']*5)
ax3.set_title('CO2 Vs Humidity')

plt.show()
#############################################################
import seaborn as sns
#corrmat = df.corr()
corrmat = df[['Temperature','Humidity','CO2']].corr()
sns.heatmap(corrmat)
plt.show()
#############################################################
from pandas.plotting import parallel_coordinates
result = df.copy()
dfPlot = df.drop('Occupancy',axis=1)
fig = plt.figure(figsize=(13,6))

for feature_name in dfPlot.columns:
   max_value = dfPlot[feature_name].max()
   min_value = dfPlot[feature_name].min()
   result[feature_name] = (dfPlot[feature_name] - min_value) / (max_value - min_value)
parallel_coordinates(result,'Occupancy',color=('r','g'))
plt.show()
#############################################################
from sklearn.model_selection import train_test_split

Xml=df.drop(['Occupancy'],axis=1)
X=Xml.values
Y = df['Occupancy'].values

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=0)

print('Train: ',X_train.shape, y_train.shape)
print('Test: ',X_test.shape, y_test.shape)
#############################################################
from sklearn.tree import DecisionTreeClassifier

MLmodel = DecisionTreeClassifier(max_depth = 3)
MLmodel.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score

print(MLmodel.score(X_test,y_test)) 

CVscores = cross_val_score(MLmodel,X,Y,cv=5)
print(CVscores)
#############################################################
from sklearn.tree import export_graphviz
import os
import matplotlib.image as mpimg

dotfile = open("myModel.dot", 'w')
export_graphviz(MLmodel,out_file=dotfile,
	feature_names=Xml.columns,class_names=['0','1'])
dotfile.close()
os.system('dot myModel.dot -Tpng -o myModel.png')
img=mpimg.imread('myModel.png')
fig = plt.figure(figsize=(20,10))
imgplot = plt.imshow(img)
plt.show()
