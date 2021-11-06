### Dataset and Python code related to **Intro to Exploratory Data Analytics** on IoT data
<br>

* In statistics, exploratory data analysis is an approach of analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods.
* A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task!
<br>
<img src="/images/eda.png" alt="EDA" width="666"/>

[Source: EDA - wikipedia]

#### Python packages used
* Pandas
* Matplotlib
* Numpy
* Seaborn
* Sklearn
    * Simple and efficient tools for predictive data analysis
    * Built on NumPy, SciPy, and matplotlib
    * Supports
        * Classification
        * Regression
        * Clustering
        * Dimensionality reduction
        * Model selection
        * Preprocessing
#### Google Colab
URL : https://colab.research.google.com/
Allows us to write and execute Python in browser and Zero configuration required
* The Notebook cell can either code or text
* Write code in code cell
* To run, click the code cell and press Ctrl+Enter
* Output will be displayed immediate after code cell

### Exercise: Room Occupancy detection using CO2, Temperature and Humidity sensors!
*Procedure:*
1. Understand the sensor data file 
2. Plot various graphs for further insights
3. Split the data into Test and Train and Build Decision Tree (ML Classification) model using Train data
4. Check the accuracy of the model using Test data
5. Export the Trained model into simple nested If-Else structure

```python
import pandas as pd
url = "https://raw.githubusercontent.com/ShamshudeenJ/AnalyticsClass/master/roomSensorData.csv"
df = pd.read_csv(url)
print(df.info())
print(df.describe())
print(df.head())
```

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 4 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Temperature  1000 non-null   float64
 1   Humidity     1000 non-null   float64
 2   CO2          1000 non-null   float64
 3   Occupancy    1000 non-null   int64  
dtypes: float64(3), int64(1)
memory usage: 31.4 KB
None
       Temperature     Humidity          CO2    Occupancy
count  1000.000000  1000.000000  1000.000000  1000.000000
mean     21.432120    25.312760   716.005630     0.350000
std       1.037998     2.412848   291.629558     0.477208
min      20.200000    22.100000   427.500000     0.000000
25%      20.627500    23.075000   464.575000     0.000000
50%      20.900000    24.970000   581.175000     0.000000
75%      22.335000    26.962500   984.250000     1.000000
max      24.410000    31.470000  1402.250000     1.000000
   Temperature  Humidity     CO2  Occupancy
0        21.53     28.29  871.33          0
1        22.09     26.13  936.43          1
2        20.29     22.70  427.60          0
3        22.76     26.82  983.33          1
4        20.29     22.72  438.75          0


```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
fig = plt.figure(figsize=(18,6))
x = df['Occupancy'].value_counts()
print(x)
plt.pie(x,labels=[0,1],shadow=True,autopct='%1.1f%%')
plt.title('Occupancy')
plt.show()
```

<img src="/images/pie.png" alt="Pie" width="400"/>

```python
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
```

<img src="/images/box.png" alt="Box" width="400"/>

```python
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
```

<img src="/images/scatter.png" alt="Scatter" width="400"/>

```python
from sklearn.model_selection import train_test_split
Xml=df.drop(['Occupancy'],axis=1)
X=Xml.values
Y = df['Occupancy'].values
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=0)
print('Train: ',X_train.shape, y_train.shape)
print('Test: ',X_test.shape, y_test.shape)
```

> Train:  (750, 3) (750,)
> Test:  (250, 3) (250,)

```python
from sklearn.tree import DecisionTreeClassifier
MLmodel = DecisionTreeClassifier(max_depth = 2)
MLmodel.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score
print(MLmodel.score(X_test,y_test)) 
CVscores = cross_val_score(MLmodel,X,Y,cv=5)
print(CVscores)
```
> 0.86
> [0.865 0.9   0.855 0.915 0.83 ]

```python
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
```

<img src="/images/tree.png" alt="Tree" width="400"/>
