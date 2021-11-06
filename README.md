### Dataset and Python code related to **Intro to Exploratory Data Analytics** on IoT data
<br>

* In statistics, exploratory data analysis is an approach of analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods.
* A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task!
<br>
<img src="/images/eda.png" alt="EDA" width="200"/>
![EDA](/images/eda.png "EDA")
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
> <class 'pandas.core.frame.DataFrame'>
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
![Pie](/images/pie.png "Pie")


