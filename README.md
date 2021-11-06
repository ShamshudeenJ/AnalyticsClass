### Dataset and Python code related to **Intro to Exploratory Data Analytics** on IoT data
<br>

* In statistics, exploratory data analysis is an approach of analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods.
* A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task!
<br>

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



