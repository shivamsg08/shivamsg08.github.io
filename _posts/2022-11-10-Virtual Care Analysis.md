---
layout: post
cover-img: /DataFiles/Posts/VirtualHealthcare/1_6069795.jpg
title: Virtual Care Adoption in New Brunswick
subtitle: Descriptive Analysis and Post Covid Prediction
gh-repo: shivamsg08/Virtual-Healthcare
gh-badge: [star, fork, follow]
comments: true

thumbnail-img: /DataFiles/Posts/VirtualHealthcare/virtual-care-735x400-1.jpg
# share-img: /assets/img/path.jpg
tags: [Python, Univariate Forecasting, Linear Regression]
---

The New Brunswick Department of Health, Health Analytics branch, fosters the strategic use of information and analytics to inform decision making as it relates to the Department’s mandate of planning, funding and monitoring a high quality & sustainable health system for the citizens of New Brunswick.

![image.png](attachment:image.png)

# Virtual Care Utilization

# About the Data

The world of digital health is continuously expanding and has enormous potential to provide adequate, cost-efficient, safe andscalable eHealth interventions to improve health and health care.**It is time to reimagine the traditional, in-person approach to care.**

Digital health solutions can change the way New Brunswickers receive services and how citizens and providers engage with the health-care system. These services or interventions should bedesigned around the patient’s needs and pertinent informationshould be shared in a proactive and efficient way through smarter use of data, devices, communication platforms and people.

This data set contains an aggregation of patient visits that have occurred during the pandemic. As part of the pandemic response, which includes the new program to support virtual care delivery for the public. The data elements within the data set will allow you to trend the progress of virtual visits, and to describe the characteristics of patients and physicians who have embraced virtual care.The data set contains 113,202 rows including title headers.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")
```

    C:\Users\Shivam Goyal\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:7: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
      from pandas import (to_datetime, Int64Index, DatetimeIndex, Period,
    C:\Users\Shivam Goyal\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:7: FutureWarning: pandas.Float64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
      from pandas import (to_datetime, Int64Index, DatetimeIndex, Period,
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YYYYMM</th>
      <th>Year</th>
      <th>Month</th>
      <th>Age Group Code (Patient)</th>
      <th>Patient Age Group</th>
      <th>Age Group (Patient)</th>
      <th>Gender (Patient)</th>
      <th>Patient Gender</th>
      <th>Health Zone (Patient)</th>
      <th>Patient Health Zone</th>
      <th>Gender (Physician)</th>
      <th>Physician Gender</th>
      <th>Health Zone (Physician)</th>
      <th>Physician Health Zone</th>
      <th>Visit Type</th>
      <th>Number of Visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>202112</td>
      <td>2021</td>
      <td>12</td>
      <td>4</td>
      <td>0-20 Years</td>
      <td>15 - 19 Years</td>
      <td>1</td>
      <td>Male</td>
      <td>3</td>
      <td>Fredricton Area</td>
      <td>1</td>
      <td>Male</td>
      <td>7</td>
      <td>Miramichi Area</td>
      <td>Office</td>
      <td>9103</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202103</td>
      <td>2021</td>
      <td>3</td>
      <td>4</td>
      <td>0-20 Years</td>
      <td>15 - 19 Years</td>
      <td>1</td>
      <td>Male</td>
      <td>3</td>
      <td>Fredricton Area</td>
      <td>2</td>
      <td>Female</td>
      <td>1</td>
      <td>Moncton Area</td>
      <td>Office</td>
      <td>4591</td>
    </tr>
    <tr>
      <th>2</th>
      <td>202111</td>
      <td>2021</td>
      <td>11</td>
      <td>4</td>
      <td>0-20 Years</td>
      <td>15 - 19 Years</td>
      <td>1</td>
      <td>Male</td>
      <td>3</td>
      <td>Fredricton Area</td>
      <td>2</td>
      <td>Female</td>
      <td>1</td>
      <td>Moncton Area</td>
      <td>Office</td>
      <td>4448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>202104</td>
      <td>2021</td>
      <td>4</td>
      <td>4</td>
      <td>0-20 Years</td>
      <td>15 - 19 Years</td>
      <td>1</td>
      <td>Male</td>
      <td>3</td>
      <td>Fredricton Area</td>
      <td>2</td>
      <td>Female</td>
      <td>1</td>
      <td>Moncton Area</td>
      <td>Office</td>
      <td>4435</td>
    </tr>
    <tr>
      <th>4</th>
      <td>202102</td>
      <td>2021</td>
      <td>2</td>
      <td>4</td>
      <td>0-20 Years</td>
      <td>15 - 19 Years</td>
      <td>1</td>
      <td>Male</td>
      <td>3</td>
      <td>Fredricton Area</td>
      <td>2</td>
      <td>Female</td>
      <td>1</td>
      <td>Moncton Area</td>
      <td>Office</td>
      <td>4370</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns, len(df.columns)
```




    (Index(['YYYYMM', 'Year', 'Month', 'Age Group Code (Patient)',
            'Patient Age Group', 'Age Group (Patient)', 'Gender (Patient)',
            'Patient Gender', 'Health Zone (Patient)', 'Patient Health Zone',
            'Gender (Physician)', 'Physician Gender', 'Health Zone (Physician)',
            'Physician Health Zone', 'Visit Type', 'Number of Visits'],
           dtype='object'),
     16)




```python
(df.groupby(['Year','Month'])['Number of Visits'].sum().reset_index()).groupby(['Year'])['Number of Visits'].sum().reset_index().mean()
```




    Year                   2020.50
    Number of Visits    2436942.75
    dtype: float64




```python
(df.groupby(['Year','Patient Gender'])['Number of Visits'].sum().reset_index()).groupby(['Patient Gender'])['Number of Visits'].mean().reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patient Gender</th>
      <th>Number of Visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>1378825.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>1058107.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not Specified</td>
      <td>10.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(['Year'])['Physician Gender'].count().reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Physician Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>22627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>36737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021</td>
      <td>43025</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022</td>
      <td>10812</td>
    </tr>
  </tbody>
</table>
</div>




```python
(df.groupby(['Year','Month'])['Number of Visits'].sum().reset_index()).groupby(['Year'])['Number of Visits'].mean().reset_index().to_csv("Monthly Average Visits.csv")
```


```python
(df.groupby(['Year','Patient Health Zone'])['Number of Visits'].sum().reset_index()).groupby(['Patient Health Zone'])['Number of Visits'].mean().reset_index()

# .to_csv("Monthly Average Visits.csv")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patient Health Zone</th>
      <th>Number of Visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bathurst Area</td>
      <td>299642.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Campbellton Area</td>
      <td>69923.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Edmunston Area</td>
      <td>115816.75</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fredricton Area</td>
      <td>566323.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Miramichi Area</td>
      <td>133572.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Moncton Area</td>
      <td>627299.75</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Saint John Area</td>
      <td>615620.50</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Unknown</td>
      <td>8744.25</td>
    </tr>
  </tbody>
</table>
</div>

![image](https://github.com/shivamsg08/Virtual-Healthcare/assets/8438005/eeaf1da5-08a0-446b-951c-31fa838cdabe)


```python
# Visits Across Time
pd.pivot_table(df, values='Number of Visits', index=['Year', 'Month'],
                    columns=['Visit Type'], aggfunc=np.sum).reset_index().to_csv("Visitsacrosstime.csv",index = False)


```
![image](https://github.com/shivamsg08/Virtual-Healthcare/assets/8438005/184ba282-c145-4721-8a1d-69ee7a247e46)

```python
# Visits Across Patient Health Zone
pd.pivot_table(df[(df['Year']==2020) | (df['Year']==2021)], 
               values='Number of Visits', index=['Patient Health Zone'],
               columns=['Visit Type'], 
               aggfunc=np.sum).reset_index().to_csv("VisitsPatientHealthZone.csv",index = False)


```
![image](https://github.com/shivamsg08/Virtual-Healthcare/assets/8438005/84b59c94-8748-40f8-b605-0e45b3451cb0)


```python
# Visits Across Physician Health Zone
pd.pivot_table(df[(df['Year']==2020) | (df['Year']==2021)], 
               values='Number of Visits', index=['Physician Health Zone'],
               columns=['Visit Type'], 
               aggfunc=np.sum).reset_index().to_csv("VisitsPhysicianHealthZone.csv",index = False)


```
![image](https://github.com/shivamsg08/Virtual-Healthcare/assets/8438005/471ef7d9-3366-4122-a1d6-a111b7cda668)

```python
# Visits Across Patient Gender
pd.pivot_table(df[(df['Year']==2020) | (df['Year']==2021)], 
               values='Number of Visits', index=['Patient Gender'],
               columns=['Visit Type'], 
               aggfunc=np.sum).reset_index().to_csv("VisitsPatientGender.csv",index = False)

```
![image](https://github.com/shivamsg08/Virtual-Healthcare/assets/8438005/2f453173-8c44-4ace-8e47-a4e300a1d8ab)

```python
# Visits Across Physician Gender
pd.pivot_table(df[(df['Year']==2020) | (df['Year']==2021)], 
               values='Number of Visits', index=['Physician Gender'],
               columns=['Visit Type'], 
               aggfunc=np.sum).reset_index().to_csv("VisitsPhysicianGender.csv",index = False)

```

![image](https://github.com/shivamsg08/Virtual-Healthcare/assets/8438005/d6a01966-7c2c-40c5-b22d-c846bc85b5fc)

```python
# Visits Across Patient Age Group
pd.pivot_table(df[(df['Year']==2020) | (df['Year']==2021)], 
               values='Number of Visits', index=['Age Group (Patient)'],
               columns=['Visit Type'], 
               aggfunc=np.sum).reset_index().to_csv("VisitsPatientAgeGroup.csv",index = False)

```
![image](https://github.com/shivamsg08/Virtual-Healthcare/assets/8438005/cbb15e89-f57a-4676-8d05-9686cd7cb7d9)


```python
# Visits Across Patient and Physician Health Zone Combined
pd.pivot_table(df[(df['Year']==2020) | (df['Year']==2021)], 
               values='Number of Visits', index=['Patient Health Zone','Physician Health Zone'],
               columns=['Visit Type'], 
               aggfunc=np.sum).reset_index().to_csv("VisitsPatientPhysicianHealthZone.csv",index = False)


```


```python
# Visits Across Patient Gender & Health Zone Combined
pd.pivot_table(df[(df['Year']==2020) | (df['Year']==2021)], 
               values='Number of Visits', index=['Patient Health Zone','Patient Gender'],
               columns=['Visit Type'], 
               aggfunc=np.sum).reset_index().to_csv("patient healthzone gender.csv",index=False)
```


```python
# Visits Across Physician Gender & Health Zone Combined
pd.pivot_table(df[(df['Year']==2020) | (df['Year']==2021)], 
               values='Number of Visits', index=['Physician Health Zone','Physician Gender'],
               columns=['Visit Type'], 
               aggfunc=np.sum).reset_index().to_csv("Physician healthzone gender.csv",index=False)
```

# Forecast

### Moving Average


```python
# Visits Across Patient and Time
forecast_data = pd.pivot_table(df[~(df['Year']==2022)], 
               values='Number of Visits', index=['YYYYMM'],
               columns=['Visit Type'], 
               aggfunc=np.sum).reset_index()
```


```python
forecast_data = forecast_data.loc[:,['YYYYMM','Virtual Care']]
forecast_data
train = forecast_data.iloc[13:,]
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Visit Type</th>
      <th>YYYYMM</th>
      <th>Virtual Care</th>
      <th>MA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>202005</td>
      <td>73908.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>202006</td>
      <td>170473.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>202007</td>
      <td>145645.0</td>
      <td>130008.666667</td>
    </tr>
    <tr>
      <th>16</th>
      <td>202008</td>
      <td>125760.0</td>
      <td>147292.666667</td>
    </tr>
    <tr>
      <th>17</th>
      <td>202009</td>
      <td>144209.0</td>
      <td>138538.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>202010</td>
      <td>137580.0</td>
      <td>135849.666667</td>
    </tr>
    <tr>
      <th>19</th>
      <td>202011</td>
      <td>140555.0</td>
      <td>140781.333333</td>
    </tr>
    <tr>
      <th>20</th>
      <td>202012</td>
      <td>133632.0</td>
      <td>137255.666667</td>
    </tr>
    <tr>
      <th>21</th>
      <td>202101</td>
      <td>169229.0</td>
      <td>147805.333333</td>
    </tr>
    <tr>
      <th>22</th>
      <td>202102</td>
      <td>154023.0</td>
      <td>152294.666667</td>
    </tr>
    <tr>
      <th>23</th>
      <td>202103</td>
      <td>165160.0</td>
      <td>162804.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>202104</td>
      <td>146006.0</td>
      <td>155063.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>202105</td>
      <td>140076.0</td>
      <td>150414.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>202106</td>
      <td>144661.0</td>
      <td>143581.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>202107</td>
      <td>108489.0</td>
      <td>131075.333333</td>
    </tr>
    <tr>
      <th>28</th>
      <td>202108</td>
      <td>105513.0</td>
      <td>119554.333333</td>
    </tr>
    <tr>
      <th>29</th>
      <td>202109</td>
      <td>124829.0</td>
      <td>112943.666667</td>
    </tr>
    <tr>
      <th>30</th>
      <td>202110</td>
      <td>122944.0</td>
      <td>117762.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>202111</td>
      <td>128506.0</td>
      <td>125426.333333</td>
    </tr>
    <tr>
      <th>32</th>
      <td>202112</td>
      <td>115565.0</td>
      <td>122338.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
train['MA'] = train['Virtual Care'].rolling(window=2).mean()
```


```python
rms = sqrt(mean_squared_error(train.loc[30:,'Virtual Care'],train.loc[30:,'MA']))
print(rms)
```

    4102.423572312672
    


```python
test = train.loc[30:,]
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Visit Type</th>
      <th>YYYYMM</th>
      <th>Virtual Care</th>
      <th>MA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>202110</td>
      <td>122944.0</td>
      <td>123886.5</td>
    </tr>
    <tr>
      <th>31</th>
      <td>202111</td>
      <td>128506.0</td>
      <td>125725.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>202112</td>
      <td>115565.0</td>
      <td>122035.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 3 --5234.994913153281
# 2 --4102.423572312672
```


```python
fit2.forecast(7)
```




    array([119851.48, 119851.48, 119851.48, 119851.48, 119851.48, 119851.48,
           119851.48])



### Simple Exponential Smoothening


```python
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

fit2 = SimpleExpSmoothing(np.asarray(train.loc[30:,'Virtual Care'])).fit(smoothing_level=0.6,optimized=False)
test['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['Virtual Care'], label='Train')
plt.plot(test['Virtual Care'], label='Test')
plt.plot(test['SES'], label='SES')
plt.legend(loc='best')
plt.show()
```


    
![png](/DataFiles/Posts/VirtualHealthcare/output_33_0.png)
    


### Holt's Winters


```python
# import statsmodels.api as sm
# sm.tsa.seasonal_decompose(train['Virtual Care']).plot()
# result = sm.tsa.stattools.adfuller(train['Virtual Care'])
# plt.show()
```


```python
sns.barplot(x = 'Year', y = 'Number of Visits', hue = 'Visit Type', data = df)
#print(df.groupby(['Year', 'Visit Type']).mean()['Number of Visits'])
plt.show()

plt.figure(figsize=(25,5))
sns.barplot(x = 'Age Group (Patient)', y = 'Number of Visits', hue = 'Visit Type', data = df)
plt.show()

sns.barplot(x = 'Patient Gender', y = 'Number of Visits', hue = 'Visit Type', data = df)
plt.show()

sns.barplot(x = 'Physician Gender', y = 'Number of Visits', hue = 'Visit Type', data = df)
plt.show()

plt.figure(figsize=(25,5))
sns.barplot(x = 'Patient Health Zone', y = 'Number of Visits', hue = 'Visit Type', data = df)
plt.show()

plt.figure(figsize=(25,5))
sns.barplot(x = 'Physician Health Zone', y = 'Number of Visits', hue = 'Visit Type', data = df)
plt.show()


```


    
![png](/DataFiles/Posts/VirtualHealthcare/output_36_0.png)
    



    
![png](/DataFiles/Posts/VirtualHealthcare/output_36_1.png)
    



    
![png](/DataFiles/Posts/VirtualHealthcare/output_36_2.png)
    



    
![png](/DataFiles/Posts/VirtualHealthcare/output_36_3.png)
    



    
![png](/DataFiles/Posts/VirtualHealthcare/output_36_4.png)
    



    
![png](/DataFiles/Posts/VirtualHealthcare/output_36_5.png)
    


## Linear Regression Model


```python
dummies=pd.get_dummies(df,drop_first=True)
df=pd.concat([df,dummies.iloc[:,16:46]],axis=1)
#list(dummies.columns)

df_reg=pd.concat([df.iloc[:,0],df.iloc[:,16:46],df.iloc[:,15]],axis=1)
#df_reg=pd.concat([df.iloc[:,0],df.iloc[:,3],df.iloc[:,28:45],df.iloc[:,15]],axis=1)

x=df_reg.loc[:,df_reg.columns!="Number of Visits"]
y=df_reg["Number of Visits"]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.3)
```


```python
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

#print('Intercept: \n', regr.intercept_)
#print('Coefficients: \n', regr.coef_)

x = sm.add_constant(x_train) # adding a constant
 
model = sm.OLS(y_train, x_train).fit()
predictions = model.predict(x_test) 
 
print_model = model.summary()
print(print_model)

regr.score(x_test,y_test)

```

                                     OLS Regression Results                                
    =======================================================================================
    Dep. Variable:       Number of Visits   R-squared (uncentered):                   0.218
    Model:                            OLS   Adj. R-squared (uncentered):              0.217
    Method:                 Least Squares   F-statistic:                              710.9
    Date:                Fri, 04 Nov 2022   Prob (F-statistic):                        0.00
    Time:                        01:20:15   Log-Likelihood:                     -5.3945e+05
    No. Observations:               79240   AIC:                                  1.079e+06
    Df Residuals:                   79209   BIC:                                  1.079e+06
    Df Model:                          31                                                  
    Covariance Type:            nonrobust                                                  
    ==========================================================================================================
                                                 coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------------------------
    Year                                       0.0361      0.002     22.958      0.000       0.033       0.039
    Age Group (Patient)_35 - 39 Years         10.6933      3.476      3.076      0.002       3.881      17.506
    Age Group (Patient)_40 - 44 Years         15.3216      3.490      4.391      0.000       8.482      22.161
    Age Group (Patient)_45 - 49 Years         22.3762      3.482      6.426      0.000      15.552      29.201
    Age Group (Patient)_50 - 54 Years         30.1556      3.442      8.761      0.000      23.410      36.902
    Age Group (Patient)_55 - 59 Years         48.4215      3.391     14.280      0.000      41.776      55.068
    Age Group (Patient)_60 - 64 Years         57.7919      3.349     17.254      0.000      51.227      64.357
    Age Group (Patient)_65 - 69 Years         95.5130      3.347     28.534      0.000      88.952     102.074
    Age Group (Patient)_70 - 74 Years        134.8046      3.422     39.390      0.000     128.097     141.512
    Age Group (Patient)_75 - 79 Years        107.7460      3.646     29.554      0.000     100.600     114.892
    Age Group (Patient)_80 - 84 Years         75.6499      3.932     19.237      0.000      67.942      83.358
    Age Group (Patient)_85 - 89 Years         31.6560      4.133      7.659      0.000      23.555      39.757
    Age Group (Patient)_90+ Years              5.8787      4.565      1.288      0.198      -3.068      14.826
    Patient Gender_Male                      -19.8723      1.560    -12.738      0.000     -22.930     -16.815
    Patient Gender_Not Specified            -129.7541     42.995     -3.018      0.003    -214.024     -45.484
    Patient Health Zone_Campbellton Area     -58.0391      3.421    -16.965      0.000     -64.745     -51.334
    Patient Health Zone_Edmunston Area       -39.6609      3.556    -11.155      0.000     -46.630     -32.692
    Patient Health Zone_Fredricton Area       27.3165      3.022      9.040      0.000      21.394      33.239
    Patient Health Zone_Miramichi Area       -48.5354      3.151    -15.403      0.000     -54.711     -42.359
    Patient Health Zone_Moncton Area          39.6895      2.942     13.491      0.000      33.923      45.456
    Patient Health Zone_Saint John Area       60.7020      3.258     18.632      0.000      54.316      67.088
    Patient Health Zone_Unknown             -102.6974      3.586    -28.637      0.000    -109.726     -95.669
    Physician Gender_Male                     10.5345      1.566      6.727      0.000       7.465      13.604
    Physician Health Zone_Campbellton Area   -32.0513      3.597     -8.911      0.000     -39.101     -25.002
    Physician Health Zone_Edmunston Area     -28.1643      3.739     -7.533      0.000     -35.492     -20.837
    Physician Health Zone_Fredricton Area      9.6762      2.993      3.233      0.001       3.810      15.542
    Physician Health Zone_Miramichi Area     -54.1623      3.079    -17.592      0.000     -60.197     -48.128
    Physician Health Zone_Moncton Area        29.3353      2.826     10.382      0.000      23.797      34.874
    Physician Health Zone_Saint John Area     34.7255      3.088     11.247      0.000      28.674      40.777
    Physician Health Zone_Unknown            -87.6980      4.727    -18.552      0.000     -96.963     -78.433
    Visit Type_Virtual Care                  -28.6571      1.611    -17.792      0.000     -31.814     -25.500
    ==============================================================================
    Omnibus:                    90032.920   Durbin-Watson:                   1.985
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21699933.674
    Skew:                           5.593   Prob(JB):                         0.00
    Kurtosis:                      83.295   Cond. No.                     1.12e+05
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [3] The condition number is large, 1.12e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.
    




    0.11132790904846412

## Final Selected Model Prediction

![image](https://github.com/shivamsg08/Virtual-Healthcare/assets/8438005/1da32400-7220-4573-9900-5d656cfd657b)
