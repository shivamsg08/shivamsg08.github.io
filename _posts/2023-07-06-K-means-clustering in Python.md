---
layout: post
cover-img: /DataFiles/Posts/Clustering/Images/clustering-algorithms-in-Machine-Learning.jpg
title: K Means Clustering with Python
subtitle: Using jupyter notebook
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
comments: true

thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

This is a demo post to show you how to write blog posts with markdown.


# Clustering using K means Algorithm

@ Author : Shivam Goyal

Date : 9 January, 2022

Version 1

### Importing the libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
import csv
sns.set_style('darkgrid')
%config InlineBackend.figure_format = 'retina'
%precision %.2f  ## magic command to display 2 decimal places

# Clustering package
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
```


```python
import warnings
warnings.filterwarnings('ignore')
```

### Define the path and list of required files


```python
path = r"D:/UoW/Semester 1/Quantitative Studies -BSMM 8320/Excel Projects/Project 2/Output" 
```


```python
# locate the data to be uploaded 
for i in os.listdir(path):
    if i.endswith(".csv"): # select ony the csv files
        print(i)
```

    clustering_raw_data.csv
    


```python
clustering_raw_data = pd.read_csv(path+'/clustering_raw_data.csv')
```


```python
clustering_raw_data.columns
```




    Index(['Customer Lifetime Value', 'Monthly Premium Auto',
           'Months Since Last Claim', 'Months Since Policy Inception',
           'Number of Open Complaints', 'Number of Policies', 'Total Claim Amount',
           'Coverage', 'Policy Type', 'Policy', 'Renew Offer Type',
           'Sales Channel', 'Vehicle Size'],
          dtype='object')



### Analyzing the variables for clustering


```python
num = clustering_raw_data.select_dtypes(include=np.number)  # Get numeric columns
n = num.shape[1]  # Number of cols

fig, axes = plt.subplots(n, 1, figsize=(24/2.54, 70/2.54))  # create subplots with n rows and 1 column

for ax, col in zip(axes, num):  # For each column...
    sns.distplot(num[col], ax=ax)   # Plot histogaerm
    ax.set(ylabel= col)
    ax.axvline(num[col].mean(), c='k')  # Plot mean
```


    
![EDA charts](/DataFiles/Posts/Clustering/Images/output_10_0.png)
    


### Check for Correlation in the data


```python
plt.figure(figsize=(10,8))
sns.heatmap(clustering_raw_data.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu")
plt.show()
```


    
![pic](/DataFiles/Posts/Clustering/Images/output_12_0.png)
    



```python
### List columns that can be dropped
columns_drop = ['Policy Type', 'Vehicle Size']
```

### Final data for Clustering after removing unnecessary variables


```python
clustering_raw_data.drop(columns=columns_drop, inplace = True)
```

#### Scatter Plot


```python
g = sns.PairGrid(clustering_raw_data)
g.map(sns.scatterplot);
```


    
![output17](/DataFiles/Posts/Clustering/Images/output_17_0.png)
    


#### Standardization


```python
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(clustering_raw_data) 
scaled_df
```




    array([[-0.76206445, -0.70219428,  1.67841106, ..., -1.14631002,
            -0.96307902, -1.03103509],
           [-0.14630573,  0.0249932 , -0.20538206, ...,  0.72219223,
             1.01635938, -1.03103509],
           [ 0.71655576,  0.43221818,  0.29035297, ...,  0.72219223,
            -0.96307902, -1.03103509],
           ...,
           [-0.43243113, -0.87671927,  0.786088  , ...,  0.72219223,
            -0.96307902, -1.03103509],
           [-0.45956231, -0.78945677,  1.57926405, ...,  0.72219223,
             0.02664018,  0.83309108],
           [ 1.40856681, -0.87671927,  1.48011705, ...,  0.72219223,
             0.02664018, -1.03103509]])



#### Normalization


```python
normalized_df = normalize(scaled_df) 
normalized_df = pd.DataFrame(normalized_df)
normalized_df
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.229530</td>
      <td>-0.211497</td>
      <td>0.505529</td>
      <td>-0.467190</td>
      <td>-0.128236</td>
      <td>-0.247465</td>
      <td>-0.049014</td>
      <td>-0.221104</td>
      <td>-0.345263</td>
      <td>-0.290075</td>
      <td>-0.310543</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.039185</td>
      <td>0.006694</td>
      <td>-0.055008</td>
      <td>-0.059191</td>
      <td>-0.114032</td>
      <td>0.564325</td>
      <td>0.653961</td>
      <td>0.212106</td>
      <td>0.193427</td>
      <td>0.272214</td>
      <td>-0.276145</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.234110</td>
      <td>0.141212</td>
      <td>0.094863</td>
      <td>-0.119184</td>
      <td>-0.139102</td>
      <td>-0.131744</td>
      <td>0.153857</td>
      <td>0.757311</td>
      <td>0.235951</td>
      <td>-0.314652</td>
      <td>-0.336855</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.016298</td>
      <td>0.124442</td>
      <td>0.096599</td>
      <td>0.201558</td>
      <td>-0.141648</td>
      <td>0.561799</td>
      <td>0.114211</td>
      <td>-0.244228</td>
      <td>-0.588586</td>
      <td>-0.320412</td>
      <td>0.277166</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.315172</td>
      <td>-0.244644</td>
      <td>-0.127169</td>
      <td>-0.062264</td>
      <td>-0.177793</td>
      <td>-0.343099</td>
      <td>-0.427272</td>
      <td>-0.306550</td>
      <td>-0.218600</td>
      <td>-0.402175</td>
      <td>-0.430553</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8094</th>
      <td>0.118296</td>
      <td>-0.177388</td>
      <td>-0.601452</td>
      <td>-0.319447</td>
      <td>-0.197588</td>
      <td>-0.187136</td>
      <td>0.140186</td>
      <td>0.367524</td>
      <td>0.335158</td>
      <td>0.012363</td>
      <td>0.386624</td>
    </tr>
    <tr>
      <th>8095</th>
      <td>0.008330</td>
      <td>0.264696</td>
      <td>0.187942</td>
      <td>-0.106637</td>
      <td>-0.205438</td>
      <td>0.612927</td>
      <td>0.012367</td>
      <td>-0.354214</td>
      <td>0.348474</td>
      <td>-0.464707</td>
      <td>-0.047756</td>
    </tr>
    <tr>
      <th>8096</th>
      <td>-0.173029</td>
      <td>-0.350802</td>
      <td>0.314537</td>
      <td>-0.318578</td>
      <td>-0.170359</td>
      <td>0.006056</td>
      <td>-0.359218</td>
      <td>-0.293731</td>
      <td>0.288971</td>
      <td>-0.385357</td>
      <td>-0.412548</td>
    </tr>
    <tr>
      <th>8097</th>
      <td>-0.095778</td>
      <td>-0.164531</td>
      <td>0.329136</td>
      <td>0.156231</td>
      <td>0.823254</td>
      <td>0.264735</td>
      <td>-0.083356</td>
      <td>-0.152992</td>
      <td>0.150513</td>
      <td>0.005552</td>
      <td>0.173625</td>
    </tr>
    <tr>
      <th>8098</th>
      <td>0.465841</td>
      <td>-0.289948</td>
      <td>0.489504</td>
      <td>-0.358428</td>
      <td>-0.140807</td>
      <td>-0.133359</td>
      <td>-0.246763</td>
      <td>-0.242778</td>
      <td>0.238843</td>
      <td>0.008810</td>
      <td>-0.340984</td>
    </tr>
  </tbody>
</table>
<p>8099 rows × 11 columns</p>
</div>



### K Means Loop


```python
### The Elbow Method

sse = {}

for k in range(1, 16):
    kmeans = KMeans(n_clusters=k, init='k-means++',random_state= 0 ,max_iter=300).fit(normalized_df)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

    plt.figure()
plt.plot(list(sse.keys()), list(sse.values()),marker='*')
plt.title('Elbow Method')                               # Set plot title
plt.xlabel('Number of clusters')                        # Set x axis name
plt.ylabel('Within Cluster Sum of Squares (WCSS)') 
plt.show()
```

    
![png](/DataFiles/Posts/Clustering/Images/output_23_14.png)
    



```python
#### The Silhouette Coefficient Method

silhouette_scores = [] 

for n_cluster in range(2, 15):
    silhouette_scores.append( 
        silhouette_score(normalized_df, KMeans(n_clusters = n_cluster).fit_predict(normalized_df))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7, 8,9, 10,11, 12,13, 14] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 
```


    
![png](/DataFiles/Posts/Clustering/Images/output_24_0.png)
    


### Summary of clusters


```python
def cluster_summary(k,norm_data): ## k is no of cluster $ norm_data is normalized data
    final_clusters = KMeans(n_clusters = k).fit_predict(norm_data)
    final_clusters_data = pd.concat([pd.DataFrame(final_clusters),clustering_raw_data],axis=1)
    final_clusters_data.rename(columns={0:"Cluster_Label"}, inplace= True)
    metrics = final_clusters_data.groupby("Cluster_Label").size().reset_index(name = "Distribution")
#     metrics["Perc"] = metrics.groupby("Cluster_Label")["Distribution"].apply(lambda x: x/float(x.sum()))
    metrics["Perc"] = 100*metrics["Distribution"]/metrics["Distribution"].sum()
    return metrics
    
metrics = cluster_summary(5,normalized_df)   
```


```python
metrics
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
      <th>Cluster_Label</th>
      <th>Distribution</th>
      <th>Perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2052</td>
      <td>25.336461</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1337</td>
      <td>16.508211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2107</td>
      <td>26.015557</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1774</td>
      <td>21.903939</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>829</td>
      <td>10.235832</td>
    </tr>
  </tbody>
</table>
</div>




```python
kmeans1 = KMeans(n_clusters=5, init='k-means++',random_state= 0 ,max_iter=300).fit(normalized_df)
kmeans1.cluster_centers_
```




    array([[-0.04920795, -0.06141981,  0.02923496, -0.01958611,  0.59769328,
            -0.04056441, -0.05085618, -0.05004137,  0.01505635, -0.0722779 ,
            -0.02862492],
           [ 0.15252393,  0.24578779,  0.0053302 , -0.01923499, -0.08539613,
            -0.08718466,  0.21197069,  0.28563643,  0.0087514 , -0.0690205 ,
            -0.03748331],
           [-0.10595862, -0.17538956, -0.08656241,  0.17939367, -0.131911  ,
            -0.1792572 , -0.13676129, -0.20394053,  0.01390859, -0.18575648,
            -0.21325831],
           [-0.07251267, -0.08624036,  0.00896877, -0.02048995, -0.08638391,
             0.55051008, -0.06158958, -0.06240628,  0.00109123, -0.06119222,
            -0.01139739],
           [-0.1062052 , -0.15340922,  0.03206375, -0.13361091, -0.11164804,
            -0.16096557, -0.13028584, -0.132813  ,  0.00401019,  0.28438093,
             0.2230994 ]])




```python

```
