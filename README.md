# Automatidata
Developing a regression model that helps estimate taxi fares before the ride, based on gathered data 
## Identify data types and relevant variables using Python
### PACE stages:
### PACE: Plan
### Task 1. Understand the situation
How can you best prepare to understand and organize the provided taxi cab information?
### Task 2a. Build dataframe
Create a pandas dataframe for data learning, and future exploratory data analysis (EDA) and statistical activities.
- import pandas as pd. pandas is used for buidling dataframes:
- import numpy as np. numpy is imported with pandas
- df = pd.read_csv('Datasets\NYC taxi data.csv')

```python
import pandas as pd
import numpy as np
df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')
```
### Task 2b. Understand the data - Inspect the data
View and inspect summary information about the dataframe by coding the following:
```python
df.head(10)
```
```python
df.info()
```
```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22699 entries, 0 to 22698
Data columns (total 18 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   Unnamed: 0             22699 non-null  int64  
 1   VendorID               22699 non-null  int64  
 2   tpep_pickup_datetime   22699 non-null  object 
 3   tpep_dropoff_datetime  22699 non-null  object 
 4   passenger_count        22699 non-null  int64  
 5   trip_distance          22699 non-null  float64
 6   RatecodeID             22699 non-null  int64  
 7   store_and_fwd_flag     22699 non-null  object 
 8   PULocationID           22699 non-null  int64  
 9   DOLocationID           22699 non-null  int64  
 10  payment_type           22699 non-null  int64  
 11  fare_amount            22699 non-null  float64
 12  extra                  22699 non-null  float64
 13  mta_tax                22699 non-null  float64
 14  tip_amount             22699 non-null  float64
 15  tolls_amount           22699 non-null  float64
 16  improvement_surcharge  22699 non-null  float64
 17  total_amount           22699 non-null  float64
dtypes: float64(8), int64(7), object(3)
memory usage: 3.1+ MB
```

```python
df.describe()
```
