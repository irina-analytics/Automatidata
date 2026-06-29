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

|    |   Unnamed: 0 |   VendorID | tpep_pickup_datetime   | tpep_dropoff_datetime   |   passenger_count |   trip_distance |   RatecodeID | store_and_fwd_flag   |   PULocationID |   DOLocationID |   payment_type |   fare_amount |   extra |   mta_tax |   tip_amount |   tolls_amount |   improvement_surcharge |   total_amount |
|---:|-------------:|-----------:|:-----------------------|:------------------------|------------------:|----------------:|-------------:|:---------------------|---------------:|---------------:|---------------:|--------------:|--------:|----------:|-------------:|---------------:|------------------------:|---------------:|
|  0 |     24870114 |          2 | 03/25/2017 8:55:43 AM  | 03/25/2017 9:09:47 AM   |                 6 |            3.34 |            1 | N                    |            100 |            231 |              1 |          13   |     0   |       0.5 |         2.76 |              0 |                     0.3 |          16.56 |
|  1 |     35634249 |          1 | 04/11/2017 2:53:28 PM  | 04/11/2017 3:19:58 PM   |                 1 |            1.8  |            1 | N                    |            186 |             43 |              1 |          16   |     0   |       0.5 |         4    |              0 |                     0.3 |          20.8  |
|  2 |    106203690 |          1 | 12/15/2017 7:26:56 AM  | 12/15/2017 7:34:08 AM   |                 1 |            1    |            1 | N                    |            262 |            236 |              1 |           6.5 |     0   |       0.5 |         1.45 |              0 |                     0.3 |           8.75 |
|  3 |     38942136 |          2 | 05/07/2017 1:17:59 PM  | 05/07/2017 1:48:14 PM   |                 1 |            3.7  |            1 | N                    |            188 |             97 |              1 |          20.5 |     0   |       0.5 |         6.39 |              0 |                     0.3 |          27.69 |
|  4 |     30841670 |          2 | 04/15/2017 11:32:20 PM | 04/15/2017 11:49:03 PM  |                 1 |            4.37 |            1 | N                    |              4 |            112 |              2 |          16.5 |     0.5 |       0.5 |         0    |              0 |                     0.3 |          17.8  |
|  5 |     23345809 |          2 | 03/25/2017 8:34:11 PM  | 03/25/2017 8:42:11 PM   |                 6 |            2.3  |            1 | N                    |            161 |            236 |              1 |           9   |     0.5 |       0.5 |         2.06 |              0 |                     0.3 |          12.36 |
|  6 |     37660487 |          2 | 05/03/2017 7:04:09 PM  | 05/03/2017 8:03:47 PM   |                 1 |           12.83 |            1 | N                    |             79 |            241 |              1 |          47.5 |     1   |       0.5 |         9.86 |              0 |                     0.3 |          59.16 |
|  7 |     69059411 |          2 | 08/15/2017 5:41:06 PM  | 08/15/2017 6:03:05 PM   |                 1 |            2.98 |            1 | N                    |            237 |            114 |              1 |          16   |     1   |       0.5 |         1.78 |              0 |                     0.3 |          19.58 |
|  8 |      8433159 |          2 | 02/04/2017 4:17:07 PM  | 02/04/2017 4:29:14 PM   |                 1 |            1.2  |            1 | N                    |            234 |            249 |              2 |           9   |     0   |       0.5 |         0    |              0 |                     0.3 |           9.8  |
|  9 |     95294817 |          1 | 11/10/2017 3:20:29 PM  | 11/10/2017 3:40:55 PM   |                 1 |            1.6  |            1 | N                    |            239 |            237 |              1 |          13   |     0   |       0.5 |         2.75 |              0 |                     0.3 |          16.55 |

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
