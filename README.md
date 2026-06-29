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
df.info()
df.describe()
```
