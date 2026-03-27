```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

```

```python
df=pd.read_csv('/content/weather_data.csv')
```

```python
df
```

```text
    hours_sunlight  humidity_level  daily_temperature
0             10.5              65               22.3
1              9.2              70               21.0
2              7.8              80               18.5
3              6.4              90               17.2
4              8.1              75               19.4
5             11.0              60               24.0
6              5.5              85               16.0
7              9.8              68               21.7
8              7.2              77               19.0
9              6.0              88               17.0
10             8.7              72               20.2
11            10.0              66               22.5
12             4.3              92               15.0
13             7.5              78               18.7
14             6.8              83               17.5
15            11.2              62               24.3
16             5.9              89               16.8
17             8.3              74               19.6
18             9.0              71               20.8
19             7.0              82               18.3
20            10.1              64               22.7
21             8.6              73               20.1
22             6.5              87               17.1
23             9.5              69               21.2
24             7.3              79               18.8
25             5.4              86               16.2
26            10.8              61               23.8
27             8.9              70               20.9
28             6.7              84               17.4
29             9.1              72               21.0
30            11.1              63               24.1
31             5.8              88               16.7
32             8.4              74               19.7
33             7.1              81               18.4
34            10.2              65               22.4
35             6.6              85               17.3
36             9.7              67               21.5
37             7.4              78               18.6
38             6.1              86               16.9
39             8.8              73               20.5
40            10.3              64               22.6
41             5.7              89               16.6
42             7.6              76               18.9
43             9.3              71               21.1
44            10.9              62               23.9
45             6.2              84               17.0
46             8.2              75               19.3
47             9.4              70               21.3
48             7.9              77               19.1
```

```python
df.shape
```

```text
(49, 3)
```

```python
df.columns
```

```text
Index(['hours_sunlight', 'humidity_level', 'daily_temperature'], dtype='object')
```

```python
model=LinearRegression()
model.fit(df[['hours_sunlight']],df['daily_temperature'])
```

```text
LinearRegression()
```

```python
model.intercept_
```

```text
np.float64(8.533832092006133)
```

```python
model.coef_
```

```text
array([1.36753934])
```

```python
df.head()
```

```text
   hours_sunlight  humidity_level  daily_temperature
0            10.5              65               22.3
1             9.2              70               21.0
2             7.8              80               18.5
3             6.4              90               17.2
4             8.1              75               19.4
```

```python
plt.scatter(df['hours_sunlight'],df['daily_temperature'])
plt.x
```

```python
x=df[['hours_sunlight']].values
y_pred=model.predict(x)
```

```python
plt.figure(figsize=(10,5))
plt.scatter(df['hours_sunlight'],df['daily_temperature'])
plt.plot(df['hours_sunlight'],y_pred,color='red')
plt.xlabel('hours_sunlight')
plt.ylabel('daily_temperature')
plt.show()
```

![output image 11-0](images/cell-11-0.png)

```python
plt.hist(df)
plt.xlabel('hours_sunlight')
plt.ylabel('daily_temperature')
plt.show()
```

![output image 12-0](images/cell-12-0.png)

