Student Performance Dataset from Kaggle

```python

import kagglehub

# Download latest version
path = kagglehub.dataset_download("larsen0966/student-performance-data-set")

print("Path to dataset files:", path)
```

```text
Using Colab cache for faster access to the 'student-performance-data-set' dataset.
Path to dataset files: /kaggle/input/student-performance-data-set
```

```python
import os
files=os.listdir(path)
print(files)
```

```text
['student-por.csv']
```

```python
import pandas as pd
df=pd.read_csv(os.path.join(path,files[0]),encoding="latin1")
df.head(10)
```

```text
  school sex  age address famsize Pstatus  Medu  Fedu      Mjob      Fjob  \
0     GP   F   18       U     GT3       A     4     4   at_home   teacher   
1     GP   F   17       U     GT3       T     1     1   at_home     other   
2     GP   F   15       U     LE3       T     1     1   at_home     other   
3     GP   F   15       U     GT3       T     4     2    health  services   
4     GP   F   16       U     GT3       T     3     3     other     other   
5     GP   M   16       U     LE3       T     4     3  services     other   
6     GP   M   16       U     LE3       T     2     2     other     other   
7     GP   F   17       U     GT3       A     4     4     other   teacher   
8     GP   M   15       U     LE3       A     3     2  services     other   
9     GP   M   15       U     GT3       T     3     4     other     other   

   ... famrel freetime  goout  Dalc  Walc health absences  G1  G2  G3  
0  ...      4        3      4     1     1      3        4   0  11  11  
1  ...      5        3      3     1     1      3        2   9  11  11  
2  ...      4        3      2     2     3      3        6  12  13  12  
3  ...      3        2      2     1     1      5        0  14  14  14  
4  ...      4        3      2     1     2      5        0  11  13  13  
5  ...      5        4      2     1     2      5        6  12  12  13  
6  ...      4        4      4     1     1      3        0  13  12  13  
7  ...      4        1      4     1     1      1        2  10  13  13  
8  ...      4        2      2     1     1      1        0  15  16  17  
9  ...      5        5      1     1     1      5        0  12  12  13  

[10 rows x 33 columns]
```

```python
df.info()
```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 649 entries, 0 to 648
Data columns (total 33 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   school      649 non-null    object
 1   sex         649 non-null    object
 2   age         649 non-null    int64 
 3   address     649 non-null    object
 4   famsize     649 non-null    object
 5   Pstatus     649 non-null    object
 6   Medu        649 non-null    int64 
 7   Fedu        649 non-null    int64 
 8   Mjob        649 non-null    object
 9   Fjob        649 non-null    object
 10  reason      649 non-null    object
 11  guardian    649 non-null    object
 12  traveltime  649 non-null    int64 
 13  studytime   649 non-null    int64 
 14  failures    649 non-null    int64 
 15  schoolsup   649 non-null    object
 16  famsup      649 non-null    object
 17  paid        649 non-null    object
 18  activities  649 non-null    object
 19  nursery     649 non-null    object
 20  higher      649 non-null    object
 21  internet    649 non-null    object
 22  romantic    649 non-null    object
 23  famrel      649 non-null    int64 
 24  freetime    649 non-null    int64 
 25  goout       649 non-null    int64 
 26  Dalc        649 non-null    int64 
 27  Walc        649 non-null    int64 
 28  health      649 non-null    int64 
 29  absences    649 non-null    int64 
 30  G1          649 non-null    int64 
 31  G2          649 non-null    int64 
 32  G3          649 non-null    int64 
dtypes: int64(16), object(17)
memory usage: 167.4+ KB
```

```python
df.isnull().sum()
```

```text
school        0
sex           0
age           0
address       0
famsize       0
Pstatus       0
Medu          0
Fedu          0
Mjob          0
Fjob          0
reason        0
guardian      0
traveltime    0
studytime     0
failures      0
schoolsup     0
famsup        0
paid          0
activities    0
nursery       0
higher        0
internet      0
romantic      0
famrel        0
freetime      0
goout         0
Dalc          0
Walc          0
health        0
absences      0
G1            0
G2            0
G3            0
dtype: int64
```

```python
df.columns
```

```text
Index(['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G1', 'G2', 'G3'],
      dtype='object')
```

```python
df.describe()
```

```text
              age        Medu        Fedu  traveltime   studytime    failures  \
count  649.000000  649.000000  649.000000  649.000000  649.000000  649.000000   
mean    16.744222    2.514638    2.306626    1.568567    1.930663    0.221880   
std      1.218138    1.134552    1.099931    0.748660    0.829510    0.593235   
min     15.000000    0.000000    0.000000    1.000000    1.000000    0.000000   
25%     16.000000    2.000000    1.000000    1.000000    1.000000    0.000000   
50%     17.000000    2.000000    2.000000    1.000000    2.000000    0.000000   
75%     18.000000    4.000000    3.000000    2.000000    2.000000    0.000000   
max     22.000000    4.000000    4.000000    4.000000    4.000000    3.000000   

           famrel    freetime       goout        Dalc        Walc      health  \
count  649.000000  649.000000  649.000000  649.000000  649.000000  649.000000   
mean     3.930663    3.180277    3.184900    1.502311    2.280431    3.536210   
std      0.955717    1.051093    1.175766    0.924834    1.284380    1.446259   
min      1.000000    1.000000    1.000000    1.000000    1.000000    1.000000   
25%      4.000000    3.000000    2.000000    1.000000    1.000000    2.000000   
50%      4.000000    3.000000    3.000000    1.000000    2.000000    4.000000   
75%      5.000000    4.000000    4.000000    2.000000    3.000000    5.000000   
max      5.000000    5.000000    5.000000    5.000000    5.000000    5.000000   

         absences          G1          G2          G3  
count  649.000000  649.000000  649.000000  649.000000  
mean     3.659476   11.399076   11.570108   11.906009  
std      4.640759    2.745265    2.913639    3.230656  
min      0.000000    0.000000    0.000000    0.000000  
25%      0.000000   10.000000   10.000000   10.000000  
50%      2.000000   11.000000   11.000000   12.000000  
75%      6.000000   13.000000   13.000000   14.000000  
max     32.000000   19.000000   19.000000   19.000000  
```

```python
df = df.drop(columns=['G1', 'G2'])
```

```python
df = df.corr(numeric_only=True)['G3'].sort_values()
```

```python
print(df.groupby('studytime')['G3'].mean())
```

```text
studytime
1    10.844340
2    12.091803
3    13.226804
4    13.057143
Name: G3, dtype: float64
```

```python
print(df.groupby('failures')['G3'].mean())
```

```text
failures
0    12.510018
1     8.642857
2     8.812500
3     8.071429
Name: G3, dtype: float64
```

```python
print(df.groupby('goout')['G3'].mean())
```

```text
goout
1    10.729167
2    12.668966
3    12.151220
4    11.971631
5    10.872727
Name: G3, dtype: float64
```

```python
print(df[['absences','G3']].corr())
```

```text
          absences        G3
absences  1.000000 -0.091379
G3       -0.091379  1.000000
```

```python
import matplotlib.pyplot as plt

df.groupby('studytime')['G3'].mean().plot(kind='bar')
plt.title("Study Time vs Remark")
plt.show()
```

![output image 14-0](images/cell-14-0.png)

```python
df_corr = df.corr(numeric_only=True)['G3'].sort_values()
df1 = df_corr[abs(df_corr) > 0.2]
print(df1)
```

```text
failures    -0.393316
Dalc        -0.204719
Fedu         0.211800
Medu         0.240151
studytime    0.249789
G1           0.826387
G2           0.918548
G3           1.000000
Name: G3, dtype: float64
```

```python
df1.plot(kind='barh')
plt.show()
```

![output image 16-0](images/cell-16-0.png)

```python
df['Total']=df[['G1','G2','G3']].sum(axis=1)
```

```python
df['Total'].mean()
```

```text
np.float64(34.875192604006166)
```

```python
df.groupby('Total')['G3'].mean()
```

```text
Total
4      0.000000
5      0.000000
7      0.000000
8      0.000000
9      0.000000
12     0.000000
13     0.000000
14     0.000000
15     0.000000
16     0.000000
17     6.500000
19     7.000000
20     5.400000
21     6.625000
22     7.818182
23     8.000000
24     8.333333
25     8.583333
26     9.050000
27     9.500000
28     9.761905
29    10.076923
30    10.380952
31    10.551724
32    10.864865
33    11.419355
34    11.440000
35    11.956522
36    12.689655
37    12.736842
38    13.120000
39    13.424242
40    13.380952
41    14.000000
42    14.466667
43    15.000000
44    15.095238
45    15.428571
46    16.230769
47    16.250000
48    16.416667
49    16.714286
50    17.000000
51    17.428571
52    17.428571
53    17.800000
54    18.166667
56    19.000000
Name: G3, dtype: float64
```

```python
df.groupby('studytime')['Total'].mean()
```

```text
studytime
1    32.051887
2    35.291803
3    38.443299
4    38.457143
Name: Total, dtype: float64
```

```python
from sklearn.linear_model import LinearRegression
x=df[['studytime']]
y=df['Total']
model=LinearRegression()
model.fit(x,y)
```

```text
LinearRegression()
```

```python
print("Base score:",model.intercept_)
print("Increse in total score per studytime lavel",model.coef_[0])
```

```text
Base score: 29.69917339378525
Increse in total score per studytime lavel 2.6809548822293485
```

```python
model.predict([[1],[2],[12],[8]])
```

```text
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
  warnings.warn(
```

```text
array([32.38012828, 35.06108316, 61.87063198, 51.14681245])
```

