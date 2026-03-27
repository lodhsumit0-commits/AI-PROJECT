```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
```

```python
df=pd.read_csv('/content/employeesalary.csv')
```

```python
df
```

```text
    Unnamed: 0      NAME       AGE    SALARY
0            0       Rob  0.058824  0.213675
1            1   Michael  0.176471  0.384615
2            2     Mohan  0.176471  0.136752
3            3    Ismail  0.117647  0.128205
4            4      Kory  0.941176  0.897436
5            5    Gautam  0.764706  0.940171
6            6     David  0.882353  0.982906
7            7    Andrea  0.705882  1.000000
8            8      Brad  0.588235  0.948718
9            9  Angelina  0.529412  0.726496
10          10    Donald  0.647059  0.786325
11          11       Tom  0.000000  0.000000
12          12    Arnold  0.058824  0.025641
13          13     Jared  0.117647  0.051282
14          14     Stark  0.176471  0.038462
15          15    Ranbir  0.352941  0.068376
16          16    Dipika  0.823529  0.170940
17          17  Priyanka  0.882353  0.153846
18          18      Nick  1.000000  0.162393
19          19      Alia  0.764706  0.299145
20          20       Sid  0.882353  0.316239
21          21     Abdul  0.764706  0.111111
```

```python
plt.scatter(df['AGE'],df['SALARY'])
```

```text
<matplotlib.collections.PathCollection at 0x7cfe0f59baa0>
```

![output image 3-1](images/cell-3-1.png)

```python
krange=range(1,10)
sse=[]
for i in krange:

     m=KMeans(i)
     m.fit(df[['AGE','SALARY']])
     sse.append(m.inertia_)
```

```python
sse
```

```text
[39748148458.54546,
 3318233941.1619043,
 1577046058.883117,
 1058800432.7238097,
 296500418.0952381,
 242500337.42857146,
 174800337.7,
 163550325.62857142,
 99466933.61666666]
```

```python
plt.plot(krange,sse,marker='o')
```

```text
[<matplotlib.lines.Line2D at 0x7cfe0f51b500>]
```

![output image 6-1](images/cell-6-1.png)

```python
model=KMeans(n_clusters=3)
model.fit(df[['AGE','SALARY']])
```

```text
KMeans(n_clusters=3)
```

```python
y=model.predict(df[['AGE','SALARY']])
```

```python
y
```

```text
array([2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
      dtype=int32)
```

```python
df['cluster']=y
```

```python
df
```

```text
    Unnamed: 0      NAME       AGE    SALARY  cluster
0            0       Rob  0.058824  0.213675        2
1            1   Michael  0.176471  0.384615        2
2            2     Mohan  0.176471  0.136752        0
3            3    Ismail  0.117647  0.128205        0
4            4      Kory  0.941176  0.897436        1
5            5    Gautam  0.764706  0.940171        1
6            6     David  0.882353  0.982906        1
7            7    Andrea  0.705882  1.000000        1
8            8      Brad  0.588235  0.948718        1
9            9  Angelina  0.529412  0.726496        1
10          10    Donald  0.647059  0.786325        1
11          11       Tom  0.000000  0.000000        0
12          12    Arnold  0.058824  0.025641        0
13          13     Jared  0.117647  0.051282        0
14          14     Stark  0.176471  0.038462        0
15          15    Ranbir  0.352941  0.068376        0
16          16    Dipika  0.823529  0.170940        0
17          17  Priyanka  0.882353  0.153846        0
18          18      Nick  1.000000  0.162393        0
19          19      Alia  0.764706  0.299145        2
20          20       Sid  0.882353  0.316239        2
21          21     Abdul  0.764706  0.111111        0
```

```python
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
```

```python
plt.scatter(df1['AGE'],df1['SALARY'],color='blue',label='cluster1')
plt.scatter(df2['AGE'],df2['SALARY'],color='yellow',label='cluster2')
plt.scatter(df3['AGE'],df3['SALARY'],color='red',label='cluster3')
plt.legend()
```

```text
<matplotlib.legend.Legend at 0x7cfe11f199d0>
```

![output image 13-1](images/cell-13-1.png)

```python
scaler=MinMaxScaler()
```

```python
scaler.fit(df[['AGE']])
df['AGE']=scaler.transform(df[['AGE']])
```

```python
scaler.fit(df[['SALARY']])
df['SALARY']=scaler.transform(df[['SALARY']])
```

```python
df
```

```text
    Unnamed: 0      NAME       AGE    SALARY  cluster
0            0       Rob  0.058824  0.213675        2
1            1   Michael  0.176471  0.384615        2
2            2     Mohan  0.176471  0.136752        0
3            3    Ismail  0.117647  0.128205        0
4            4      Kory  0.941176  0.897436        1
5            5    Gautam  0.764706  0.940171        1
6            6     David  0.882353  0.982906        1
7            7    Andrea  0.705882  1.000000        1
8            8      Brad  0.588235  0.948718        1
9            9  Angelina  0.529412  0.726496        1
10          10    Donald  0.647059  0.786325        1
11          11       Tom  0.000000  0.000000        0
12          12    Arnold  0.058824  0.025641        0
13          13     Jared  0.117647  0.051282        0
14          14     Stark  0.176471  0.038462        0
15          15    Ranbir  0.352941  0.068376        0
16          16    Dipika  0.823529  0.170940        0
17          17  Priyanka  0.882353  0.153846        0
18          18      Nick  1.000000  0.162393        0
19          19      Alia  0.764706  0.299145        2
20          20       Sid  0.882353  0.316239        2
21          21     Abdul  0.764706  0.111111        0
```

```python
df.drop('cluster', axis=1)
```

```text
    Unnamed: 0      NAME       AGE    SALARY
0            0       Rob  0.058824  0.213675
1            1   Michael  0.176471  0.384615
2            2     Mohan  0.176471  0.136752
3            3    Ismail  0.117647  0.128205
4            4      Kory  0.941176  0.897436
5            5    Gautam  0.764706  0.940171
6            6     David  0.882353  0.982906
7            7    Andrea  0.705882  1.000000
8            8      Brad  0.588235  0.948718
9            9  Angelina  0.529412  0.726496
10          10    Donald  0.647059  0.786325
11          11       Tom  0.000000  0.000000
12          12    Arnold  0.058824  0.025641
13          13     Jared  0.117647  0.051282
14          14     Stark  0.176471  0.038462
15          15    Ranbir  0.352941  0.068376
16          16    Dipika  0.823529  0.170940
17          17  Priyanka  0.882353  0.153846
18          18      Nick  1.000000  0.162393
19          19      Alia  0.764706  0.299145
20          20       Sid  0.882353  0.316239
21          21     Abdul  0.764706  0.111111
```

```python
df
```

```text
    Unnamed: 0      NAME       AGE    SALARY  cluster
0            0       Rob  0.058824  0.213675        2
1            1   Michael  0.176471  0.384615        2
2            2     Mohan  0.176471  0.136752        0
3            3    Ismail  0.117647  0.128205        0
4            4      Kory  0.941176  0.897436        1
5            5    Gautam  0.764706  0.940171        1
6            6     David  0.882353  0.982906        1
7            7    Andrea  0.705882  1.000000        1
8            8      Brad  0.588235  0.948718        1
9            9  Angelina  0.529412  0.726496        1
10          10    Donald  0.647059  0.786325        1
11          11       Tom  0.000000  0.000000        0
12          12    Arnold  0.058824  0.025641        0
13          13     Jared  0.117647  0.051282        0
14          14     Stark  0.176471  0.038462        0
15          15    Ranbir  0.352941  0.068376        0
16          16    Dipika  0.823529  0.170940        0
17          17  Priyanka  0.882353  0.153846        0
18          18      Nick  1.000000  0.162393        0
19          19      Alia  0.764706  0.299145        2
20          20       Sid  0.882353  0.316239        2
21          21     Abdul  0.764706  0.111111        0
```

```python
df.drop('cluster', axis=1,inplace=True)
```

```python
df
```

```text
    Unnamed: 0      NAME       AGE    SALARY
0            0       Rob  0.058824  0.213675
1            1   Michael  0.176471  0.384615
2            2     Mohan  0.176471  0.136752
3            3    Ismail  0.117647  0.128205
4            4      Kory  0.941176  0.897436
5            5    Gautam  0.764706  0.940171
6            6     David  0.882353  0.982906
7            7    Andrea  0.705882  1.000000
8            8      Brad  0.588235  0.948718
9            9  Angelina  0.529412  0.726496
10          10    Donald  0.647059  0.786325
11          11       Tom  0.000000  0.000000
12          12    Arnold  0.058824  0.025641
13          13     Jared  0.117647  0.051282
14          14     Stark  0.176471  0.038462
15          15    Ranbir  0.352941  0.068376
16          16    Dipika  0.823529  0.170940
17          17  Priyanka  0.882353  0.153846
18          18      Nick  1.000000  0.162393
19          19      Alia  0.764706  0.299145
20          20       Sid  0.882353  0.316239
21          21     Abdul  0.764706  0.111111
```

```python
model1=KMeans(n_clusters=3)
```

```python
model1.fit(df[['AGE','SALARY']])
```

```text
KMeans(n_clusters=3)
```

```python
y1=model1.predict(df[['AGE','SALARY']])
```

```python
y1
```

```text
array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
      dtype=int32)
```

```python
df['cluster']=y1
```

```python
df
```

```text
    Unnamed: 0      NAME       AGE    SALARY  cluster
0            0       Rob  0.058824  0.213675        1
1            1   Michael  0.176471  0.384615        1
2            2     Mohan  0.176471  0.136752        1
3            3    Ismail  0.117647  0.128205        1
4            4      Kory  0.941176  0.897436        2
5            5    Gautam  0.764706  0.940171        2
6            6     David  0.882353  0.982906        2
7            7    Andrea  0.705882  1.000000        2
8            8      Brad  0.588235  0.948718        2
9            9  Angelina  0.529412  0.726496        2
10          10    Donald  0.647059  0.786325        2
11          11       Tom  0.000000  0.000000        1
12          12    Arnold  0.058824  0.025641        1
13          13     Jared  0.117647  0.051282        1
14          14     Stark  0.176471  0.038462        1
15          15    Ranbir  0.352941  0.068376        1
16          16    Dipika  0.823529  0.170940        0
17          17  Priyanka  0.882353  0.153846        0
18          18      Nick  1.000000  0.162393        0
19          19      Alia  0.764706  0.299145        0
20          20       Sid  0.882353  0.316239        0
21          21     Abdul  0.764706  0.111111        0
```

```python
model.cluster_centers_
```

```text
array([[3.29090909e+01, 5.61363636e+04],
       [3.82857143e+01, 1.50000000e+05],
       [3.40000000e+01, 8.05000000e+04]])
```

```python
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1['AGE'],df1['SALARY'],color='blue',label='cluster1')
plt.scatter(df2['AGE'],df2['SALARY'],color='yellow',label='cluster2')
plt.scatter(df3['AGE'],df3['SALARY'],color='red',label='cluster3')
plt.legend()
```

```text
<matplotlib.legend.Legend at 0x7cfe11f69a90>
```

![output image 29-1](images/cell-29-1.png)

