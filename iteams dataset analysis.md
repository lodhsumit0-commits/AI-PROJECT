---
jupyter:
  colab:
    authorship_tag: ABX9TyN4aHJRgQaJJJlO+GgjEUxb
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="7" executionInfo="{\"elapsed\":2101,\"status\":\"ok\",\"timestamp\":1770877308813,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="GFvMDM3ffqk9"}
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```
:::

::: {.cell .code execution_count="8" executionInfo="{\"elapsed\":13,\"status\":\"ok\",\"timestamp\":1770877313137,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="dH0whDMogoEU"}
``` python
df=pd.read_csv('/content/items.csv')
```
:::

::: {.cell .code execution_count="9" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":424}" executionInfo="{\"elapsed\":189,\"status\":\"ok\",\"timestamp\":1770877324444,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="O0-MB6awg3jk" outputId="051b01ed-48e0-439f-84ab-4239790e76fd"}
``` python
df
```

::: {.output .execute_result execution_count="9"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 1381,\n  \"fields\": [\n    {\n      \"column\": \"id\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 398,\n        \"min\": 0,\n        \"max\": 1380,\n        \"num_unique_values\": 1381,\n        \"samples\": [\n          309,\n          741,\n          265\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"name\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 421,\n        \"samples\": [\n          \"Google Woodtop Bottle Black\",\n          \"Google Felt Strap Keyring\",\n          \"Google Bear Baby Blanket Beige\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"brand\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          \"Android\",\n          \"Google Cloud\",\n          \"YouTube\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"variant\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 44,\n        \"samples\": [\n          \"BLUE\",\n          \" 6M\",\n          \" XXS\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"category\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 21,\n        \"samples\": [\n          \"Apparel\",\n          \"Writing Instruments\",\n          \"Fun\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"price_in_usd\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 23,\n        \"min\": 1,\n        \"max\": 313,\n        \"num_unique_values\": 70,\n        \"samples\": [\n          35,\n          14,\n          75\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code execution_count="13" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":29,\"status\":\"ok\",\"timestamp\":1770877466717,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="BfdwWg9jg540" outputId="42edad08-569e-41ba-b0c5-5855e77e1bd5"}
``` python
import os

files=os.listdir('/content/')
print(files)
```

::: {.output .stream .stdout}
    ['.config', 'items.csv', 'sample_data']
:::
:::

::: {.cell .code execution_count="19" executionInfo="{\"elapsed\":44,\"status\":\"ok\",\"timestamp\":1770878029483,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="vKI6U7SDiqKE"}
``` python
#load csv files into data frame
df=pd.read_csv(os.path.join('/content/', files[1]), encoding="latin1")
```
:::

::: {.cell .code execution_count="20" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":206}" executionInfo="{\"elapsed\":48,\"status\":\"ok\",\"timestamp\":1770878058902,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="r1cppjyTjo9t" outputId="7599f73b-5f47-40ff-88c3-6ef6f11e5859"}
``` python
df.head()
```

::: {.output .execute_result execution_count="20"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 1381,\n  \"fields\": [\n    {\n      \"column\": \"id\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 398,\n        \"min\": 0,\n        \"max\": 1380,\n        \"num_unique_values\": 1381,\n        \"samples\": [\n          309,\n          741,\n          265\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"name\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 421,\n        \"samples\": [\n          \"Google Woodtop Bottle Black\",\n          \"Google Felt Strap Keyring\",\n          \"Google Bear Baby Blanket Beige\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"brand\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          \"Android\",\n          \"Google Cloud\",\n          \"YouTube\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"variant\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 44,\n        \"samples\": [\n          \"BLUE\",\n          \" 6M\",\n          \" XXS\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"category\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 21,\n        \"samples\": [\n          \"Apparel\",\n          \"Writing Instruments\",\n          \"Fun\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"price_in_usd\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 23,\n        \"min\": 1,\n        \"max\": 313,\n        \"num_unique_values\": 70,\n        \"samples\": [\n          35,\n          14,\n          75\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code execution_count="21" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":24,\"status\":\"ok\",\"timestamp\":1770878085247,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="1NdRuaw-jvFk" outputId="dc272c6d-9cbd-45e1-c775-c29cedf58cef"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1381 entries, 0 to 1380
    Data columns (total 6 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   id            1381 non-null   int64 
     1   name          1381 non-null   object
     2   brand         1381 non-null   object
     3   variant       973 non-null    object
     4   category      1381 non-null   object
     5   price_in_usd  1381 non-null   int64 
    dtypes: int64(2), object(4)
    memory usage: 64.9+ KB
:::
:::

::: {.cell .code execution_count="22" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":272}" executionInfo="{\"elapsed\":175,\"status\":\"ok\",\"timestamp\":1770878120959,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="KfyRmgzZj0as" outputId="1617688e-5840-4c2b-ebfd-680bef9e0aec"}
``` python
df.isnull().sum()
```

::: {.output .execute_result execution_count="22"}
```{=html}
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>name</th>
      <td>0</td>
    </tr>
    <tr>
      <th>brand</th>
      <td>0</td>
    </tr>
    <tr>
      <th>variant</th>
      <td>408</td>
    </tr>
    <tr>
      <th>category</th>
      <td>0</td>
    </tr>
    <tr>
      <th>price_in_usd</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>
```
:::
:::

::: {.cell .code execution_count="23" executionInfo="{\"elapsed\":23,\"status\":\"ok\",\"timestamp\":1770878167310,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="BjKxiVxcj_L8"}
``` python
df=df.dropna()
```
:::

::: {.cell .code execution_count="24" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":272}" executionInfo="{\"elapsed\":23,\"status\":\"ok\",\"timestamp\":1770878203917,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="FfpIzAbokHd0" outputId="7f630a9c-fc92-4f2d-9251-42de994f22d8"}
``` python
df.isnull().sum()
```

::: {.output .execute_result execution_count="24"}
```{=html}
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>0</td>
    </tr>
    <tr>
      <th>name</th>
      <td>0</td>
    </tr>
    <tr>
      <th>brand</th>
      <td>0</td>
    </tr>
    <tr>
      <th>variant</th>
      <td>0</td>
    </tr>
    <tr>
      <th>category</th>
      <td>0</td>
    </tr>
    <tr>
      <th>price_in_usd</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>
```
:::
:::

::: {.cell .code execution_count="25" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":17,\"status\":\"ok\",\"timestamp\":1770878291226,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="y-Ilp9bqkXhE" outputId="58e5aa76-acf6-4bfa-f988-94b35cc29abc"}
``` python
df.fillna(method='ffill', inplace=True)
```

::: {.output .stream .stderr}
    /tmp/ipython-input-3970806690.py:1: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
      df.fillna(method='ffill', inplace=True)
    /tmp/ipython-input-3970806690.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df.fillna(method='ffill', inplace=True)
:::
:::

::: {.cell .code execution_count="26" executionInfo="{\"elapsed\":14,\"status\":\"ok\",\"timestamp\":1770878325825,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="YaNoGRLtkle8"}
``` python
df=df.drop_duplicates()
```
:::

::: {.cell .code execution_count="28" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":300}" executionInfo="{\"elapsed\":36,\"status\":\"ok\",\"timestamp\":1770878390373,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="MQ2FQ3gQku00" outputId="6728b125-d593-4471-ef35-3e7d1eb6384e"}
``` python
df.describe()
```

::: {.output .execute_result execution_count="28"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 8,\n  \"fields\": [\n    {\n      \"column\": \"id\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 435.1360882996779,\n        \"min\": 0.0,\n        \"max\": 1380.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          530.2189105858171,\n          486.0,\n          973.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"price_in_usd\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 333.7721139539433,\n        \"min\": 1.0,\n        \"max\": 973.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          27.564234326824256,\n          22.0,\n          973.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::
:::

::: {.cell .code execution_count="29" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":24,\"status\":\"ok\",\"timestamp\":1770878420685,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="FmVrH6Nnk0-c" outputId="90d29522-ca96-48f4-f3c5-782b8b1d89d0"}
``` python
df.columns
```

::: {.output .execute_result execution_count="29"}
    Index(['id', 'name', 'brand', 'variant', 'category', 'price_in_usd'], dtype='object')
:::
:::

::: {.cell .code execution_count="31" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":429}" executionInfo="{\"elapsed\":22,\"status\":\"ok\",\"timestamp\":1770878528710,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="HXNWij9mlGnE" outputId="6f324720-611c-4d5a-b59b-22383e3a3044"}
``` python
profit=df.groupby('category')['price_in_usd'].sum()
profit.sort_values(ascending=False).head(10)
```

::: {.output .execute_result execution_count="31"}
```{=html}
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
      <th>price_in_usd</th>
    </tr>
    <tr>
      <th>category</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Apparel</th>
      <td>16966</td>
    </tr>
    <tr>
      <th>Campus Collection</th>
      <td>2526</td>
    </tr>
    <tr>
      <th>Clearance</th>
      <td>1970</td>
    </tr>
    <tr>
      <th>Uncategorized Items</th>
      <td>1454</td>
    </tr>
    <tr>
      <th>Shop by Brand</th>
      <td>957</td>
    </tr>
    <tr>
      <th>Bags</th>
      <td>665</td>
    </tr>
    <tr>
      <th>New</th>
      <td>600</td>
    </tr>
    <tr>
      <th>Accessories</th>
      <td>550</td>
    </tr>
    <tr>
      <th>Lifestyle</th>
      <td>277</td>
    </tr>
    <tr>
      <th>Drinkware</th>
      <td>206</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>
```
:::
:::

::: {.cell .code execution_count="33" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":602}" executionInfo="{\"elapsed\":885,\"status\":\"ok\",\"timestamp\":1770878946886,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="sz-rR23YlkVc" outputId="acd6a0ee-78ca-454d-b758-99a879f75b17"}
``` python
profit.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top profitable products")
plt.ylabel("profit")
plt.show()
```

::: {.output .display_data}
![](vertopal_220b56030fc042f1bf61d9ae1c57167e/8a74f12daf515feef0c12d2e78867fa3e1af4b8d.png)
:::
:::

::: {.cell .code id="3ut8fXwupGGs"}
``` python
from sklearn.
```
:::
