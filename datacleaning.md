---
jupyter:
  colab:
    authorship_tag: ABX9TyM73IO2ZeKsbdzU0BvXUVb/
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="1" executionInfo="{\"elapsed\":1573,\"status\":\"ok\",\"timestamp\":1771825511468,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="C-PDpCvu-z40"}
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

```
:::

::: {.cell .code execution_count="3" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":424}" executionInfo="{\"elapsed\":117,\"status\":\"ok\",\"timestamp\":1771825561853,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="9kdMwk5LB8V1" outputId="16fbd8df-d8c5-42e6-f101-81185d55048c"}
``` python
df=pd.read_csv('/content/dirty_cafe_sales.csv')
df
```

::: {.output .execute_result execution_count="3"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 10000,\n  \"fields\": [\n    {\n      \"column\": \"Transaction ID\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 10000,\n        \"samples\": [\n          \"TXN_2919952\",\n          \"TXN_4265056\",\n          \"TXN_2463115\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Item\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 10,\n        \"samples\": [\n          \"Juice\",\n          \"Cake\",\n          \"UNKNOWN\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Quantity\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 7,\n        \"samples\": [\n          \"2\",\n          \"4\",\n          \"ERROR\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Price Per Unit\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 8,\n        \"samples\": [\n          \"3.0\",\n          \"1.5\",\n          \"2.0\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Total Spent\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 19,\n        \"samples\": [\n          \"4.0\",\n          \"9.0\",\n          \"3.0\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Payment Method\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          \"Cash\",\n          \"ERROR\",\n          \"UNKNOWN\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Location\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"In-store\",\n          \"ERROR\",\n          \"Takeaway\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Transaction Date\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 367,\n        \"samples\": [\n          \"2023-05-09\",\n          \"2023-05-28\",\n          \"2023-11-15\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code execution_count="4" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":206}" executionInfo="{\"elapsed\":114,\"status\":\"ok\",\"timestamp\":1771825582318,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="rzm4dIwfCJyl" outputId="7dd2fca1-0418-4d4d-a0ca-716fe1653366"}
``` python
df.head()
```

::: {.output .execute_result execution_count="4"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 10000,\n  \"fields\": [\n    {\n      \"column\": \"Transaction ID\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 10000,\n        \"samples\": [\n          \"TXN_2919952\",\n          \"TXN_4265056\",\n          \"TXN_2463115\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Item\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 10,\n        \"samples\": [\n          \"Juice\",\n          \"Cake\",\n          \"UNKNOWN\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Quantity\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 7,\n        \"samples\": [\n          \"2\",\n          \"4\",\n          \"ERROR\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Price Per Unit\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 8,\n        \"samples\": [\n          \"3.0\",\n          \"1.5\",\n          \"2.0\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Total Spent\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 19,\n        \"samples\": [\n          \"4.0\",\n          \"9.0\",\n          \"3.0\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Payment Method\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          \"Cash\",\n          \"ERROR\",\n          \"UNKNOWN\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Location\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4,\n        \"samples\": [\n          \"In-store\",\n          \"ERROR\",\n          \"Takeaway\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Transaction Date\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 367,\n        \"samples\": [\n          \"2023-05-09\",\n          \"2023-05-28\",\n          \"2023-11-15\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code execution_count="5" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":335}" executionInfo="{\"elapsed\":52,\"status\":\"ok\",\"timestamp\":1771825659482,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="LwJjVucCCN3d" outputId="f17d25c1-334c-4501-a436-4d2e2aae9195"}
``` python
pd.isnull(df).sum()
```

::: {.output .execute_result execution_count="5"}
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
      <th>Transaction ID</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Item</th>
      <td>333</td>
    </tr>
    <tr>
      <th>Quantity</th>
      <td>138</td>
    </tr>
    <tr>
      <th>Price Per Unit</th>
      <td>179</td>
    </tr>
    <tr>
      <th>Total Spent</th>
      <td>173</td>
    </tr>
    <tr>
      <th>Payment Method</th>
      <td>2579</td>
    </tr>
    <tr>
      <th>Location</th>
      <td>3265</td>
    </tr>
    <tr>
      <th>Transaction Date</th>
      <td>159</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>
```
:::
:::

::: {.cell .code execution_count="7" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":13,\"status\":\"ok\",\"timestamp\":1771825732749,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="wLExnwuGCtmd" outputId="176ecaf7-2978-477c-a95d-f6141928e121"}
``` python
df.shape
```

::: {.output .execute_result execution_count="7"}
    (4550, 8)
:::
:::

::: {.cell .code execution_count="8" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":192,\"status\":\"ok\",\"timestamp\":1771825750058,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="AfhfRlC1Cys1" outputId="687069ed-d7bd-45e1-d3f1-fe37fad72cfc"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Index: 4550 entries, 0 to 9999
    Data columns (total 8 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   Transaction ID    4550 non-null   object
     1   Item              4550 non-null   object
     2   Quantity          4550 non-null   object
     3   Price Per Unit    4550 non-null   object
     4   Total Spent       4550 non-null   object
     5   Payment Method    4550 non-null   object
     6   Location          4550 non-null   object
     7   Transaction Date  4550 non-null   object
    dtypes: object(8)
    memory usage: 319.9+ KB
:::
:::

::: {.cell .code execution_count="10" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":20,\"status\":\"ok\",\"timestamp\":1771825812266,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="Ohg9HT8wCgr1" outputId="f9093d13-9888-4fd0-c246-047b6a726ddd"}
``` python
drop_dta=df.dropna(inplace=True)
print(drop_dta)
```

::: {.output .stream .stdout}
    None
:::
:::

::: {.cell .code execution_count="14" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":335}" executionInfo="{\"elapsed\":46,\"status\":\"ok\",\"timestamp\":1771826228689,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="WCwdfbgQEpsk" outputId="f92fe441-8e07-41bf-ee03-ece61d5fb010"}
``` python
pd.isnull(df).sum()
```

::: {.output .execute_result execution_count="14"}
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
      <th>Transaction ID</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Item</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Quantity</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Price Per Unit</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Total Spent</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Payment Method</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Location</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Transaction Date</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>
```
:::
:::
