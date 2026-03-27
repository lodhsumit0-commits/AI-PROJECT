---
jupyter:
  colab:
    authorship_tag: ABX9TyMz7WOpMA2jxft/iSGATuxs
    mount_file_id: 13astrtwqUw8wIkuTym27ikZ49Ph93d7Q
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="53" executionInfo="{\"elapsed\":12,\"status\":\"ok\",\"timestamp\":1770184475750,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="xCkvAwyMITPk"}
``` python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import seaborn as sns
```
:::

::: {.cell .code id="kfXIy89ZJkxm"}
``` python
# Load data
df = pd.read_csv('/content/Car_Purchasing_Data.xls')
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":565}" executionInfo="{\"elapsed\":279,\"status\":\"ok\",\"timestamp\":1770184306207,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="f71IbdSWNM4m" outputId="4ddf505e-8240-4333-efb3-dec1eff6c35a"}
``` python
df
```

::: {.output .execute_result execution_count="4"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 500,\n  \"fields\": [\n    {\n      \"column\": \"Customer Name\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 498,\n        \"samples\": [\n          \"Neville\",\n          \"Matthew Colon\",\n          \"Emerald U. Hanson\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Customer e-mail\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 500,\n        \"samples\": [\n          \"consequat.auctor@lacuspede.co.uk\",\n          \"facilisis@Nullainterdum.edu\",\n          \"pellentesque.a.facilisis@nonlacinia.co.uk\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Country\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 1,\n        \"samples\": [\n          \"USA\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Gender\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 7,\n        \"min\": 20,\n        \"max\": 70,\n        \"num_unique_values\": 43,\n        \"samples\": [\n          65\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Annual Salary\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 11703.378227774132,\n        \"min\": 20000.0,\n        \"max\": 100000.0,\n        \"num_unique_values\": 500,\n        \"samples\": [\n          74420.10254\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Credit Card Debt\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3489.1879728382028,\n        \"min\": 100.0,\n        \"max\": 20000.0,\n        \"num_unique_values\": 500,\n        \"samples\": [\n          10274.13558\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Net Worth\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 173536.75634000328,\n        \"min\": 20000.0,\n        \"max\": 1000000.0,\n        \"num_unique_values\": 500,\n        \"samples\": [\n          551344.3365\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Car Purchase Amount\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 10773.178744235345,\n        \"min\": 9000.0,\n        \"max\": 80000.0,\n        \"num_unique_values\": 500,\n        \"samples\": [\n          46082.80993\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":71,\"status\":\"ok\",\"timestamp\":1770183449845,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="SPCKWgZkJp1t" outputId="607841f3-de7d-4c73-be2a-6ef1bfe8225a"}
``` python
df.columns
```

::: {.output .execute_result execution_count="23"}
    Index(['Customer Name', 'Customer e-mail', 'Country', 'Gender', 'Age',
           'Annual Salary', 'Credit Card Debt', 'Net Worth',
           'Car Purchase Amount'],
          dtype='object')
:::
:::

::: {.cell .code id="22PiS9ZRKLIP"}
``` python
df=("Annual Salary")
df
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":348}" executionInfo="{\"elapsed\":589,\"status\":\"ok\",\"timestamp\":1770184479787,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="72xAaA_WNogp" outputId="efec18c8-6787-44f0-d398-4f8394bc6d87"}
``` python
df.head(5)
```

::: {.output .execute_result execution_count="6"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 500,\n  \"fields\": [\n    {\n      \"column\": \"Customer Name\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 498,\n        \"samples\": [\n          \"Neville\",\n          \"Matthew Colon\",\n          \"Emerald U. Hanson\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Customer e-mail\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 500,\n        \"samples\": [\n          \"consequat.auctor@lacuspede.co.uk\",\n          \"facilisis@Nullainterdum.edu\",\n          \"pellentesque.a.facilisis@nonlacinia.co.uk\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Country\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 1,\n        \"samples\": [\n          \"USA\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Gender\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 7,\n        \"min\": 20,\n        \"max\": 70,\n        \"num_unique_values\": 43,\n        \"samples\": [\n          65\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Annual Salary\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 11703.378227774132,\n        \"min\": 20000.0,\n        \"max\": 100000.0,\n        \"num_unique_values\": 500,\n        \"samples\": [\n          74420.10254\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Credit Card Debt\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3489.1879728382028,\n        \"min\": 100.0,\n        \"max\": 20000.0,\n        \"num_unique_values\": 500,\n        \"samples\": [\n          10274.13558\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Net Worth\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 173536.75634000328,\n        \"min\": 20000.0,\n        \"max\": 1000000.0,\n        \"num_unique_values\": 500,\n        \"samples\": [\n          551344.3365\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Car Purchase Amount\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 10773.178744235345,\n        \"min\": 9000.0,\n        \"max\": 80000.0,\n        \"num_unique_values\": 500,\n        \"samples\": [\n          46082.80993\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":404,\"status\":\"ok\",\"timestamp\":1770184923086,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="PNPJ-zZKOmPO" outputId="7bfa70de-9da4-49f9-b867-f43415acddf8"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 9 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   Customer Name        500 non-null    object 
     1   Customer e-mail      500 non-null    object 
     2   Country              500 non-null    object 
     3   Gender               500 non-null    int64  
     4   Age                  500 non-null    int64  
     5   Annual Salary        500 non-null    float64
     6   Credit Card Debt     500 non-null    float64
     7   Net Worth            500 non-null    float64
     8   Car Purchase Amount  500 non-null    float64
    dtypes: float64(4), int64(2), object(3)
    memory usage: 35.3+ KB
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":366}" executionInfo="{\"elapsed\":20,\"status\":\"ok\",\"timestamp\":1770184973374,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="Nr1CjdlPPds1" outputId="3164fde8-8622-40b0-af39-7bf752757097"}
``` python
df.isnull().sum()
```

::: {.output .execute_result execution_count="12"}
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
      <th>Customer Name</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Customer e-mail</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Country</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Annual Salary</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Credit Card Debt</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Net Worth</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Car Purchase Amount</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>
```
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":300}" executionInfo="{\"elapsed\":55,\"status\":\"ok\",\"timestamp\":1770185116034,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="e_wzjXQ1P2Lm" outputId="218dbe73-60ef-427f-87e9-e8167398b303"}
``` python
df.describe()
```

::: {.output .execute_result execution_count="14"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 8,\n  \"fields\": [\n    {\n      \"column\": \"Gender\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 176.5748291161384,\n        \"min\": 0.0,\n        \"max\": 500.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.506,\n          1.0,\n          0.5004647139007077\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 163.5863319986333,\n        \"min\": 7.990338855772022,\n        \"max\": 500.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          46.224,\n          46.0,\n          500.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Annual Salary\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 33833.992685238365,\n        \"min\": 500.0,\n        \"max\": 100000.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          62127.23960756,\n          62915.497035,\n          500.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Credit Card Debt\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 6577.689838790608,\n        \"min\": 100.0,\n        \"max\": 20000.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          9607.645048629198,\n          9655.035568,\n          500.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Net Worth\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 325405.92166202975,\n        \"min\": 500.0,\n        \"max\": 1000000.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          431475.71362506005,\n          426750.12065,\n          500.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Car Purchase Amount\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 26495.91248523194,\n        \"min\": 500.0,\n        \"max\": 80000.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          44209.79921842,\n          43997.78339,\n          500.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":449}" executionInfo="{\"elapsed\":134,\"status\":\"ok\",\"timestamp\":1770182781716,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="6mT3LSlZQnGg" outputId="4c8cb716-e64d-4694-861a-e6000fb85d41"}
``` python
ans=sns.countplot(x='Gender',data= df)
```

::: {.output .display_data}
![](vertopal_aa4be45707df4da4bdfecfb287ccc354/e5743e38cf254206a2d2b7c064128f346c8b6e2b.png)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":472}" executionInfo="{\"elapsed\":774,\"status\":\"ok\",\"timestamp\":1770183783638,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="GnNjnm8LHcsv" outputId="b11c3f4c-7a9c-40ab-cbab-f86901357562"}
``` python
import matplotlib.pyplot as plt

sns.scatterplot(x='Annual Salary', y='Car Purchase Amount', color='DarkGreen', label='Annual Salary vs Car Purchase Amount', data=df)
plt.xlabel('Annual Salary')
plt.ylabel('Car Purchase Amount')
plt.title('Annual Salary Vs Car Purchase Amount')
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_aa4be45707df4da4bdfecfb287ccc354/acb9c0805ee33a4688f4f4fa7027ff44d466c647.png)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":236,\"status\":\"ok\",\"timestamp\":1770185996618,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="j7ks2Dw8MsZU" outputId="97c1b79a-0d16-486f-9bf5-e8e1800874a9"}
``` python
df['Annual Salary'].max()
```

::: {.output .execute_result execution_count="20"}
    100000.0
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":177,\"status\":\"ok\",\"timestamp\":1770186061710,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="c9fa8NB9T1KN" outputId="45a84395-6b84-4cbf-fef4-b8c84b270016"}
``` python
df['Annual Salary'].min()
```

::: {.output .execute_result execution_count="21"}
    20000.0
:::
:::

::: {.cell .code execution_count="49" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":41,\"status\":\"ok\",\"timestamp\":1770182635050,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="CInnn2TqTqrJ" outputId="44fabe98-9ee7-4204-da44-fc02084d1e7a"}
``` python
salary=np.array([[60000]])
predicted_ammount=model.predict(salary)
print("Maximum car Purchasing Amount:",predicted_ammount[0])
```

::: {.output .stream .stdout}
    Maximum car Purchasing Amount: 42999.92476763238
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
:::
:::

::: {.cell .code execution_count="50" executionInfo="{\"elapsed\":9,\"status\":\"ok\",\"timestamp\":1770184225685,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="6syeENkuM2qc"}
``` python
x=df[['Annual Salary']]
y=df['Car Purchase Amount']
```
:::

::: {.cell .code execution_count="54" executionInfo="{\"elapsed\":200,\"status\":\"ok\",\"timestamp\":1770184497035,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="XdSFLDmGM75-"}
``` python
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
```
:::

::: {.cell .code execution_count="55" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":80}" executionInfo="{\"elapsed\":677,\"status\":\"ok\",\"timestamp\":1770184527256,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="VT_kV7lmNboO" outputId="897cdb94-9f42-4aaf-c7ef-8acdb3d9ddc1"}
``` python
model=LinearRegression()
model.fit(x_train,y_train)
```

::: {.output .execute_result execution_count="55"}
```{=html}
<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LinearRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div>
```
:::
:::
