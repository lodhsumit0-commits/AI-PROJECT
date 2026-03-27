---
jupyter:
  colab:
    authorship_tag: ABX9TyN36ru17ghgNTPET/NpmW2B
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="25" executionInfo="{\"elapsed\":13,\"status\":\"ok\",\"timestamp\":1771310366819,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="wyKCxkvj3F0c"}
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
```
:::

::: {.cell .code execution_count="26" executionInfo="{\"elapsed\":14,\"status\":\"ok\",\"timestamp\":1771310367022,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="QD1GQR513YRT"}
``` python
import pandas as pd
df=pd.read_csv('/content/multilinear_regression_dataset_marketing.csv')
```
:::

::: {.cell .code execution_count="27" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":424}" executionInfo="{\"elapsed\":9,\"status\":\"ok\",\"timestamp\":1771310367034,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="qVEt5hRt4pos" outputId="15835f85-5a2e-4e93-a2f6-8f43eea898a6"}
``` python
df
```

::: {.output .execute_result execution_count="27"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 250,\n  \"fields\": [\n    {\n      \"column\": \"ad_spend\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 8.950686442455947,\n        \"min\": 1.0,\n        \"max\": 47.64,\n        \"num_unique_values\": 235,\n        \"samples\": [\n          6.03,\n          15.29,\n          10.34\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"email_sends\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 18.096979463513,\n        \"min\": 5.0,\n        \"max\": 95.5,\n        \"num_unique_values\": 210,\n        \"samples\": [\n          36.9,\n          17.6,\n          41.3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"site_visits\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 61.34380195580947,\n        \"min\": 30.0,\n        \"max\": 412.9,\n        \"num_unique_values\": 231,\n        \"samples\": [\n          215.7,\n          196.9,\n          182.3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"revenue\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 338.7363672271083,\n        \"min\": 128.77,\n        \"max\": 2498.18,\n        \"num_unique_values\": 248,\n        \"samples\": [\n          458.66,\n          931.48,\n          1420.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code execution_count="28" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":206}" executionInfo="{\"elapsed\":4,\"status\":\"ok\",\"timestamp\":1771310367041,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="xjfhQGKO4zhc" outputId="e40694d9-07dd-4c02-cdd5-8a9cea58a69a"}
``` python
df.head()
```

::: {.output .execute_result execution_count="28"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 250,\n  \"fields\": [\n    {\n      \"column\": \"ad_spend\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 8.950686442455947,\n        \"min\": 1.0,\n        \"max\": 47.64,\n        \"num_unique_values\": 235,\n        \"samples\": [\n          6.03,\n          15.29,\n          10.34\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"email_sends\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 18.096979463513,\n        \"min\": 5.0,\n        \"max\": 95.5,\n        \"num_unique_values\": 210,\n        \"samples\": [\n          36.9,\n          17.6,\n          41.3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"site_visits\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 61.34380195580947,\n        \"min\": 30.0,\n        \"max\": 412.9,\n        \"num_unique_values\": 231,\n        \"samples\": [\n          215.7,\n          196.9,\n          182.3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"revenue\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 338.7363672271083,\n        \"min\": 128.77,\n        \"max\": 2498.18,\n        \"num_unique_values\": 248,\n        \"samples\": [\n          458.66,\n          931.48,\n          1420.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code execution_count="29" executionInfo="{\"elapsed\":3,\"status\":\"ok\",\"timestamp\":1771310367045,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="iwV58a6k44x0"}
``` python
from sklearn import linear_model
model = linear_model.LinearRegression()
```
:::

::: {.cell .code execution_count="30" executionInfo="{\"elapsed\":2,\"status\":\"ok\",\"timestamp\":1771310367049,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="3n8G6zI-GSge"}
``` python
x=df[['ad_spend','email_sends','site_visits']]
y=df['revenue']
```
:::

::: {.cell .code execution_count="31" executionInfo="{\"elapsed\":3,\"status\":\"ok\",\"timestamp\":1771310367053,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="MWpp4pD_G8lf"}
``` python
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
```
:::

::: {.cell .code execution_count="32" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":80}" executionInfo="{\"elapsed\":47,\"status\":\"ok\",\"timestamp\":1771310367102,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="TPFZmqDE5Iyk" outputId="a9b332a3-4b79-42f6-9c2c-373abe92b98f"}
``` python
model.fit(x_train,y_train)
```

::: {.output .execute_result execution_count="32"}
```{=html}
<style>#sk-container-id-2 {
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

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
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

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
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

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
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

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
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

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
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

#sk-container-id-2 a.estimator_doc_link {
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

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LinearRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div>
```
:::
:::

::: {.cell .code execution_count="33" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":8,\"status\":\"ok\",\"timestamp\":1771310367111,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="TH7D26mtI8yH" outputId="11daa883-86be-4c6c-e78d-3a269a428f64"}
``` python
model.score(x_train,y_train)
```

::: {.output .execute_result execution_count="33"}
    0.36626650882034084
:::
:::

::: {.cell .code execution_count="34" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":9,\"status\":\"ok\",\"timestamp\":1771310367118,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="cLAhjVXJIaUn" outputId="60865d8c-7ef2-4675-ceca-575919a99fa5"}
``` python
model.predict([[25.95,64.1,176.7	]])
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
:::

::: {.output .execute_result execution_count="34"}
    array([621.9725041])
:::
:::

::: {.cell .code execution_count="35" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" executionInfo="{\"elapsed\":3074,\"status\":\"ok\",\"timestamp\":1771310370204,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="C1H14GytNa2a" outputId="722d8f24-5986-4bd2-e391-33ef18833633"}
``` python
all_variables=['ad_spend','email_sends','site_visits','revenue']

#create the pair plot
sns.pairplot(df[all_variables], kind='scatter')

#Display the plot
plt.show()
```

::: {.output .display_data}
![](vertopal_259824c599074a63b6fbcaf30d9b0adf/b351dfc9dec6098b3f510e20afedb26f02259524.png)
:::
:::

::: {.cell .code execution_count="36" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":175}" executionInfo="{\"elapsed\":11,\"status\":\"ok\",\"timestamp\":1771310370222,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="0MASMXulQNNb" outputId="d9975a9e-3ffd-4995-b991-24d007bf38b2"}
``` python
df.corr()
```

::: {.output .execute_result execution_count="36"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 4,\n  \"fields\": [\n    {\n      \"column\": \"ad_spend\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.4517134387281339,\n        \"min\": -0.0808086163188266,\n        \"max\": 1.0,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          -0.0808086163188266,\n          0.26447501350020397,\n          1.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"email_sends\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.5091775713371612,\n        \"min\": -0.0808086163188266,\n        \"max\": 1.0,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          1.0,\n          0.06751279210159736,\n          -0.0808086163188266\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"site_visits\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.43416819101991533,\n        \"min\": -0.0198597770222265,\n        \"max\": 1.0,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          -0.0198597770222265,\n          0.6362008562256924,\n          0.3347772807325116\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"revenue\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.41263982970988783,\n        \"min\": 0.06751279210159736,\n        \"max\": 1.0,\n        \"num_unique_values\": 4,\n        \"samples\": [\n          0.06751279210159736,\n          1.0,\n          0.26447501350020397\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::
:::

::: {.cell .code execution_count="37" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":564}" executionInfo="{\"elapsed\":33,\"status\":\"ok\",\"timestamp\":1771310370256,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="1BU2jbJTQuoD" outputId="5004ba9a-696c-4491-e572-59d651bca419"}
``` python
plt.figure(figsize=(10,6))
plt.scatter(x=df['ad_spend'],y=df['revenue'])
plt.xlabel('ad_spend')
plt.ylabel('revenue')
plt.title('ad_spend vs revenue')
plt.grid(True)
plt.plot(x,model.predict(x),"r-",label='Line of best Fit')
plt.show()
```

::: {.output .display_data}
![](vertopal_259824c599074a63b6fbcaf30d9b0adf/72caf2df457c81a1100c8b69755d534799f9326f.png)
:::
:::

::: {.cell .code execution_count="37" executionInfo="{\"elapsed\":2,\"status\":\"ok\",\"timestamp\":1771310370260,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="Pd5hMBgIUchN"}
``` python
```
:::
