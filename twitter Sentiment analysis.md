---
jupyter:
  colab:
  kernelspec:
    display_name: myenv
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.13.11
  nbformat: 4
  nbformat_minor: 5
---

::: {#c28c872e .cell .code execution_count="2" executionInfo="{\"elapsed\":120,\"status\":\"ok\",\"timestamp\":1771406355643,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="c28c872e"}
``` python
import pandas as pd
import numpy as np
df=pd.read_csv('Twitter_hate.csv')
```
:::

::: {#88ca4275 .cell .code execution_count="3" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":206}" executionInfo="{\"elapsed\":79,\"status\":\"ok\",\"timestamp\":1771406355726,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="88ca4275" outputId="5a9a3057-c0b7-42b5-b75e-90a9012054d2"}
``` python
df.head()
```

::: {.output .execute_result execution_count="3"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 31962,\n  \"fields\": [\n    {\n      \"column\": \"id\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9226,\n        \"min\": 1,\n        \"max\": 31962,\n        \"num_unique_values\": 31962,\n        \"samples\": [\n          12228,\n          14710,\n          19320\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"label\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"tweet\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 29530,\n        \"samples\": [\n          \"\\\"even if you like #windows 10, you should be   at #microsoft\\\" #computer #spam #spyware #unreliable #os #ms #win \",\n          \"the countdown begins! 10 hours till the musical! \\u00f0\\u009f\\u0098\\u008a\\u00f0\\u009f\\u0098\\u0084\\u00f0\\u009f\\u0098\\u0085 #itsthefinalcountdown #youngcarers #rctcbc #rctcouncil #rct #carersweek   \\u00f0\\u009f\\u0098\\u0081\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {#0395372a .cell .code execution_count="4" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":7,\"status\":\"ok\",\"timestamp\":1771406355738,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="0395372a" outputId="1bcc3bce-ad50-4aa4-cac7-fc0b1c2ea346"}
``` python
df.shape
```

::: {.output .execute_result execution_count="4"}
    (31962, 3)
:::
:::

::: {#13a4f773 .cell .code execution_count="5" executionInfo="{\"elapsed\":18,\"status\":\"ok\",\"timestamp\":1771406355757,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="13a4f773"}
``` python
hatred_comment= df[df['label']==1]
```
:::

::: {#0bdcf94d .cell .code execution_count="6" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":398}" executionInfo="{\"elapsed\":64,\"status\":\"ok\",\"timestamp\":1771406355823,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="0bdcf94d" outputId="20f59a1d-15d0-43b7-8fec-d11c7679f6aa"}
``` python
hatred_comment['tweet'].tail(10)
```

::: {.output .execute_result execution_count="6"}
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
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31912</th>
      <td>i couldn't end #2016 without mentioning #trump...</td>
    </tr>
    <tr>
      <th>31926</th>
      <td>a follow up from the gentlemen who were kicked...</td>
    </tr>
    <tr>
      <th>31929</th>
      <td>did  keep #colinpowell and #condoleezzarice fr...</td>
    </tr>
    <tr>
      <th>31930</th>
      <td>@user #feminismiscancer #feminismisterrorism #...</td>
    </tr>
    <tr>
      <th>31933</th>
      <td>@user judd is a  &amp;amp; #homophobic #freemilo #...</td>
    </tr>
    <tr>
      <th>31934</th>
      <td>lady banned from kentucky mall. @user  #jcpenn...</td>
    </tr>
    <tr>
      <th>31946</th>
      <td>@user omfg i'm offended! i'm a  mailbox and i'...</td>
    </tr>
    <tr>
      <th>31947</th>
      <td>@user @user you don't have the balls to hashta...</td>
    </tr>
    <tr>
      <th>31948</th>
      <td>makes you ask yourself, who am i? then am i a...</td>
    </tr>
    <tr>
      <th>31960</th>
      <td>@user #sikh #temple vandalised in in #calgary,...</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> object</label>
```
:::
:::

::: {#3ddb8ccd .cell .code execution_count="7" executionInfo="{\"elapsed\":1,\"status\":\"ok\",\"timestamp\":1771406355829,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="3ddb8ccd"}
``` python
import re
def clean_text(text):
    text=re.sub(r'[<>!@^;#]','',text)
    text=re.sub(r'[&"?/\|+-_$]','',text)
    return text
```
:::

::: {#edcb02d7 .cell .code execution_count="9" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":178}" executionInfo="{\"elapsed\":12,\"status\":\"ok\",\"timestamp\":1771406355992,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="edcb02d7" outputId="ef283e03-4962-409b-ec73-a7da7bb82c87"}
``` python
df.isnull().sum()
```

::: {.output .execute_result execution_count="9"}
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
      <th>label</th>
      <td>0</td>
    </tr>
    <tr>
      <th>tweet</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>
```
:::
:::

::: {#fff94af8 .cell .code execution_count="10" executionInfo="{\"elapsed\":22,\"status\":\"ok\",\"timestamp\":1771406356019,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="fff94af8"}
``` python
abbreviations = {
    "A3": "Anytime, Anywhere, Anyplace",
    "ADIH": "Another Day In Hell",
    "AFK": "Away From Keyboard",
    "AFAIK": "As Far As I Know",
    "ASAP": "As Soon As Possible",
    "ASL": "Age, Sex, Location",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "BAE": "Before Anyone Else",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRUH": "Bro",
    "BRT": "Be Right There",
    "BSAAW": "Big Smile And A Wink",
    "BTW": "By The Way",
    "BWL": "Bursting With Laughter",
    "CSL": "Can’t Stop Laughing",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "DM": "Direct Message",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FIMH": "Forever In My Heart",
    "FOMO": "Fear Of Missing Out",
    "FR": "For Real",
    "FWIW": "For What It's Worth",
    "FYP": "For You Page",
    "FYI": "For Your Information",
    "G9": "Genius",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GMTA": "Great Minds Think Alike",
    "GN": "Good Night",
    "GOAT": "Greatest Of All Time",
    "GR8": "Great!",
    "HBD": "Happy Birthday",
    "IC": "I See",
    "ICQ": "I Seek You",
    "IDC": "I Don’t Care",
    "IDK": "I Don't Know",
    "IFYP": "I Feel Your Pain",
    "ILU": "I Love You",
    "ILY": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMU": "I Miss You",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "IYKYK": "If You Know, You Know",
    "JK": "Just Kidding",
    "KISS": "Keep It Simple, Stupid",
    "L": "Loss",
    "L8R": "Later",
    "LDR": "Long Distance Relationship",
    "LMK": "Let Me Know",
    "LMAO": "Laughing My A** Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "M8": "Mate",
    "MFW": "My Face When",
    "MID": "Mediocre",
    "MRW": "My Reaction When",
    "MTE": "My Thoughts Exactly",
    "NVM": "Never Mind",
    "NRN": "No Reply Necessary",
    "NPC": "Non-Player Character",
    "OIC": "Oh I See",
    "OP": "Overpowered",
    "PITA": "Pain In The A**",
    "POV": "Point Of View",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A** Off",
    "RN": "Right Now",
    "SK8": "Skate",
    "STATS": "Your Sex And Age",
    "SUS": "Suspicious",
    "TBH": "To Be Honest",
    "TFW": "That Feeling When",
    "THX": "Thank You",
    "TIME": "Tears In My Eyes",
    "TLDR": "Too Long, Didn’t Read",
    "TNTL": "Trying Not To Laugh",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "W": "Win",
    "W8": "Wait...",
    "WB": "Welcome Back",
    "WTF": "What The F**k",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "WYD": "What You Doing?",
    "WYWH": "Wish You Were Here",
    "ZZZ": "Sleeping, Bored, Tired",
    "AAA": "Anywhere, Anyplace, Anything"


}
```
:::

::: {#2f22bd5b .cell .code execution_count="11" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":6,\"status\":\"ok\",\"timestamp\":1771406356026,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="2f22bd5b" outputId="276e783f-c5f0-4267-a924-92256f35d3ad"}
``` python
print(abbreviations['AAA'])
```

::: {.output .stream .stdout}
    Anywhere, Anyplace, Anything
:::
:::

::: {#c9028be4 .cell .code execution_count="12" executionInfo="{\"elapsed\":1,\"status\":\"ok\",\"timestamp\":1771406356030,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="c9028be4"}
``` python
def chat_conversation(text):
     new_text=[]
     for w in text. split():# w is a variable you can put your name on it
          if w.upper()in abbreviations:
             new_text.append(abbreviations[w.upper()])#if the words in in the abbrevaiation it will come in if
          else:
            new_text.append(w)#if the word are not there in abbreviation iot will come to else
     return" ".join(new_text)# merge the word and "" should put space
```
:::

::: {#f9a26b25 .cell .code execution_count="13" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":35}" executionInfo="{\"elapsed\":18,\"status\":\"ok\",\"timestamp\":1771406356050,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="f9a26b25" outputId="b255ce1e-1585-47ff-ec5b-96c00e08c763"}
``` python
chat_conversation("WB to Canada")
```

::: {.output .execute_result execution_count="13"}
``` json
{"type":"string"}
```
:::
:::

::: {#075c6c5e .cell .markdown id="075c6c5e"}
Spelling Check
:::

::: {#00c9682f .cell .code execution_count="14" executionInfo="{\"elapsed\":2658,\"status\":\"ok\",\"timestamp\":1771406358710,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="00c9682f"}
``` python
from textblob import TextBlob
```
:::

::: {#cb72ce05 .cell .code execution_count="15" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":35}" executionInfo="{\"elapsed\":654,\"status\":\"ok\",\"timestamp\":1771406359366,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="cb72ce05" outputId="48a0fe48-225c-423f-efda-3e4137da27aa"}
``` python
incorrext_text = '''Ths is a smple paragraf wth severl spelin correctn
sistem and see how it handels misstakes in a realistc sebtence'''

text_Blb = TextBlob(incorrext_text)

text_Blb.correct().string
```

::: {.output .execute_result execution_count="15"}
``` json
{"type":"string"}
```
:::
:::

::: {#1b84b9c1 .cell .code execution_count="16" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":35}" executionInfo="{\"elapsed\":397,\"status\":\"ok\",\"timestamp\":1771406359766,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="1b84b9c1" outputId="d9e1c027-0002-43e1-9241-fb037e39638f"}
``` python
wrng_text = '''the snwfll is vey thck durng Decmbr and Janry'''

text_Blb = TextBlob(wrng_text)

text_Blb.correct().string
```

::: {.output .execute_result execution_count="16"}
``` json
{"type":"string"}
```
:::
:::

::: {#d23a6428 .cell .markdown id="d23a6428"}
Remove stopword
:::

::: {#f4129e1e .cell .markdown id="f4129e1e"}
The movie is good
:::

::: {#48a5140d .cell .markdown id="48a5140d"}
after removing stopword

movie good
:::

::: {#7c300af3 .cell .markdown id="7c300af3"}
-the -is -am -are
:::

::: {#cce876eb .cell .code execution_count="17" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":214,\"status\":\"ok\",\"timestamp\":1771406359982,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="cce876eb" outputId="6c34dc5d-ef3d-4635-d73b-777e20af88a2"}
``` python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
```

::: {.output .stream .stderr}
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
:::
:::

::: {#fc01c53f .cell .code execution_count="18" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":26,\"status\":\"ok\",\"timestamp\":1771406359983,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="fc01c53f" outputId="b3b533ac-872e-40a9-8f79-ef3cd4055203"}
``` python
stops = set(stopwords.words('english'))
print(stops)
```

::: {.output .stream .stdout}
    {'couldn', 'up', 'what', 'being', 'haven', "i'll", 'me', 'will', 'him', "it'd", "you've", 'you', 'where', 'am', 'any', 'and', 'ma', 'them', 'these', 'out', 'didn', 'won', 'at', 'itself', 'under', 'it', 'nor', 'to', 'her', "we'll", 'between', 'this', 'so', "needn't", 'once', "she's", 'because', 'while', 'your', 'ours', 'but', 'some', "we've", "won't", 'all', "they're", 'himself', 'such', 'why', 'm', 'as', 'when', "haven't", 'in', 't', 'those', 'mightn', 'further', 'off', 'during', 'if', 'now', "should've", 'are', "she'd", 'having', 'not', "doesn't", 'our', 'weren', 'from', 'i', 'same', 'other', 'few', 'is', "you're", "weren't", 'again', 'been', 're', "they've", "isn't", "they'll", "she'll", 'theirs', 'who', 'no', 'whom', "wasn't", 'be', 'does', "aren't", 'more', 'then', 'by', 'have', 'above', 'there', 'for', "i've", "they'd", 'each', 'most', 'into', "hasn't", 'how', 'only', 'his', 'too', 'd', "we'd", 'which', 'both', 'were', 'or', 'through', 'has', 've', 'he', 'on', 'own', 'do', "he'd", 'should', 'with', 'shan', 'over', 'just', "didn't", 'here', 'its', 'yours', 'hers', 'don', 's', 'wouldn', "shan't", 'hasn', "shouldn't", 'isn', 'than', 'was', 'aren', "it'll", 'yourself', 'my', 'needn', 'can', 'themselves', 'the', 'mustn', 'before', 'down', 'doesn', "he's", 'of', 'she', 'until', "that'll", 'an', 'below', 'we', 'against', "it's", 'o', "wouldn't", 'y', "mightn't", 'll', "you'll", 'doing', 'wasn', 'yourselves', 'a', 'had', 'after', 'their', "he'll", "i'd", "hadn't", "don't", 'shouldn', 'ourselves', 'very', 'about', 'that', "couldn't", 'ain', 'hadn', "mustn't", "i'm", 'myself', "we're", 'herself', 'they', "you'd", 'did'}
:::
:::

::: {#4bb9fc89 .cell .code execution_count="19" executionInfo="{\"elapsed\":1,\"status\":\"ok\",\"timestamp\":1771406359985,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="4bb9fc89"}
``` python
def remove_stopword(text):
    new_text = []

    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)
```
:::

::: {#36a73d18 .cell .markdown id="36a73d18"}
new
:::

::: {#5b9469b7 .cell .code execution_count="20" executionInfo="{\"elapsed\":22,\"status\":\"ok\",\"timestamp\":1771406360009,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="5b9469b7"}
``` python
def remove_stopword2(text):
    new_text = []

    for word in text.split():
        if word.lower() not in stopwords.words('english'):
            new_text.append(word)

    # Join the list into a single string separated by spaces
    return " ".join(new_text)
```
:::

::: {#800a8346 .cell .code execution_count="21" executionInfo="{\"elapsed\":2,\"status\":\"ok\",\"timestamp\":1771406360012,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="800a8346"}
``` python
full_text = 'join the class now'
```
:::

::: {#6fea1869 .cell .code execution_count="22" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":35}" executionInfo="{\"elapsed\":16,\"status\":\"ok\",\"timestamp\":1771406360029,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="6fea1869" outputId="b9755c43-3d02-452b-fd8b-c42b64101930"}
``` python
remove_stopword(full_text)
```

::: {.output .execute_result execution_count="22"}
``` json
{"type":"string"}
```
:::
:::

::: {#Osijw_bWDEn8 .cell .code id="Osijw_bWDEn8"}
``` python
```
:::
