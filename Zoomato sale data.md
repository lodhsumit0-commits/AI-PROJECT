---
jupyter:
  colab:
    authorship_tag: ABX9TyM37Qu7EHN79o85Yo7iQiRP
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="17" executionInfo="{\"elapsed\":773,\"status\":\"ok\",\"timestamp\":1771829827101,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="C-PDpCvu-z40"}
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

```
:::

::: {.cell .code execution_count="9" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":608}" executionInfo="{\"elapsed\":6090,\"status\":\"ok\",\"timestamp\":1771829344136,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="od_FYOHFP8w7" outputId="d9e3383a-42ff-41f5-96e2-a113414ee6d8"}
``` python
df=pd.read_csv('zomato.csv',engine='python',on_bad_lines='skip')
df.head()
```

::: {.output .execute_result execution_count="9"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 34203,\n  \"fields\": [\n    {\n      \"column\": \"url\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 34203,\n        \"samples\": [\n          \"https://www.zomato.com/bangalore/gappe-bannerghatta-road-bangalore?context=eyJzZSI6eyJlIjpbIjE4ODQ4NjgyIiwiMTg3MDY5MDEiLCIxODQ2MDA3NiIsIjE4Njc1Njg0IiwxODY3OTI4MywiMTgzNzc5NDEiLCIxODYyMjgwMiIsIjE4NjQyOTMzIiwiMTg1OTU4NDIiLCIxODQ2MDA1NCIsIjE4NjMzODMzIiwiMTg2NTg5ODciLCIxODg5MDQwNSIsIjYwMTYzIiwiMTgyMDI5ODgiLCIxODc4MzU0MiIsIjE4NTMxNzIzIiwiNTMwNDAiLCIxODUzNjI5MSIsIjYwOTc5IiwiMTg4MDEwNTAiLCI1MTQ0NiIsIjUxNDA2Il0sInQiOiJEZWxpdmVyeSBSZXN0YXVyYW50cyBpbiBCYW5uZXJnaGF0dGEgUm9hZCJ9fQ==\",\n          \"https://www.zomato.com/bangalore/gowdaaa-mane-oota-bellandur?context=eyJzZSI6eyJlIjpbIjE4NjMyMTc1IiwiNTMzMDEiLCIxODc2NjMyOCIsIjE4NTcxMjQ3Iiw1NjE2NywiMTg3MTQ0MjQiLCIxODU3MTI4NiIsIjE4NTc3Njk0IiwiMTg1NTUxMzkiLCIxODY2ODczOSIsIjE4NzIzMDAxIiwiMTg3NDMyMjUiLCIxODg3NzUxNyIsIjE4ODkxNjAyIiwiMTg4NTU1MTkiLCIxODI4NzQ5NCIsIjE4ODY2Njg0IiwiMTg1NzA3NDIiLCIxODU3MTM2MyIsIjE4NTc5MTc3IiwiNTUwNTYiLCI2MDQ5OCIsIjE4NzU0OTI2Il0sInQiOiJEaW5lLU91dCBSZXN0YXVyYW50cyBpbiBCZWxsYW5kdXIifX0=\",\n          \"https://www.zomato.com/bangalore/fruit-juice-bar-koramangala-6th-block?context=eyJzZSI6eyJlIjpbIjE4NTk1MDAxIiwiNTk1MTMiLCIxODI4NDE5NiIsIjE4MzMzMTc4IiwxODU5ODE5NCwiMTg0NTA4ODgiLCIxODcyODA4MiIsIjUxNDMyIiwiNTc5NDQiLCI2MTQ2MCIsIjE4MzEwOTU5IiwiMTg3MzAzMTEiLCIxODU3MTQ3NCIsIjU2OTAzIiwiMTgyMTY2ODgiLCIxODMwODc3MSIsIjE4MzUzMTAzIiwiMTgzNjk4MTMiLCIxODc2MTg3NSIsIjE4Nzk1MDAxIiwiMTg4ODgxMDkiLCIxODg5NjY4OCIsIjE4NjY1NjE3IiwiMTg1OTgyMzIiXSwidCI6IkRlbGl2ZXJ5IFJlc3RhdXJhbnRzIGluIEtvcmFtYW5nYWxhIDR0aCBCbG9jayJ9fQ==\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"address\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 9259,\n        \"samples\": [\n          \"128/54/10, First floor, Opposite IIM - B, Dasarapalya, Bannerghatta Road, Bangalore\",\n          \"100G, Opposite Grace Apartments, Hennur Road, Kalyan Nagar, Bangalore\",\n          \"4C/211, 3rd Block, 4th Cross, Harbour Layout, CMR Road, Kalyan Nagar, Bangalore\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"name\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 7275,\n        \"samples\": [\n          \"A 1 Pizza and Food\",\n          \"Spice\",\n          \"What the Food\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"online_order\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"No\",\n          \"Yes\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"book_table\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"No\",\n          \"Yes\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"rate\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 60,\n        \"samples\": [\n          \"4.1/5\",\n          \"4.0/5\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"votes\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 712.2728525962408,\n        \"min\": 0.0,\n        \"max\": 16345.0,\n        \"num_unique_values\": 1816,\n        \"samples\": [\n          2217.0,\n          2229.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"phone\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 11081,\n        \"samples\": [\n          \"+91 8123474755\",\n          \"080 41139926\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"location\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 84,\n        \"samples\": [\n          \"North Bangalore\",\n          \"Banashankari\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"rest_type\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 90,\n        \"samples\": [\n          \"Quick Bites, Sweet Shop\",\n          \"Bakery, Quick Bites\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"dish_liked\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 4057,\n        \"samples\": [\n          \"Chocolate Mousse, Pasta, Chocolate Cake, Lemon Tarts, Veg Burger\",\n          \"Salads, Chicken Sandwich, Sandwiches, Cheesecake, Spaghetti Aglio Olio, Arugula Salad, Cheese Chilli Toast\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"cuisines\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2359,\n        \"samples\": [\n          \"Bakery, Mithai\",\n          \"Rajasthani, North Indian\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"approx_cost(for two people)\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 63,\n        \"samples\": [\n          \"1,050\",\n          \"560\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"reviews_list\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 14978,\n        \"samples\": [\n          \"[('Rated 4.0', 'RATED\\\\n  Such a clean and nice place. Small establishment and in decent location. I wish them all the luck to succeed.\\\\nJust like to know how you want to juice made'), ('Rated 4.0', 'RATED\\\\n  The search for quality and taste in sweets ends here...\\\\nThey have s choice of around 30 varieties here starting with the humble laddu to the kaju burfi ...\\\\nPrice varies from Rs 350/- to Rs 700/-\\\\nTried their specialty-THE GHEE MYSOREPAK which was just too good... Yummmmie\\\\nParking available\\\\nService is excellent')]\",\n          \"[('Rated 3.0', \\\"RATED\\\\n  This place is so hard to find even though it's right on the main road. We only knew it exists because of zomato. The alcohol is a bit expensive. The ambience is great. No doubt.\\\"), ('Rated 4.0', 'RATED\\\\n  I went here on a weekday evening and I was thrilled to see the ambience. The only down point of going to this place is that it ends at 11 PM on weekdays. It is on the roof top with great ambience and good decor. Great place to have a conversation over a couple of drinks. The music is not too loud and there is usually less crowd.\\\\nService -4/5\\\\nAmbience-4/5'), ('Rated 5.0', 'RATED\\\\n  Awesome ambiance ..staff was also very courteous and could handle large crowd easily..recommended to visit here by having reservation before ,and..service was prompt as well..'), ('Rated 4.0', \\\"RATED\\\\n  It's such a lovely place. Great location and pretty view. But the scene is super dead and it was empty on a Saturday night. So, if you forgot to make reservations and need a nice place to chill at, Quench your thirst here :P\\\"), ('Rated 1.0', 'RATED\\\\n  Pathetic service and most of the menu is not available\\\\nThe view is good but the place was entirely empty on a Friday night, most probably because of the lack of good food and drinks'), ('Rated 1.0', \\\"RATED\\\\n  The place looks amazing but the manager and the staffs doesn't know how to behave professionally. The Manager was very rude. The food they offer is not fresh. Giving one star only for the good-looking place.\\\"), ('Rated 1.0', \\\"RATED\\\\n  FELT CHEATED HERE. This was my worst experience in Bangalore. The employees weren't aware of the Zomato gold offer, the service was bad. I'm so disappointed with the hospitality of the employees. Momos were really bad, it wasn't cooked well.\\\\nThe MANAGER was very unprofessional. They knew about the Zomato Gold membership and still pretended to not know anything about it. Gave some senseless reason for their mistakes. Denied Zomato Gold offer, weren't ready to accept two of my Zomato gold offers. They asked me to pay the full amount. This was really disappointing.\\\"), ('Rated 4.0', 'RATED\\\\n  When we hear the word \\\"Quench\\\".. it brings the thirst out of ourselves. I tried the cocktail called New York which was a combination of Scotch Cranberry Juice and few others and tried a few starters and also The Long Island...Adding to this the Roof Top Ambience provides a perfect setting for an Anniversary celebration for couples especially.\\\\n\\\\nOne place to be tried out few times...'), ('Rated 4.0', \\\"RATED\\\\n  Located on the rooftop of Ramada Encore, it's a nice place to chill out.. The LIIT was moderately strong and food was very well made. Staff was courteous too. The only drawback was there was no Budweiser available on that day. Overall a good experience.\\\"), ('Rated 4.0', \\\"RATED\\\\n  This place took me by surprise and we loved the ambiance as soon as we walked in.\\\\nA different entrance and a very attractive outlay.\\\\nSeating was great, but the sofas we're slightly hard, and it could be improved.\\\\nService was good.\\\\nThe only drawback was the food, which is an integral part but I'm still giving 4 because I was so impressed with the set up of the place and the view from the rooftop! Great job but lot of room to improve on the food\\\"), ('Rated 5.0', \\\"RATED\\\\n  Enjoyed in quench bar , the place is crowded on weekends and weekdays it's fine , the staff is less I guess to mange the bar but I was taken care well, the selections of the bar was good and my meeting with the clients was very fruitful. The food is nice.\\\"), ('Rated 5.0', \\\"RATED\\\\n  A lovely rooftop venue to hang out with friends and grab a drink. An outdoor venue at it's best overlooking the koramangala ringroad. A place to be at anytime of the week.\\\"), ('Rated 5.0', 'RATED\\\\n  Ambience is really amazing. Though food is good, Service is bit slow. The open roof on a clear night sky would be a great experience. \\\\nWill visit again.'), ('Rated 3.5', \\\"RATED\\\\n  It's a roof top bar, so a good place to hangout with friends and have a beer. Nice ambience. Decent food.\\\\nNachos, chicken wings and fresh lime are worth a try.\\\"), ('Rated 4.5', 'RATED\\\\n  Enjoyed last Thursday evening thoroughly! The rooftop ambience was superb and we were lucky with the weather that evening. Some great IPL promo offers going on that ensured that the 2 odd hours we spent there did not burn a hole in our pockets!'), ('Rated 5.0', 'RATED\\\\n  Lovely place.... roof tops are fun & this place also has a nice view with lovely sitting & ambience.\\\\nComing to food & drinks.... Yummy is the word... They do complete justice to ur pocket as the quantity is great.... Overall would recommend this place to people looking for a fun evening with friends & family.'), ('Rated 4.5', 'RATED\\\\n  Took a chance and tried this place .... Was well above what I expected ... Good food ... Professional and well mannered staff !! Great cosy effect over all ..... Would definitely recommend it ......'), ('Rated 5.0', 'RATED\\\\n  Excellent food, Amazing staff ,wide varieties of Cocktails and Good on pockets too:). The best place to relax around Domlur with a view at the rooftop...Thumbs Up!'), ('Rated 5.0', 'RATED\\\\n  Good ambience and excellent service bar in domlur area . Right place to hang with friends,.will be difinalety back again. Wish if they would have d j live, too.'), ('Rated 5.0', \\\"RATED\\\\n  Try the Wingless Flight. Absolutely awesome. And the food is super as well. It's a rooftop place and hence if there's a good wind flowing it'll make your day. The ambience is good and the staff is very polite and courteous. The location is very convenient as well. Worth a visit.\\\"), ('Rated 5.0', 'RATED\\\\n  Lovely place.... roof tops are fun & this place also has a nice view with lovely sitting & ambience.\\\\nComing to food & drinks.... Yummy is the word... They do complete justice to ur pocket as the quantity is great.... Overall would recommend this place to people looking for a fun evening with friends & family.'), ('Rated 4.5', 'RATED\\\\n  Took a chance and tried this place .... Was well above what I expected ... Good food ... Professional and well mannered staff !! Great cosy effect over all ..... Would definitely recommend it ......'), ('Rated 5.0', 'RATED\\\\n  Excellent food, Amazing staff ,wide varieties of Cocktails and Good on pockets too:). The best place to relax around Domlur with a view at the rooftop...Thumbs Up!'), ('Rated 5.0', 'RATED\\\\n  Good ambience and excellent service bar in domlur area . Right place to hang with friends,.will be difinalety back again. Wish if they would have d j live, too.'), ('Rated 5.0', \\\"RATED\\\\n  Try the Wingless Flight. Absolutely awesome. And the food is super as well. It's a rooftop place and hence if there's a good wind flowing it'll make your day. The ambience is good and the staff is very polite and courteous. The location is very convenient as well. Worth a visit.\\\"), ('Rated 5.0', 'RATED\\\\n  Lovely place.... roof tops are fun & this place also has a nice view with lovely sitting & ambience.\\\\nComing to food & drinks.... Yummy is the word... They do complete justice to ur pocket as the quantity is great.... Overall would recommend this place to people looking for a fun evening with friends & family.'), ('Rated 4.5', 'RATED\\\\n  Took a chance and tried this place .... Was well above what I expected ... Good food ... Professional and well mannered staff !! Great cosy effect over all ..... Would definitely recommend it ......'), ('Rated 5.0', 'RATED\\\\n  Excellent food, Amazing staff ,wide varieties of Cocktails and Good on pockets too:). The best place to relax around Domlur with a view at the rooftop...Thumbs Up!'), ('Rated 5.0', 'RATED\\\\n  Good ambience and excellent service bar in domlur area . Right place to hang with friends,.will be difinalety back again. Wish if they would have d j live, too.'), ('Rated 5.0', \\\"RATED\\\\n  Try the Wingless Flight. Absolutely awesome. And the food is super as well. It's a rooftop place and hence if there's a good wind flowing it'll make your day. The ambience is good and the staff is very polite and courteous. The location is very convenient as well. Worth a visit.\\\"), ('Rated 5.0', 'RATED\\\\n  The service was good, ambience is good. Would recommend to visit. Staff very good. View of Banglore is supper from here. Have a good choices of spirit.'), ('Rated 5.0', 'RATED\\\\n  First time beautiful experience enjoyed the view is perfect good service friendly staff special thanks to Mr samsul for the hospitality would recommend to all my friends'), ('Rated 5.0', 'RATED\\\\n  A great place to unwind after a long days work! Loved the food & the variety of cocktails! Burnt whiskey & kiwi frozen margarita are must haves! Mojito lovers have a lot of flavours to choose from!'), ('Rated 5.0', \\\"RATED\\\\n  Really nice place. Good service. Enjoyed it. I'm glad to have come here. Thank you. Windy nice place.\\\\n???????????????????????\\\")]\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"menu_item\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 6167,\n        \"samples\": [\n          \"['Veg Combo3', 'Non Veg Combo1', 'Non Veg Combo2', 'Non Veg Combo3', 'Fried Rice Combo2', 'Noodles Combo2', 'Non Veg Bowl', 'Fried Rice Combo1', 'Fried Rice Combo3', 'Fried Rice Combo4', 'Veg Bowl', 'Noodles Combo1', 'Fried Rice Combo2', 'Noodles Combo2', 'Non Veg Bowl', 'Fried Rice Combo1', 'Fried Rice Combo3', 'Fried Rice Combo4', 'Fish Fingers', 'Dim Sim Chilli Chicken', 'Honey Chilli Chicken', 'Spring Rolls', 'Crispy Peking Baby Corn', 'Mushroom Pepper Salt', 'Crispy Chilli Vegetable', 'Golden Fried Baby Corn', 'Crispy Dragon Vegetable', 'Peppered Corn Kernels', 'Chilli Garlic Potato', 'Honey Spiced Potato', 'Stir Fired Chinese Greens', 'Veg Steamed Momo\\u00c3\\u0083\\\\x83\\u00c3\\u0082\\\\x83\\u00c3\\u0083\\\\x82\\u00c3\\u0082\\\\x82\\u00c3\\u0083\\\\x83\\u00c3\\u0082\\\\x82\\u00c3\\u0083\\\\x82\\u00c3\\u0082\\\\x92s', 'Szechuan Crispy Veg', 'Drums Of Heaven', 'Thai Fried Chicken', 'Dragon Rolls With Garlic Sauce', 'Chicken Spring Rolls', 'Fried Chicken Wontons', 'Lemon Basil Chicken', 'Chinese Fried Chicken (half)', 'Chinese Fried Chicken (full)', 'Golden Fried Prawns', 'Dragon Chicken', 'Crispy Peking Chicken', 'Chicken Steamed Momo\\u00c3\\u0083\\\\x83\\u00c3\\u0082\\\\x83\\u00c3\\u0083\\\\x82\\u00c3\\u0082\\\\x82\\u00c3\\u0083\\\\x83\\u00c3\\u0082\\\\x82\\u00c3\\u0083\\\\x82\\u00c3\\u0082\\\\x92s', 'Beijing Chicken', 'Burnt Pepper Chicken', 'Dragon Rolls', 'Pepper Chicken', 'Chicken Pepper Salt', 'Chicken Lollypop', 'Chilli Basil Chicken', 'Shredded Threaded Chicken', 'Crispy Spinach Chicken', 'Crispy Peking Lamb', 'Crispy Shredded Lamb With Red & Green Chilli', 'Sliced Lamb With Chilli And Spring Onion', 'Crispy Konjee Lamb', 'Chicken Steak In Szechuan Sauce', 'Crispy Chicken Honey Pepper', 'Grilled Chicken', 'Thai Chilly Chicken', 'Pan Fried Chicken With Celery & Onion', 'Veg Fried Dumplings', 'Crispy Crunchy Spinach', 'Baby Corn Mushroom Pepper Celery', 'Tsing Hai Potato', 'Hunan Tofu', 'Crispy Chilly Baby Corn', 'Chicken Oriental Salad', 'Chicken With Beans Sprouts', 'Hot & Sour (non -veg )', 'Sweet Corn (non-veg)', 'Lung Fung (non-veg)', 'Manchow (non-veg)', 'Lemon Pepper (non-veg)', 'Wonton (non-veg)', 'Hot & Sour (veg )', 'Sweet Corn (veg)', 'Lung Fung (veg)', 'Manchow (veg)', 'Jade Corn (veg)', 'Jade Corn (non-veg)', 'Lemon Pepper (veg)', 'Wonton (veg)', 'Hunan Wonton(veg)', 'Hunan Wonton(non-veg)', 'Noodles (veg)', 'Noodles (non-veg)', 'Tom Yam Phak (veg)', 'Tom Yam Phak (non-veg)', 'Tom Kha Phak (veg)', 'Tom Kha Phak (non-veg)', 'Tom Yam (veg)', 'Tom Yam (non-veg)', 'Chilli Chowmein (veg)', 'Chilli Garlic Chowmein (veg)', 'Chilli Chowmein (non-veg)', 'Chilli Garlic Chowmein (non-veg)', 'Chinese Chopsuey (veg)', 'Singapore Noodles (veg)', 'American Chopsuey (veg)', 'Dragon Chopsuey (veg)', 'Danmein Noodles (veg)', 'Butter Garlic Noodles (veg)', 'Chinese Chopsuey (non-veg)', 'Singapore Noodles (non-veg)', 'American Chopsuey (non-veg)', 'Dragon Chopsuey (non-veg)', 'Danmein Noodles (chicken)', 'Butter Garlic Noodles (chicken)', 'Mixed Chowmein (non-veg)', 'Hakka Noodles (veg)', 'Hakka Noodles (non-veg)', 'Meefun (veg)', 'Meefun (chicken)', 'Cantonese Noodles (veg)', 'Cantonese Noodles (chicken)', 'Veg Koithio', 'Egg Koithio', 'Chicken Koithio', 'Prawns Koithio', 'Mixed Koithio', 'Egg Fried Rice', 'Shanghai Fried Rice (veg)', 'Szechuan Fried Rice (veg)', 'Chilli Garlic Rice (veg)', 'Butter Garlic Rice', 'Chicken Fried Rice', 'Prawn Fried Rice', 'Shanghai Fried Rice (non-veg)', 'Szechuan Fried Rice (non-veg)', 'Mixed Fried Rice (non-veg)', 'Chilli Garlic Rice (non-veg)', 'Butter Garlic Rice (non-veg)', 'Veg Fried Rice', 'Ginger Capsicum Rice (veg)', 'Ginger Capsicum Rice (non-veg)', 'Mushroom Tomato Rice (veg)', 'Mushroom Tomato Rice (non-veg)', 'Thai Vegetable Red Curry With Steamed Rice', 'Thai Vegetable Green Curry With Steamed Rice', 'Thai Prawn Red Curry With Steamed Rice', 'Thai Chicken Green Curry With Steamed Rice', 'Thai Chicken Red Curry With Steamed Rice', 'Basil Fried Rice', 'Phad Thai (veg)', 'Phad Thai (non-veg)', 'Khao Phad Kapro (veg)', 'Khao Phad Kapro (non-veg)', 'Vegetable Fuyoung', 'Chicken Fuyoung', 'Prawn Fuyoung', 'Mixed Fuyoung', 'Prawns In Chilli Sauce', 'Chilli Fish', 'Fish In Pickle Chilli Sauce', 'Prawns In Garlic / Hot Garlic Sauce', 'Prawns In Black Pepper Sauce', 'Sweet And Sour Prawn', 'Hunan Prawn', 'Fish Manchurian', 'Fish In Oysters Sauce', 'Prawns In Hot Bean Sauce', 'Kung Pao Prawn', 'Fish In Chilli Garlic Sauce', 'Chilli Chicken', 'Chicken In Pickle Chilli Sauce', 'Chicken Manchurian', 'Garlic Chicken', 'Hong Kong Chicken', 'Szechuan Chicken', 'Kung Pao Chicken', 'Tsing Hoi Chicken', 'Hunan Chicken', 'Sweet & Sour Chicken', 'Ginger Chicken', 'Honey Hunan Chicken', 'Roast Chicken With Vegetables', 'Chicken In Black Pepper Sauce', 'Chicken In Hot Bean Sauce', 'Chicken With Lemon Sauce', 'General Tao Chicken', 'Great Wall Chicken', 'Roast Chicken In Chilly Plum Sauce', 'Chicken In Hoisen Sauce', 'Stir Fried Broccoli', 'Cauliflower Manchurian', 'Choice Of Vegetables In Pepper Salt', 'Veg Balls In Hot Garlic Sauce', 'Mixed Veg Hong Kong Baby Corn Mushroom', 'Vegetable Dumpling Manchurian', 'Potato Black Pepper Sauce', 'Buddha\\u00c3\\u0083\\\\x83\\u00c3\\u0082\\\\x83\\u00c3\\u0083\\\\x82\\u00c3\\u0082\\\\x82\\u00c3\\u0083\\\\x83\\u00c3\\u0082\\\\x82\\u00c3\\u0083\\\\x82\\u00c3\\u0082\\\\x92s Delight', 'Baby Corn And Bamboo Shoot In Chilly Bean Sauce', 'Mix Veg In Devils Sauce', 'Mafu Tufu', 'Kung Pao Tofu', 'Mix Veg In Pickle Chilli Sauce', 'Fried Tofu & Broccoli In Chilli Plum Sauce', 'Veg Steamed Rice / Koithio', 'Hunan Steamed Rice / Koithio (veg)', 'Szechwan Steamed Rice / Koithio (veg)', 'Triple Szechuan Steamed Rice / Koithio (veg)', 'Chicken Steamed Rice / Koithio', 'Prawn Steamed Rice / Koithio', 'Mixed Steamed Rice / Koithio (non-veg)', 'Hunan Steamed Rice / Koithio (non-veg)', 'Szechwan Steamed Rice / Koithio (non-veg)', 'Triple Szechuan Steamed Rice / Koithio (non-veg)', 'Vegetable Stewed Rice', 'Chicken Stewed Rice', 'Prawns Stewed Rice', 'Mixed Stewed Rice (non-veg)', 'Por Pia Sai Pak (veg)', 'Por Pia Phak (veg)', 'Chilli Basil Chicken (non-veg)', 'Prawns In Samble Sauce (non-veg)', 'Kai Phad Namprik (non-veg)', 'Veg Combo1', 'Veg Combo2', 'Veg Combo3', 'Veg Combo4', 'Non Veg Combo1', 'Non Veg Combo2', 'Non Veg Combo3', 'Non Veg Combo4']\",\n          \"['Mango Pastry', 'Blueberry Pastry', 'Pineapple Pastry', 'Strawberry Pastry', 'Butterscotch Pastry', 'White Forest Pastry', 'Black Forest Pastry', 'Vanilla Choco Pastry', 'Black Current Pastry', 'Truffle Chocolate Pastry', 'Mango Pastry', 'Blueberry Pastry', 'Pineapple Pastry', 'Strawberry Pastry', 'Butterscotch Pastry', 'White Forest Pastry', 'Black Forest Pastry', 'Choco Topee Pastry', 'Black Current Pastry', 'Truffle Chocolate Pastry', 'Mango Cake', 'Blueberry Cake', 'Pineapple Cake', 'Red Velvet Cake', 'Strawberry Cake', 'Butterscotch Cake', 'White Forest Cake', 'Black Forest Cake', 'Irish Coffee Cake', 'Vanilla Choco Cake', 'Raspberry Cake', 'Truffle Chocolate Cake', 'Mango Cake', 'Blueberry Cake', 'Pineapple Cake', 'Red Velvet Cake', 'Strawberry Cake', 'Butterscotch Cake', 'White Forest Cake', 'Black Forest Cake', 'Irish Coffee Cake', 'Vanilla Choco Cake', 'Black Current Cake', 'Truffle Chocolate Cake', 'Mango Cake', 'Pineapple Cake', 'Strawberry Cake', 'Butterscotch Cake', 'White Forest Cake', 'Black Forest Cake', 'Vanilla Choco Cake', 'Almond and Honey Cake', 'Black Current Cake', 'Truffle Chocolate Cake', 'Mango Cake', 'Pineapple Cake', 'Red Velvet Cake', 'Strawberry Cake', 'Butterscotch Cake', 'Black Forest Cake', 'Vanilla Choco Cake', 'Truffle Chocolate Cake']\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"listed_in(type)\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 7,\n        \"samples\": [\n          \"Buffet\",\n          \"Cafes\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"listed_in(city)\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 19,\n        \"samples\": [\n          \"Banashankari\",\n          \"Brookefield\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code execution_count="10" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":617}" executionInfo="{\"elapsed\":84,\"status\":\"ok\",\"timestamp\":1771829368754,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="LwJjVucCCN3d" outputId="22a24e78-9c66-4663-97aa-865ddd7591e8"}
``` python
pd.isnull(df).sum()
```

::: {.output .execute_result execution_count="10"}
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
      <th>url</th>
      <td>0</td>
    </tr>
    <tr>
      <th>address</th>
      <td>1</td>
    </tr>
    <tr>
      <th>name</th>
      <td>1</td>
    </tr>
    <tr>
      <th>online_order</th>
      <td>1</td>
    </tr>
    <tr>
      <th>book_table</th>
      <td>1</td>
    </tr>
    <tr>
      <th>rate</th>
      <td>5124</td>
    </tr>
    <tr>
      <th>votes</th>
      <td>1</td>
    </tr>
    <tr>
      <th>phone</th>
      <td>724</td>
    </tr>
    <tr>
      <th>location</th>
      <td>14</td>
    </tr>
    <tr>
      <th>rest_type</th>
      <td>153</td>
    </tr>
    <tr>
      <th>dish_liked</th>
      <td>18824</td>
    </tr>
    <tr>
      <th>cuisines</th>
      <td>29</td>
    </tr>
    <tr>
      <th>approx_cost(for two people)</th>
      <td>177</td>
    </tr>
    <tr>
      <th>reviews_list</th>
      <td>1</td>
    </tr>
    <tr>
      <th>menu_item</th>
      <td>1</td>
    </tr>
    <tr>
      <th>listed_in(type)</th>
      <td>1</td>
    </tr>
    <tr>
      <th>listed_in(city)</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>
```
:::
:::

::: {.cell .code execution_count="11" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":18,\"status\":\"ok\",\"timestamp\":1771829388009,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="wLExnwuGCtmd" outputId="ebacb859-da68-45ee-c048-98aa43b938fa"}
``` python
df.shape
```

::: {.output .execute_result execution_count="11"}
    (34203, 17)
:::
:::

::: {.cell .code execution_count="12" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":18,\"status\":\"ok\",\"timestamp\":1771829401540,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="AfhfRlC1Cys1" outputId="1afc1814-0d24-4e55-c008-264a7dd0afa8"}
``` python
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 34203 entries, 0 to 34202
    Data columns (total 17 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   url                          34203 non-null  object 
     1   address                      34202 non-null  object 
     2   name                         34202 non-null  object 
     3   online_order                 34202 non-null  object 
     4   book_table                   34202 non-null  object 
     5   rate                         29079 non-null  object 
     6   votes                        34202 non-null  float64
     7   phone                        33479 non-null  object 
     8   location                     34189 non-null  object 
     9   rest_type                    34050 non-null  object 
     10  dish_liked                   15379 non-null  object 
     11  cuisines                     34174 non-null  object 
     12  approx_cost(for two people)  34026 non-null  object 
     13  reviews_list                 34202 non-null  object 
     14  menu_item                    34202 non-null  object 
     15  listed_in(type)              34202 non-null  object 
     16  listed_in(city)              34202 non-null  object 
    dtypes: float64(1), object(16)
    memory usage: 4.4+ MB
:::
:::

::: {.cell .code execution_count="13" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":300}" executionInfo="{\"elapsed\":32,\"status\":\"ok\",\"timestamp\":1771829484909,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="cRZAWKNNRByU" outputId="4da03b12-3899-43b9-c54b-f9aebb5e598b"}
``` python
df.describe()
```

::: {.output .execute_result execution_count="13"}
``` json
{"summary":"{\n  \"name\": \"df\",\n  \"rows\": 8,\n  \"fields\": [\n    {\n      \"column\": \"votes\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 12551.566232419334,\n        \"min\": 0.0,\n        \"max\": 34202.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          257.40266651073034,\n          39.0,\n          34202.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::
:::

::: {.cell .code execution_count="23" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":435}" executionInfo="{\"elapsed\":655,\"status\":\"ok\",\"timestamp\":1771830823579,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="zYoU1BaYSAxN" outputId="b07cc164-f767-469d-bc8c-fc6226ce20bc"}
``` python
corr_data = df.corr(numeric_only=True)
sns.heatmap(corr_data, annot=True)
plt.show()
```

::: {.output .display_data}
![](vertopal_b61acf8d8a2049fb8898aa8c1f60eb00/28fee3c2c5249f49a4045da7f28ffc69e68d4eb5.png)
:::
:::

::: {.cell .code execution_count="25" executionInfo="{\"elapsed\":20,\"status\":\"ok\",\"timestamp\":1771830968944,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="apkpr8KPWbFP"}
``` python
order_count=df["name"].value_counts()
```
:::

::: {.cell .code execution_count="26" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":27,\"status\":\"ok\",\"timestamp\":1771831111096,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="hUo2fQflXERH" outputId="e8c46867-f002-4455-c503-3cacf28e375d"}
``` python
print(order_count)
```

::: {.output .stream .stdout}
    name
    Cafe Coffee Day         59
    Onesta                  56
    Empire Restaurant       47
    Petoo                   46
    Just Bake               46
                            ..
    Night Food Delivery      1
    FoodIshq                 1
    Desi Chinese kitchen     1
    Nagappa Naidu Hotel      1
    Sri Bhagya Veg           1
    Name: count, Length: 7275, dtype: int64
:::
:::

::: {.cell .code execution_count="27" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":22,\"status\":\"ok\",\"timestamp\":1771831184026,\"user\":{\"displayName\":\"Game Vinator\",\"userId\":\"04722404070732496232\"},\"user_tz\":-330}" id="tI1uMS9vXTfA" outputId="9cca9e6c-3bf1-4d44-ca04-877976968ed9"}
``` python
print("Max:",order_count.max())
print("Min:",order_count.min())
print("Mean:",order_count.mean())
```

::: {.output .stream .stdout}
    Max: 59
    Min: 1
    Mean: 4.701305841924398
:::
:::
