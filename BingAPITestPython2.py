
# coding: utf-8

# In[1]:

import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

subscription_key = "89535e470d874b68bcaefed5e39f9f8d"
#key1:89535e470d874b68bcaefed5e39f9f8d
#key2:ff1aee6fc7554c27a759150a7c6d4f83
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
search_term = "puppies"


# In[2]:

headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
params  = {"q": search_term, "license": "public", "imageType": "photo"}


# In[3]:

response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()
thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:16]]


# In[4]:

thumbnail_urls


# In[5]:

import urllib

urllib.urlretrieve(thumbnail_urls[0], "local-filename.jpg")


# In[ ]:



