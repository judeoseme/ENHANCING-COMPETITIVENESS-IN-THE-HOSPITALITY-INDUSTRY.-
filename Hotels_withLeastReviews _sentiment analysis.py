#!/usr/bin/env python
# coding: utf-8

# In[78]:


from wordcloud import WordCloud, STOPWORDS


# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
import warnings
warnings.filterwarnings("ignore")


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn import metrics


# ## Loading Pre-trained Model for Sentiment Analysis

# In[3]:


#!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[4]:


analyser = SentimentIntensityAnalyzer()


# In[5]:


analyser.polarity_scores("crossover food ever fully satisfied one top restaurant would recommend phuket")


# ## Actual Data Loading 

# In[6]:


data = pd.read_csv("tourist_accommodation_reviews.csv")
data.shape


# In[7]:


data.head()


# In[8]:


#Unique Restaurants
data['Hotel/Restaurant name'].nunique()


# In[9]:


data['Location'].nunique()


# ## Check for Null values 

# In[10]:



data.isnull().sum()


# ## Joining Original Data with the Pre_Processed Reviews

# In[11]:


cleaned_reviews = pd.read_csv("Raw_CleanedReviews.csv")
cleaned_reviews.head()


# In[12]:


cleaned_reviews.shape, data.shape


# In[13]:


combined_data = pd.concat([data,cleaned_reviews],axis=1)
combined_data.head()


# In[14]:


combined_data['Location_Hotel'] = combined_data['Location']+'_'+combined_data['Hotel/Restaurant name']


# In[15]:


combined_data.columns


# In[16]:


processed_data = combined_data[['ID', 'Review Date', 'Location', 'Hotel/Restaurant name','Processed_review', 'Location_Hotel']]


# In[17]:


processed_data.head()


# ## Check on  Location_Hotel with Least Reviews

# In[31]:


LocationReviewCount = pd.DataFrame(combined_data.groupby(['Location_Hotel']).                         agg(Total_Reviews=('Processed_review','count'))).reset_index().                    sort_values(by="Total_Reviews",ascending=True)
LocationReviewCount.head(10)


# In[36]:


Location_Hotel_30 = list(LocationReviewCount['Location_Hotel'].values)[:30]
Location_Hotel_30


# ## Selecting 30 hotels with Least reviews
# 

# In[37]:


df = combined_data[combined_data['Location_Hotel'].isin(Location_Hotel_30)]
df.shape


# In[39]:


df['Location_Hotel'].nunique()


# In[40]:


df.head()


# In[41]:


df = df.drop(columns=['Review'])
df.head()


# In[43]:


df.shape


# ## Fetching Processed Reviews for 30 Restaurants

# In[42]:


reviews = pd.DataFrame(df['Processed_review'])
reviews.head()


# In[44]:


reviews.shape


# ## Creating Embeddings Count Vectorizer 

# In[45]:



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Considering 3 grams and mimnimum frq as 0
CountVect = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1)
CountVect.fit(reviews['Processed_review'])
desc_matrix = CountVect.transform(reviews['Processed_review'])


# In[46]:


desc_matrix.shape


# In[47]:


from sklearn.cluster import KMeans


# In[48]:


# implement kmeans
num_clusters = 2
km = KMeans(n_clusters=num_clusters)
km.fit(desc_matrix)
clusters = km.labels_.tolist()


# In[49]:


# create DataFrame from all of the input files.
ReviewsClassification = {'Reviews_Class': reviews['Processed_review'].tolist(), 'Cluster': clusters}
frame = pd.DataFrame(ReviewsClassification, index = [clusters])
frame


# In[50]:


frame[frame['Cluster']==0]["Reviews_Class"].values[-10:]


# In[51]:


frame[frame['Cluster']==1]["Reviews_Class"].values[0:10]


# ## Combining Review Classification Result with Dataset

# In[52]:


HotelReviewCount = df


# In[53]:


HotelReviewCount = HotelReviewCount.reset_index(drop=True)
frame = frame.reset_index(drop=True)


# In[54]:


final_result = pd.concat([HotelReviewCount,frame], axis=1)


# In[55]:


final_result.head()


# In[56]:


final_result.to_csv("Final_Result_Hotels_withLeastReviews.csv",index=False)


# In[57]:


final_result.shape


# In[116]:


# AS we some samples of reviews falling in two different cluster, 
# with observation concludes that cluster 0 refers to negative and cluster 1 refers to positive class 


# In[58]:


Hotels_with_Negative_reviews = pd.DataFrame(final_result[final_result['Cluster']==0].groupby(['Location_Hotel']).                         agg(Total_Negative_Reviews=('Cluster','count'))).reset_index().                    sort_values(by="Total_Negative_Reviews",ascending=False)
Hotels_with_Negative_reviews.head(10)


# In[64]:


Hotels_with_positive_reviews = pd.DataFrame(final_result[final_result['Cluster']==1].groupby(['Location_Hotel']).                         agg(Total_Positive_Reviews=('Cluster','count'))).reset_index().                    sort_values(by="Total_Positive_Reviews",ascending=False)
Hotels_with_positive_reviews.head(10)


# ## Most Common Words

# In[60]:


negative_reviews =  final_result[final_result['Cluster']==0]


# In[66]:


reviews_list = list(negative_reviews['Reviews_Class'].values)


# In[69]:


reviews_list[-10:]


# In[70]:


analyser.polarity_scores("go drink interesting place relax atmosphere friendly staff place russian food come highly recommend every customer get free shot whisky place order get warm well dada uncle yura")


# In[ ]:





# In[71]:


from collections import Counter


# In[72]:


negative_reviews_wordlist = []
negative_reviews_statments = ''
for i in reviews_list:
    negative_reviews_statments = negative_reviews_statments+' '+i
    word_list = i.split(' ')
    negative_reviews_wordlist.extend(word_list)


# In[73]:


negative_reviews_wordlist[:5]


# In[74]:


wordcounter = Counter(negative_reviews_wordlist)


# In[75]:


wordcounter.most_common(20)


# In[76]:


import matplotlib.pyplot as plt
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[79]:


# Generate word cloud
wordcloud = WordCloud(width= 1000, height = 500, random_state=1, background_color='salmon', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(negative_reviews_statments)
# Plot
plot_cloud(wordcloud)


# In[ ]:





# ## Positive Reviews

# In[80]:


positive_reviews =  final_result[final_result['Cluster']==1]


# In[81]:


reviews_list_positive = list(positive_reviews['Processed_review'].values)


# In[82]:


reviews_list_positive[:2]


# In[83]:


analyser.polarity_scores("visit resort restaurant pleasantly surprise quality food attentiveness staff careful ask spicy hot hot food top service yoo good busy")


# In[84]:


from collections import Counter


# In[85]:


positive_reviews_wordlist = []
positive_reviews_statments = ''
for i in reviews_list_positive:
    positive_reviews_statments = positive_reviews_statments+' '+i
    word_list = i.split(' ')
    positive_reviews_wordlist.extend(word_list)


# In[86]:


positive_reviews_wordlist[:5]


# In[87]:


wordcounter_positive = Counter(positive_reviews_wordlist)


# In[88]:


wordcounter_positive.most_common(20)


# In[89]:


import matplotlib.pyplot as plt
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[90]:


# Generate word cloud
wordcloud = WordCloud(width= 1000, height = 500, random_state=1, background_color='salmon', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(positive_reviews_statments)
# Plot
plot_cloud(wordcloud)


# In[ ]:




