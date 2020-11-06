import sys,tweepy,csv,re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords


stop = stopwords.words("english")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

consumerKey = '5wKyFWhTOC8EbBrIOCukZMsxE'
consumerSecret = 'CVl9ssF3ZFWBbX6U13xHNKvOr4ucRL5xKhHGWrIg8zrLENiuJi'
accessToken = '880413204932841472-daya39ha3F5BBDUfGomXtds5CfzN8ir'
accessTokenSecret = '5bjWFgvnmT2cYU462rpgLZ5amLJDQtbpyhSMa7k8C8Yie'
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

tweets = searchTerm = input("Enter Keyword/Tag to search about: ")
NoOfTerms = int(input("Enter how many tweets to search: "))

tweets = tweepy.Cursor(api.search, q=searchTerm,lang='en').items(NoOfTerms)

sentiment_score_ratio = []

tweets_list = [[tweet.user.name, tweet.text,tweet.user.screen_name, tweet.user.id_str, tweet.user.location,  tweet.user.description, tweet.user.verified, tweet.user.followers_count, tweet.user.friends_count, tweet.user.statuses_count, tweet.user.listed_count, tweet.user.created_at] for tweet in tweets]
user_df = pd.DataFrame(tweets_list,columns = ['name','content','screen_name','id','location','status','verified','followerscount','friends_count','statuscount','listedcount','created_at'])
#print(user_df)

# text preprocessing 
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
# clean tweet
def clean_tweet(tweet):
    #remove twitter return handles (RT @xxx:)
    tweet = np.vectorize(remove_pattern)(tweet,"RT @[\w]*:")

    #remove twitter handles
    tweet = np.vectorize(remove_pattern)(tweet,"@[\w]*")

    #remove URL links 
    tweet = np.vectorize(remove_pattern)(tweet,"https?://[A-Za-z0-9./]*")

    #remove special characters,numbers,punctuations

    tweet = np.core.defchararray.replace(tweet,"[^a-zA-Z]"," ")

    return tweet

#sentiment scores
def sentiment_score(df):
    label = []
    for tweet in df['text'] :
        tweet_clean = clean_tweet(tweet)
        tweet_clean = str(tweet_clean)
        sentiment = analyzer.polarity_scores(tweet_clean)

        if(sentiment['compound']>=0.05):
            label.append('pos')
        elif(sentiment['compound']<=-0.05):
            label.append('neg')
        else:
            label.append('neu')
    tweet_df['label'] = label
    pos = label.count('pos')
    neg = label.count('neg')
    if(neg==0):
        neg+=1
    return pos/neg
bot = []

for i in range(len(user_df)):
    user_tweets = api.user_timeline(screen_name = user_df['screen_name'][i],count = 20,full_text=True,tweet_mode= "extended")
    data = [[tweet.created_at,tweet.full_text] for tweet in user_tweets]
    tweet_df = pd.DataFrame(data,columns=['created_at','text'])
    score = sentiment_score(tweet_df)
    sentiment_score_ratio.append(score)

user_df['score_ratio'] = sentiment_score_ratio

def fake_detection_algorithm(df):
    for i in range(len(df)):
        if(df['verified'][i] == True):
            bot.append(0)
            continue
            
        else:
            if((df['status'][i]==" ") or (df['name'][i].isnumeric()==True)):
                bot.append(1)
            else:
                if((df['followerscount'][i]<df['statuscount'][i]) or (df['friends_count'][i]<10) or (df['statuscount'][i]>200)):
                    bot.append(1)
                    continue
            
            if(df['followerscount'][i]/df['friends_count'][i]<0.5):
                bot.append(1)
                continue
            
            if(df['score_ratio'][i]<1):
                bot.append(1)
            else:
                bot.append(0)

    
    user_df['bot'] = bot
    print(len(bot))
    
    

fake_detection_algorithm(user_df)
print(user_df)

rule_accuracy = (bot.count(1)/len(bot))*100
print('Accuracy acchieved using our rule based algo for bot detection is : ',rule_accuracy)
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# implementing machine learning classifiers

tweet = user_df['content']
bot = user_df['bot']
data = {'X':tweet,'y':bot}
print(len(tweet))
clf_df = pd.DataFrame(data)

print(len(clf_df))

#preprocessing of tweets
tweet_clean = []
for j in range(len(user_df['content'])):
    cltweet = clean_tweet(user_df['content'][j])
    tweet_clean.append(str(cltweet))

print(len(tweet_clean))
clf_df['clean_tweet'] = tweet_clean


# feature extraction using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfIdfVectorizer=TfidfVectorizer(use_idf=True,stop_words = stop)
X_tf = tfIdfVectorizer.fit_transform(clf_df['clean_tweet']).toarray()


# spliting data into train and  test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_tf,clf_df['y'],test_size = 0.3,random_state = 0)


# Applying decision tree classifier

from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)
dct.fit(X_train,y_train)

from sklearn import metrics
y_pred = dct.predict(X_test)
print(y_pred)
# predicting accuracy
print("Accuracy got in predicting bot using DCT classifier is ",metrics.accuracy_score(y_test,y_pred))


# Applying Random forest Classifier for bot classification










            


        
            








# bot-->1 and normal_account-->0
