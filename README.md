# Twitter Sentiment Analysis Classification Project:

## Combating the Spread of Misinformation and Extremism in a Networked World


### Background:  

#### The Influence of Technology on the Rise of Hate Speech:

Hateful and offensive speech in social networks and other online fora is a persistent and toxic problem. 
Violence attributed to online hate speech has increased worldwide. White-supremacist groups use social media as a tool to distribute their message, where they can incubate their hate online and allow it to spread. But when their rhetoric reaches certain people, the online messages can turn into real-life violence. Several incidents in recent years have shown that when online hate goes offline, it can be deadly.


#### Challenges for Tech Giants in Monitoring Hate Speech:

For all the advances being made in the field, artificial intelligence still struggles when it comes to identifying hate speech. When he testified before Congress in April 2018, Facebook CEO Mark Zuckerberg said it was “one of the hardest” problems to solve.

Since humans can’t always agree on what can be classified as hate speech, it is especially complicated to create a universal machine learning algorithm that would identify it. 

One more complication is that it is hard to distinguish hate speech from just an offensive language, even for a human. This becomes a problem especially when labeling is done by random users based on their own subjective judgment, like in this dataset, where users were suggested to label tweets as “hate speech”, “offensive language” or “neither”. 

## Methodology:

#### Dataset:

A publicly available Twitter sentiment analysis dataset available through the Analytics Vindhya website was used for this project. The dataset consists of over 30,000 tweets that have been pre-labeled as "hate speech" (includes racist or sexist content" or "not hate speech".

The performance of five classifiers (Multinomial Naive Bayes, Support Vector Machine, Logistic Regression, Random Forest and AdaBoost) were initially tested and compared, using different methods for: 

- text cleaning (tokenizing, stemming and lemmatizing)
- correcting for class imbalance (oversampling, undersampling, SMOTE and class weights)
- text vectorization methods - count vectorizer, tfidf vectorizer, tfidf with 1-2 n-grams, tfidf with 2-3 ngrams; word embedding methods using Word2Vec and GLoVe were also explored.

#### Data Cleaning and Exploration:

Tweets were cleaned of punctuation



#### Text Vectorizers:

Count Vectorizer, TFIDF Vectorizer, Word2Vec, GLoVe

#### Classifiers:

Naive Bayes, Support Vector Machine, Logistic Regression, Random Forest, 


## Results:


## Conclusions:


#### Limitations:

So when designing a model, it is important to follow criteria that will help to distinguish between hate speech and offensive language.



#### Future Work:




  