# Twitter Sentiment Analysis Classification Project:

## Combating the Spread of Misinformation and Extremism in a Networked World


## Background:  

#### The Influence of Technology on the Rise of Hate Speech:

Hateful and offensive speech in social networks and other online fora is a persistent and toxic problem. 
Violence attributed to online hate speech has increased worldwide. White-supremacist groups use social media as a tool to distribute their message, where they can incubate their hate online and allow it to spread. But when their rhetoric reaches certain people, the online messages can turn into real-life violence. Several incidents in recent years have shown that when online hate goes offline, it can be deadly.


#### Challenges for Tech Giants in Monitoring Hate Speech:

For all the advances being made in the field, artificial intelligence still struggles when it comes to identifying hate speech. When he testified before Congress in April 2018, Facebook CEO Mark Zuckerberg said it was “one of the hardest” problems to solve.

Since humans can’t always agree on what can be classified as hate speech, it is especially complicated to create a universal machine learning algorithm that would identify it. 

One more complication is that it is hard to distinguish hate speech from just an offensive language, even for a human. This becomes a problem especially when labeling is done by random users based on their own subjective judgment, like in this dataset, where users were suggested to label tweets as “hate speech”, “offensive language” or “neither”. 


## Methodology:

#### Dataset:

A publicly available Twitter sentiment analysis dataset available was used for this project. The public dataset consists of over 30,000 tweets that have been pre-labeled as "hate speech" for including racist or sexist content). 

#### Data Cleaning and Exploration:

Tweets were cleaned to remove user handle names, punctuation and other non-numerical text. Simple word counts, phrases and predictive word embedding were explored for each class and can be found in the data exploration notebook. 

Frequency of Words Represented in Class 0 (pre-labeled as not hate):

![](data/wordcloud0.png)

Frequency of Words Represented in Class 1 (pre-labaled as hate):

![](data/wordcloud1.png)

#### Training and Testing Predictive Models:

The data was divided into training, validation and test sets based off the portion of the dataset that was publicly available.

The performance of five classifiers (Multinomial Naive Bayes, Support Vector Machine, Logistic Regression, Random Forest and AdaBoost) were initially compared using different methods for:

- text cleaning (tokenizing, stemming and lemmatizing)
- correcting for class imbalance (oversampling, undersampling, SMOTE and class weights)
- text vectorization methods - count vectorizer, tfidf vectorizer, tfidf with 1-2 n-grams, tfidf with 2-3 ngrams; word embedding methods using Word2Vec and GLoVe were also explored.

The best performance for each model was chosen for further tuning of model hyperparameters, and then final F1 scores for each model were compared to determine the final classification model.

### Final Model Selection and Performance:

The best performing model from preliminary analyses was a logistic regression classifer, using a simple word frequency vectorizer (count vectorizer) and oversampling on the training set in order to correct for class imbalances. 

The probability threshold used to determine binary classification was lowered to .2 during the model training phase in order to reduce the number of false negatives and false positives. Final scores are reported below:

Accuracy: 0.96
Precision: 0.74 
Recall: 0.62
F1: 0.68

![](visualizations/final_cm.png)


## Conclusions:


#### Limitations:

So when designing a model, it is important to follow criteria that will help to distinguish between hate speech and offensive language.



#### Next Steps: 




  