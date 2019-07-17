# Twitter-Hate-Classification-Project


## Background:  

#### Technology's Influnce on the Rise of Hate Speech and Violence:

Hateful and offensive speech in social networks and other online fora is a persistent and toxic problem. 

Violence attributed to online hate speech has increased worldwide. Societies confronting the trend must deal with questions of free speech and censorship on widely used tech platforms.

White-supremacist groups use social media as a tool to distribute their message, where they can incubate their hate online and allow it to spread. But when their rhetoric reaches certain people, the online messages can turn into real-life violence. Several incidents in recent years have shown that when online hate goes offline, it can be deadly.

“I think that the white-supremacist movement has used technology in a way that has been unbelievably effective at radicalizing people,” said Adam Neufeld, vice president of innovation and strategy for the Anti-Defamation League.

“We should not kid ourselves that online hate will stay online,” Neufeld added. “Even if a small percentage of those folks active online go on to commit a hate crime, it’s something well beyond what we’ve seen for America.”


#### Challenges for Tech Giants in Monitoring Hate Speech:

For all the advances being made in the field, artificial intelligence still struggles when it comes to identifying hate speech. When he testified before Congress in April 2018, Facebook CEO Mark Zuckerberg said it was “one of the hardest” problems to solve.

Since humans can’t always agree on what can be classified as hate speech, it is especially complicated to create a universal machine learning algorithm that would identify it. Besides, the datasets used to train models tend to “reflect the majority view of the people who collected or labeled the data”, according to Tommi Gröndahl from the Aalto University, Finland (source).

One more complication is that it is hard to distinguish hate speech from just an offensive language, even for a human. This becomes a problem especially when labeling is done by random users based on their own subjective judgment, like in this dataset, where users were suggested to label tweets as “hate speech”, “offensive language” or “neither”. So when designing a model, it is important to follow criteria that will help to distinguish between hate speech and offensive language.

#### My Purpose in Selecting this Topic:

- Important/hot topic issue that continues to be a hotbed of research and experimentation
- Aligns with my specific areas of PhD training (behavioral science; attitude and persuasion; prevention of violence)
 -- nice story as to why I have come back to (or "moved to") data science, AI, etc.:
   - rise of technology has made it possible for hate speech to thrive and grow online
   - the magnitude of this problem now means that traditional research methodology (labeling by hand, etc.) is no longer sufficient 


## Potential Challenges:

1. Challenging classification topic, getting a good fit might take a lot of work: 
- use alternative datasets with more rigorous coding coding mechanisms, different topic: Fake News, Customer Sentiment, etc.).... Need to decide on appropriate method for cleaning text (e.g., handling hashtags, etc.)

2. Thoroughly exhausting the options for finding a good fit for just one model can be consuming (cleaning strategy, vectorization strategy, class-balancing, grid search, etc.)
 --> Naive Bayes, SVM, Random Forest:  each with lemmitization, TFIDF and Word2 Vec Vectorization; class weight + threshold, Random Search (vs. Grid Search)
 --> with Word2Vec, only need to tokenize beforehand??
 
3. Plan for organizing code in a way that's clean and easy to reproduce tests with various iterations.  Challenge here has just been getting the right order of splitting, vectorization, up/downsampling

--> functions (x_train, y_train, x_test, y_test, classifier) 
--> pipelines 

4. RNN (?) and/or clustering

   
## Schedule/Action Plan: 

Week ending 7/19:

Wednesday COB: re-run a few models (including Random Forest) with class weights and ROC adjustment, grid search, etc. 
--> with TFIDF and Word2Vec (embeddings trained on my own data)

Thursday COB:  Have re-run "baseline models" with pre-trained word-embeddings from Glove and/or RNN

Friday COB: continue and refine Thursday's work as needed; quick check on major questions before the weekend


Week 7/22 - 7/26:

Monday: have the results for all baseline (non RNN) models completed, if not RNN

Tuesday/Wednesday: notebook cleanup (imported functions, etc.), draft of read me preparation, visualization preparation

Thursday: rough draft of ppt presentation

Friday/Monday of following week: "bonus" analyses:  clustering, additional data exploration



