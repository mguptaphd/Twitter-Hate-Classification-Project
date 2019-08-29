#!/usr/bin/env python
# coding: utf-8

# ## Functions to View Classifier Metrics:

# In[7]:


def compare_predictions_dataframe (y_true, y_pred, y_pred_prob):
    '''
    Combines and returns actual and predicted classes in a single datafame
    
    Parameters:
    
    y_true: actual y value 
    
    y_pred: predicted y value
    
    y_pred_prob: predicted probability for y

    '''
    pred_df = pd.DataFrame(y_pred_prob, columns = ['prob 0', 'prob 1'])
    pred_df['predicted class'] = y_pred
    pred_df.index = y_true.index 
    pred_df['actual class'] = y_true
    
    return pred_df.round(2)


# In[12]:


def print_confusion_matrix (y_true, y_pred):
    '''
    Returns simple confusion matrix with labeled index and column headings.
    '''
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index = ['actual 0','actual 1'], 
          columns = ['predicted 0', 'predicted 1'])  
        
    return cm 


# In[2]:


def seaborn_confusion_matrix (y_true, y_pred):
    '''
    Visualizes confusion matrix in seaborn.  Figure can be saved by wrapping function in plt.savefig
    '''
    cm = confusion_matrix(y_true, y_pred) 
    sns.heatmap(cm, square=True, annot=True, fmt = 'g', cmap='YlGnBu', cbar=False, 
                xticklabels=['Not Hate', 'Hate Speech'], 
                yticklabels=['Not Hate', 'Hate Speech'])
    plt.title('Confusion Matrix:')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
#     plt.savefig('data/cm')

    tn, fp, fn, tp = cm.ravel()


# In[23]:


def print_classification_report(y_true, y_pred):
    '''
    Returns pre-labeled classification report
    '''
    report = print(metrics.classification_report(y_true, y_pred, target_names = ['not hate', 'hate speech']))

    return report


# In[15]:


def compare_classification_metrics (y_train, y_train_pred, y_val, y_val_pred):
    '''
    Returns a dictionary containing metrics for train and validation sets
    '''
    #prints train and validation set metrics
    metrics_dict = {
    'Train Accuracy' : round(metrics.accuracy_score(y_train, y_train_pred),2),
    'Train Precision' : round(metrics.precision_score(y_train, y_train_pred),2),
    'Train Recall' : round(metrics.recall_score(y_train, y_train_pred),2),
    'Train F1': round(metrics.f1_score(y_train, y_train_pred),2),

    'Validation Accuracy': round(metrics.accuracy_score(y_val, y_val_pred),2),
    'Validation Precision' : round(metrics.precision_score(y_val, y_val_pred),2),
    'Validation Recall': round(metrics.recall_score(y_val, y_val_pred),2),
    'Validation F1': round(metrics.f1_score(y_val, y_val_pred),2)
    }
    
    return metrics_dict   #  to return as a dataframe: pd.DataFrame(metrics_dict, index=[0])


# In[3]:


def adjust_threshold_and_score (y_true, y_pred_prob, threshold, RNN = False):
    '''
    Plots confusion matrix and calculates classification metrics based on adjusted probability threshold
    
    Parameters:
    
    y_true: actual y values
    
    y_pred_prob: predicted probabilities for y 
    
    threshold: enter the adjusted probability threshold for classification (can be a float or integer)
    
    RNN: indicate as True if classifier is an RNN; default value is False
    '''
    #initializes a list to store adjusted y predictions
    y_pred_adjusted = []

    #for loop assigns a predicted class for y based on the threshhold value that was entered 
    if RNN == False:

        for item in y_pred_prob:
            if item[0] <= threshold:
                y_pred_adjusted.append(1)
            else:
                y_pred_adjusted.append(0)
    else:

          for item in y_pred_prob:
            if item[0] >= threshold:
                y_pred_adjusted.append(1)
            else:
                y_pred_adjusted.append(0)
    
    #print metrics according to adjusted threshold
    print('Adjusted Accuracy: ' + str(metrics.accuracy_score(y_true, y_pred_adjusted)))
    print('Adjusted Precision: ' + str(metrics.precision_score(y_true, y_pred_adjusted)))
    print('Adjusted Recall: ' + str(metrics.recall_score(y_true, y_pred_adjusted)))
    print('Adjusted F1 Score: ' + str(metrics.f1_score(y_true, y_pred_adjusted)))
    print('\n')
    
    # calls confusion matrix function 
    cm = seaborn_confusion_matrix(y_true, y_pred_adjusted)
    
    #return list of adjusted y predictions and confusion matrix
    return y_pred_adjusted, cm


# ## Functions to Quickly Compare Different Vectorization Methods and Class Imbalance Corrections:

# ### Single Vectorizer Functions
# ##### Used to examine the performance of a model with a single vectorizer in detail, after doing initial comparisons

# In[10]:


def single_vector_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorizer):
    
    '''
    Apply the specified text vectorizer,make predictions and calculate scores with specified classifier.

    No explicit correction for class imbalances is conducted in this function, 
    but up or downsampled X_train and y_train variables can be passed as arguments

    Parameters:

    X_train_col: specify cleaned text column to be used for vectorization and predictions

    y_train: enter as a one-dimensional vector; function transforms into an array

    X_val_col: specify cleaned text column to be used for vectorization and predictions

    y_val:  enter as a one-dimensional vector; function transforms into an array

    vectorizer: indicate text vectorization method; uses default parameters if none are specified

    classifier: name of classifier; uses default parameters if none are specified

    '''
    
    X_train_transformed = vectorizer.fit_transform(X_train_col)
    X_val_transformed = vectorizer.transform(X_val_col)
            
    fitted_classifier = classifier.fit(X_train_transformed, y_train.values.ravel())
    
    y_train_pred = classifier.predict(X_train_transformed)
    y_val_pred = classifier.predict(X_val_transformed)
    
    y_train_prob = classifier.predict_proba(X_train_transformed)
    y_val_prob = classifier.predict_proba(X_val_transformed)
     
    metrics_dict = compare_classification_metrics(y_train, y_train_pred, y_val, y_val_pred)
        
    pred_df = compare_predictions_dataframe(y_val, y_val_pred, y_val_prob)
                     
    return fitted_classifier, X_val_transformed, pd.DataFrame(y_train_pred), pd.DataFrame(y_val_pred), y_val_prob, metrics_dict, pred_df


# In[ ]:


def smote_vector_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorizer, sample_class = 'not majority'):
    '''
    Returns five values: Transformed X_val, classification metrics dictionary, validation confusion matrix, 
    predicted values for y_val, predicted probabilities for y_val, predictions dataframe
     
    Applies the specified text vectorizer, use SMOTE to rebalance class sizes, 
    and then make predictions and calculate scores for specified classifier

    Parameters:

    X_train_col: specify cleaned text column to be used for vectorization and predictions

    y_train: enter as a one-dimensional vector; function transforms into an array

    X_val_col: specify cleaned text column to be used for vectorization and predictions

    y_val_col:  enter as a one-dimensional vector; function transforms into an array

    vectorizer: indicate text vectorization method; uses default parameters if none are specified

    classifier: name of classifier; uses default parameters if none are specified
    
    sample_class: default is 'not majority'; other choices include: 'minority', 'not minority', 
    'not majority' or 'all'. Can also enter a float from 0 to 1
    
    '''
    
    X_train_transformed = vectorizer.fit_transform(X_train_col)
    X_val_transformed = vectorizer.transform(X_val_col)
    
    smote = SMOTE(random_state=1, sampling_strategy = sample_class)
    
    X_train2, y_train2 = smote.fit_resample(X_train_transformed, y_train)
    
    model = classifier.fit(X_train2, y_train2)
    
#     pipe = make_pipeline(smote, classifier) 
    
#     model = pipe.fit(X_train_transformed, y_train.values.ravel())

    y_train_pred = classifier.predict(X_train2)
    y_val_pred = classifier.predict (X_val_transformed)
    
    y_train_prob = classifier.predict_proba(X_train2)
    y_val_prob = classifier.predict_proba(X_val_transformed)
    
#     y_train_pred = model.predict(X_train_transformed)
#     y_val_pred = model.predict (X_val_transformed)
    
#     y_train_prob = model.predict_proba(X_train_transformed)
#     y_val_prob = model.predict_proba(X_val_transformed)
    
    # print scores  
    metrics_dict = compare_classification_metrics(y_train2, y_train_pred, y_val, y_val_pred)
        
    pred_df = compare_predictions_dataframe(y_val, y_val_pred, y_val_prob)
                     
    return model, X_val_transformed, pd.DataFrame(y_train_pred), pd.DataFrame(y_val_pred), y_val_prob, metrics_dict, pred_df


# In[ ]:


def upsample_vector_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorizer):
    
    '''
    Returns five values: Transformed X_val, classification metrics dictionary, validation confusion matrix, 
    predicted values for y_val, predicted probabilities for y_val, predictions dataframe
     
    Oversamples from the minoriy class (label = 1) until it is the size of the minority class (label = 0)

    Parameters:

    X_train_col: specify cleaned text column to be used for vectorization and predictions

    y_train: enter as a one-dimensional vector; function transforms into an array

    X_val_col: specify cleaned text column to be used for vectorization and predictions

    y_val_col:  enter as a one-dimensional vector; function transforms into an array

    vectorizer: indicate text vectorization method; uses default parameters if none are specified

    classifier: name of classifier; uses default parameters if none are specified

    '''
    #upsample data 
    X_train_col_up, y_train_up = upsample_training_data(X_train_col, y_train)
    
    # perform vectorization
    X_train_up_transformed = vectorizer.fit_transform(X_train_col_up.values.ravel())
    X_val_transformed = vectorizer.transform(X_val_col)
                
    fitted_classifier = classifier.fit(X_train_up_transformed, y_train_up.values.ravel())
    
    #train and validate classifier
    y_train_up_pred = classifier.predict(X_train_up_transformed)
    y_val_pred = classifier.predict(X_val_transformed)
    
    y_train_up_prob = classifier.predict_proba(X_train_up_transformed)
    y_val_prob = classifier.predict_proba(X_val_transformed)
    
    # print scores  
    metrics_dict = compare_classification_metrics(y_train_up, y_train_up_pred, y_val, y_val_pred)
    
#     conf_matrix = print_confusion_matrix(y_val, y_val_pred)
    
    pred_df = compare_predictions_dataframe(y_val, y_val_pred, y_val_prob)
                     
    return fitted_classifier, X_val_transformed, pd.DataFrame(y_train_up_pred), pd.DataFrame(y_val_pred), y_val_prob, metrics_dict, pred_df


# In[ ]:


def downsample_vector_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorizer):
 
    '''
    Downsamples from the majority class (label = 0) until it is the size of the minority class (label = 1)

    Parameters:

    X_train_col: specify cleaned text column to be used for vectorization and predictions

    y_train: enter as a one-dimensional vector; function transforms into an array

    X_val_col: specify cleaned text column to be used for vectorization and predictions

    y_val_col:  enter as a one-dimensional vector; function transforms into an array

    vectorizer: indicate text vectorization method; uses default parameters if none are specified

    classifier: name of classifier; uses default parameters if none are specified

    '''
    #downsample data 
    X_train_col_down, y_train_down = downsample_training_data(X_train_col, y_train)

    #perform vectorization
    X_train_down_transformed = vectorizer.fit_transform(X_train_col_down.values.ravel())
    X_val_transformed = vectorizer.transform(X_val_col.values.ravel())
                
    fitted_classifier = classifier.fit(X_train_down_transformed, y_train_down.values.ravel())
    
    #train and validate classifier
    y_train_down_pred = classifier.predict(X_train_down_transformed)
    y_val_pred = classifier.predict(X_val_transformed)
    
    y_train_down_prob = classifier.predict_proba(X_train_down_transformed)
    y_val_prob = classifier.predict_proba(X_val_transformed)
    
    # print scores  
    metrics_dict = compare_classification_metrics(y_train_down, y_train_down_pred, y_val, y_val_pred)
    
    # conf_matrix = print_confusion_matrix(y_val, y_val_pred.ravel())
    
    pred_df = compare_predictions_dataframe(y_val, y_val_pred, y_val_pred_prob)
                     
    return fitted_classifier, X_val_transformed, pd.DataFrame(y_train_down_pred), 
    pd.DataFrame(y_val_pred), y_val_prob, metrics_dict, pred_df


# ### Multiple Comparison Functions
# 
# ##### Use to compare the performance of a model with multiple vectorization strategies at once.

# In[ ]:


def compare_vectorization_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorization_list):
    '''
    Compares classification model performance using different text vectorizers,
    (declared in 'vectorization list') outside the function.  
        
    Parameters:
    
    X_train_col: cleaned text column in training set
    
    y_train: target variable in training set
    
    X_val_col: cleaned text column in validation set
    
    y_val: target variable in validation set 
    
    classifier: name of classifier; uses default parameters if none are specified
    
    vectorization_list: list of tuples specficifying each name and vectorization method to be used

    '''
    metrics_dict2 = {}
                
    for name, vectorizer in vectorization_list:
                
        X_train_transformed = vectorizer.fit_transform(X_train_col)
        X_val_transformed = vectorizer.transform (X_val_col)

        classifier.fit(X_train_transformed, y_train.values.ravel())
    
        y_train_pred = classifier.predict (X_train_transformed)
        y_val_pred = classifier.predict (X_val_transformed)   
        
        scores = compare_classification_metrics (y_train, y_train_pred, y_val, y_val_pred)
        
        metrics_dict2[name] = scores
        
    return pd.DataFrame(metrics_dict2)


# In[ ]:


def smote_compare_vectorization_model(X_train_col, y_train, X_val_col, y_val, 
                                      classifier, vectorization_list, sample_class = 'not majority'):
  
    '''
    Compares the performance of a single classifier using different text vectorization methods.
    
    Uses SMOTE to correct for class imbalance before fitting classifier. Sampling methods are indicated below.

    Parameters:
    
    X_train_col: cleaned text column in training set
    
    y_train: target variable in training set
    
    X_val_col: cleaned text column in validation set
    
    y_val: target variable in validation set 
    
    classifier: name of classifier; uses default parameters if none are specified
    
    vectorization_list: list of tuples specficifying each name and vectorization method to be used
    
    sample_class: default is 'not majority'; other choices include: 'minority', 'not minority', 
    'not majority' or 'all'. Can also enter a float between 0 to 1

    '''   
    metrics_dict2 = {}
        
    for name, vectorizer in vectorization_list:
              
        X_train_transformed = vectorizer.fit_transform(X_train_col)
        X_val_transformed = vectorizer.transform (X_val_col)
        
        smote = SMOTE(random_state=1, sampling_strategy = sample_class)
        
        smote.fit(X_train_transformed)
    
        pipe = make_pipeline(smote, classifier) 
    
        model = pipe.fit(X_train_transformed, y_train.values.ravel())
    
        y_train_pred = model.predict(X_train_transformed)
        y_val_pred = model.predict (X_val_transformed)
        
        y_train_pred_prob = classifier.predict_proba(X_train_transformed)
        y_val_pred_prob = classifier.predict_proba(X_val_transformed)
    
        scores = compare_classification_metrics (y_train, y_train_pred, y_val, y_val_pred)
        
        metrics_dict2[name] = scores
        
    return pd.DataFrame(metrics_dict2)


# In[14]:


def upsample_compare_vectorization_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorization_list):
    
    '''
    Oversamples from the minority class (label = 1) until it is the size of the minority class (label = 0)
    
    Vectorization list should be identified as a tuple outside the function
    
    Returns a pandas dataframe with metrics from all tests
        
    Parameters:
    
    X_train_col: cleaned text column in training set
    
    y_train: target variable in training set
    
    X_val_col: cleaned text column in validation set
    
    y_val: target variable in validation set 
    
    classifier: name of classifier; uses default parameters if none are specified
    
    vectorization_list: list of tuples specficifying each name and vectorization method to be used

    '''
    X_train_col_up, y_train_up = upsample_training_data(X_train_col, y_train)
    
    metrics_dict2 = {}

    for name, vectorizer in vectorization_list:
    
        # perform vectorization
        X_train_up_transformed = vectorizer.fit_transform(X_train_col_up.values.ravel())
        X_val_transformed = vectorizer.transform(X_val_col)

        classifier.fit(X_train_up_transformed, y_train_up.values.ravel())

        #train and validate classifier
        y_train_up_pred = classifier.predict(X_train_up_transformed)
        y_val_pred = classifier.predict(X_val_transformed)

        y_train_up_pred_prob = classifier.predict_proba(X_train_up_transformed)
        y_val_pred_prob = classifier.predict_proba(X_val_transformed)

        scores = compare_classification_metrics (y_train_up, y_train_up_pred, y_val, y_val_pred)

        metrics_dict2[name] = scores
        
    return pd.DataFrame(metrics_dict2)


# In[ ]:


def downsample_compare_vectorization_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorization_list):
 
    '''
    downsamples from the majority class (label = 0) until it is the size of the minority class (label = 1)

    Vectorization list should be identified as a tuple outside the function
    
    Returns a pandas dataframe with metrics from all tests
        
    Parameters:
    
    X_train_col: cleaned text column in training set
    
    y_train: target variable in training set
    
    X_val_col: cleaned text column in validation set
    
    y_val: target variable in validation set 
    
    classifier: name of classifier; uses default parameters if none are specified
    
    vectorization_list: list of tuples specficifying each name and vectorization method to be used

    '''
   #downsample data 
    X_train_col_down, y_train_down = downsample_training_data(X_train_col, y_train)
    
    metrics_dict2 = {}

    for name, vectorizer in vectorization_list:

        #perform vectorization
        X_train_down_transformed = vectorizer.fit_transform(X_train_col_down.values.ravel())
        X_val_transformed = vectorizer.transform(X_val_col.values.ravel())

        classifier.fit(X_train_down_transformed, y_train_down.values.ravel())

        #train and validate classifier
        y_train_down_pred = classifier.predict(X_train_down_transformed)
        y_val_pred = classifier.predict(X_val_transformed)

        y_train_down_pred_prob = classifier.predict_proba(X_train_down_transformed)
        y_val_pred_prob = classifier.predict_proba(X_val_transformed)

        scores = compare_classification_metrics (y_train_down, y_train_down_pred, y_val, y_val_pred)

        metrics_dict2[name] = scores
        
    return pd.DataFrame(metrics_dict2)


# ## Wrapper Functions

# In[17]:


def wrapper_single_vectorization(X_train_col, y_train, X_val_col, y_val, classifier, 
                                 vectorizer, sampling = None, sample_class = 'not majority'):
    
    '''
    Apply the specified text vectorizer and and then make predictions and calculate scores for specified classifier

    Parameters:

    X_train_col: specify cleaned text column to be used for vectorization and predictions

    y_train: enter as a one-dimensional vector; function transforms into an array

    X_val_col: specify cleaned text column to be used for vectorization and predictions

    y_val_col:  enter as a one-dimensional vector; function transforms into an array

    vectorizer: indicate text vectorization method; uses default parameters if none are specified

    classifier: name of classifier; uses default parameters if none are specified
    
    sample_class: default is 'not majority'

    '''
    
    if sampling is None:
        results = single_vector_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorizer)
    
    elif sampling is 'smote':
        results = smote_vector_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorizer, sample_class)

    elif sampling is 'upsample':
        results = upsample_vector_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorizer)
                        
    elif sampling is 'downsample':
        results = downsample_vector_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorizer)
        
    else:
        results = print('Sampling method does not exist. Indicate: smote, upsample or downsample')
                
    return results
    


# In[6]:


def wrapper_compare_vectorizations(X_train_col, y_train, X_val_col, y_val, classifier, 
                                   vectorization_list, sampling = None, sample_class = 'not majority'):
    '''
    Returns a dataframe with train and validation metrics for multiple vectorizers.

    Parameters:
    
    X_train_col: cleaned text column in training set
    
    y_train: target variable in training set
    
    X_val_col: cleaned text column in validation set
    
    y_val: target variable in validation set 
    
    classifier: name of classifier; uses default parameters if none are specified
    
    vectorization_list: list of tuples specficifying each name and vectorization method to be used

    sampling: optional parameter; choices
    '''
    
    if sampling is None:
        results = compare_vectorization_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorization_list)
        
    elif sampling is 'smote':
        results = smote_compare_vectorization_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorization_list, sample_class)
                        
    elif sampling is 'upsample':
        results = upsample_compare_vectorization_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorization_list)
        
    elif sampling is 'downsample':
        results = downsample_compare_vectorization_model(X_train_col, y_train, X_val_col, y_val, classifier, vectorization_list)

    else:
        results = print('Sampling method does not exist. Indicate: smote, upsample or downsample')
        
    return results


# ### Upsampling and Downsampling

# In[ ]:


def upsample_training_data(X_train_col, y_train):
    
    '''
    draws samples from the minoriy class (label = 1) until it is the size of the minority class (label = 0)

    returns a single upsampled dataframe (with both X_train and y_train in one dataframe)

    '''
    training_data = pd.DataFrame(X_train_col)
    training_data['label']= y_train

    train_0 = training_data[training_data.label==0]
    train_1 = training_data[training_data.label==1]

    train_1_up = resample(train_1, 
          replace=True,    
          n_samples=len(train_0),   
          random_state=10)

    train_upsampled = pd.concat([train_1_up, train_0])
    
    X_train_col_up = train_upsampled.drop(['label'], axis = 1)
    y_train_up = train_upsampled.label
    
    return X_train_col_up, y_train_up


# In[19]:


def downsample_training_data(X_train_col, y_train):
    
    '''
    draws samples from the majority class (label = 0) until it is the size of the minority class (label = 1)

    returns a single downsampled dataframe (with both X_train and y_train in one dataframe)

    '''
    training_data = pd.DataFrame(X_train_col)
    training_data['label']= y_train
    
    train_0 = training_data[training_data.label==0]
    train_1 = training_data[training_data.label==1]

    train_0_down = resample(train_0, 
          replace=True,    
          n_samples=len(train_1),   
          random_state=10)

    train_downsampled = pd.concat([train_0_down, train_1])
    
    X_train_col_down = train_downsampled.drop(['label'], axis = 1)
    y_train_down = train_downsampled.label
    
    return X_train_col_down, y_train_down


# ## Functions for Word Embeddings/Deep Learning (Word2Vec, Glove, RNN)

# In[41]:


def avg_word_vectors(sentence, model_name, dimsize):
    '''
    Calculate the mean word embedding for every sentence in the dataset.
        
    Parameters:
    
    wordlist: list of the list of words in each sentence
    
    size: size of hidden layer
    
    model_name: name of wv2 model
    
    '''

    sumvec = np.zeros(shape = (1, dimsize))
    wordcnt = 0
    
    for word in sentence:
        if word in model_name:
            sumvec += model_name[word]
            wordcnt +=1
    
    if wordcnt == 0:
        return sumvec
    
    else:
        return sumvec / wordcnt


# In[ ]:


def mean_embedding_compare_classification(X_train, y_train, X_val, y_val, classifier_list, 
                                          sampling = None, sample_class = 'not majority'):
    '''
    Function to allow for quick comparison of word2vec and glove word embeddings with each of the classifiers.
        
    Returns a pandas dataframe with the metrics for each classifier
    
    Parameters:
    
    X_train, X_val: mean train and validation word embeddings from word2vec or glove
    
    y_train, y_val: actual y values for train and validation sets
    
    classifier_list: classifiers for comparison. Should be declared outside the function in a tuple 'classifier_list'

    sampling: default is none; options include 'upsample', 'downsample', 'smote'
    
    sample_class: can identify when 'smote' is selected for sampling in previous parameter; default is 'not majority'
    '''  
   
    metrics_dict = {}
    
    for name, classifier in classifier_list:

        if sampling is 'upsample':
        
            X_train, y_train = upsample_training_data(X_train, y_train)
            model = classifier.fit(X_train, y_train.values.ravel())

        elif sampling is 'downsample':

            X_train, y_train = downsample_training_data(X_train, y_train)
            model = classifier.fit(X_train, y_train.values.ravel())

        elif sampling is 'smote':
    
            smote = SMOTE(random_state=1, sampling_strategy = sample_class)
            pipe = make_pipeline(smote, classifier) 
            model = pipe.fit(X_train, y_train.values.ravel())
            
        else:
            
            model = classifier.fit(X_train, y_train)
    
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
    
        metrics_dict[name] = compare_classification_metrics(y_train, y_train_pred, y_val, y_val_pred)
    
    return pd.DataFrame(metrics_dict)


# In[42]:


def pca_smote_w2v_model(X_train_w2v, y_train, X_val_w2v, y_val, classifier):
    
    pca = decomposition.PCA(n_components=10)
    
    smote = SMOTE(random_state=10, sampling_strategy='not majority')
        
    pipe = make_pipeline(pca, smote, classifier) 
    
    model = pipe.fit(X_train_w2v, y_train)
    
    y_train_pred = model.predict(X_train_w2v_2)
    y_val_pred = model.predict (X_val_w2v_2)
    
    metrics_dict = compare_classification_metrics(y_train, y_train_pred, y_val, y_val_pred)
    
    return metrics_dict


# In[11]:


def plot_accuracy_loss (model_title, model_name, y_acc, y_loss):
    '''
    Plots accuracy and loss for RNN classifier
    
    Parameters:
    
    model_title: enter as a string, to be included in graph title
    
    model_name: enter variable name for model
    
    '''
    # Create Figure and Subplots
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), sharex = True)

    # Plot
    ax1.plot(model_name.history.history['acc'], color='blue', label='train accuracy')
    ax1.plot(model_name.history.history['val_acc'], color='green', label='val accuracy')

    ax2.plot(model_name.history.history['loss'], color='blue', label='train loss')
    ax2.plot(model_name.history.history['val_loss'], color = 'green', label= 'val_loss')

    # Title, X and Y labels
    ax1.set_title('Accuracy for {}'.format(model_title)); ax2.set_title('Loss for {}'.format(model_title))
    ax1.set_xlabel('Epochs');  ax2.set_xlabel('Epochs')  
    ax1.set_ylabel('Accuracy');  ax2.set_ylabel('Loss')  
    ax1.set_ylim(y_acc);  ax2.set_ylim(y_loss)  

    ax1.legend(); ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    fig.savefig('results/RNN/figures/Accuracy_Loss_{}'.format(model_title))


# In[1]:


def tsne_plot(model):
    '''
    Plots the word embeddings created by the word2vec model
    
    Parameters:
    
    model: variable name for word2vec model
    
    '''
    
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=250, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

#tnse_plot(model_w2v)

