import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from sqlalchemy import create_engine

from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

def load_data(database_filepath):
    '''
    Load data from a SQLite database.

    Args:
        database_filepath (str): The file path of the SQLite database.

    Returns:
        X (pd.Series): A pandas Series containing the messages.
        Y (pd.DataFrame): A pandas DataFrame containing the categories.
        category_names (numpy.ndarray): An array of category names.
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name="DisasterResponse", con=engine)
    X = df["message"]
    Y = df[df.columns[4:]]
    category_names = df[df.columns[4:]].columns.values
    return X, Y, category_names

def tokenize(text):    
    '''
    Tokenize the input text.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        clean_tokens (list): A list of cleaned and lemmatized tokens.
    '''
    # get list of all urls using regex
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # convert to lowercase
    text = text.lower()
    
    # remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    '''
    Build and return a machine learning model pipeline.

    Args:
        None

    Returns:
        cv (GridSearchCV): A grid search object for model tuning.
    '''
    # build a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_features=10)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, n_jobs=-1, verbose=3)))
    ])
    
    # specify parameters for grid search
    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_features': [5, 100], 
        'clf__estimator__n_estimators': [50, 100],
        # 'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1, cv=2)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the performance of the model on the test data.

    Args:
        model: The trained machine learning model.
        X_test (pd.Series): Test features.
        Y_test (pd.DataFrame): True labels for the test data.
        category_names (numpy.ndarray): An array of category names.

    Returns:
        None
    '''
    # predict test labels
    Y_test_pred = model.predict(X_test)

    # print classification report on test data
    print(classification_report(Y_test.values, Y_test_pred, target_names=category_names))

def save_model(model, model_filepath):
    '''
    Save the trained model to a file using pickle.

    Args:
        model: The trained machine learning model.
        model_filepath (str): The file path to save the model.

    Returns:
        None
    '''
    # save the model
    with open(model_filepath, mode="wb") as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()