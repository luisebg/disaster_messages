import sys
import pandas as pd
import numpy as np
import nltk
import re
import pickle

from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """ Loads the data from the database_filepath database.
    It returns the independent variables (X), the dependent variables (y), and the names of the dependent variables
    
    - Input:
        database_filepath, str with the path of the database to read
        
    - Output:
        X - Pandas dataframe with the dependent variables (messages).
        y - Pandas dataframe with the independent variables (labeled features for the classifier)
        y.columns - list of columns names.
    """
    # Load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    conn = engine.connect()
    # Query table
    df = pd.read_sql_query("SELECT * FROM labeled_messages;", conn)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre','child_alone'])
    y.related.replace(2,1,inplace=True)
    
    return X, y, y.columns


def tokenize(text):
    """ tokenize the given text (sentence). 
    it converts text input in lowercase, remove stop words and reduce que words with lemmatization
    
    - Input:
        text - string with a sequence of words.
        
    - Output:
        lemmed - list of lemmatized words
        
    """
    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.lower().strip()
    text = word_tokenize(text)
    
    # Remove stop words
    words = [word for word in text if word not in stopwords.words("english")]
    
    # Reduce words
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]
    
    return lemmed


def build_model():
    """ Creates the pipeline with three processes:
    - CountVectorizer (count the frequency of words in a corpus)
    - TfidfTransformer (returns the frequency of terms in a corpus)
    - MultiOutputClassifier (creates a classifier for each class)
    
    - Input:
        None
        
    - Output:
        pipeline - sknlearn object with the methods fit and transform
        
    """
    pipeline = Pipeline([
        ('cvect', CountVectorizer(tokenizer=tokenize)),
        ('tdidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """ It prints the model metrics using X_test to generate y_pred and then compare it with y_test
    
    - Input:
        model - sklearn model with the predict method
        X_test - numpy array or pandas dataframe with the testing data
        y_test - numpy array or pandas dataframe with the test labels
        category_names - list of str with the names of the categories
        
    - Output:
        None
    """
    y_pred = model.predict(X_test)
    print('Classification report: {}'.format(classification_report(y_test, y_pred,
                                                                   target_names=category_names)))


def save_model(model, model_filepath):
    """ Save the model into a pickle file
    
    -Input:
        model - sklearn model
        model_filepath - str with the path of the model to save
        
    - Output:
        None
        
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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