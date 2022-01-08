import sys
import pandas as pd
import numpy as np
import nltk
import re
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def transform_categories(dataframe):
    """ proprocess categories.csv file. It defines the column names and values based on the 'categories' column.
    
    - Input
        dataframe - Pandas dataframe with the categoriesinformation.
        
    - Output
        preprocessed_df - Pandas dataframe with the categories in the right format.
        
    """
    preprocessed_df=dataframe['categories'].str.split(";", expand=True)
    # Get columns names
    col_names=preprocessed_df.iloc[0].apply(lambda x: x.split("-")[0]).tolist()
    # Filter each column to only have a numerical value in column values
    for col in preprocessed_df:
        preprocessed_df[col] = preprocessed_df[col].apply(lambda x: x.split("-")[1]).astype(np.int) 
    # set columns names
    preprocessed_df.columns=col_names
    return preprocessed_df

def load_data(messages_filepath, categories_filepath):
    """ load the data for messages and categories. It calls transform_categories function to clean categories dataframe.
    
    - Input
        messages_filepath - str with the path of disaster_messages.csv file.
        categories_filepath - str with the path of disaster_categories.csv file.
        
    - Output
        df - Pandas dataframe with the meessages and categories data.
        
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    
    # Transform
    categories_df = transform_categories(categories_df)
    df = pd.concat([messages_df, categories_df], axis=1)
    return df

def clean_data(df):
    """ Eliminate duplicated rows from df.
    
    - Input
        df - Pandas dataframe with duplicated rows.
        
    - Output
        df - Pandas dataframe without duplicated rows.
        
    """
    return df[~df.duplicated()]

def save_data(df, database_filename, table_name):
    """ export the columns to an sqlite database. If table already exists it replaces with the new one.
    
    - Input
        df - Pandas dataframe to export into a database.
        database_filename - str with the name of the database where data is exported.
        table_name - str with the name of the table to save
        
    - Output
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(f'{table_name}', engine, index=False, if_exists='replace')  

def tokenize(text):
    """ tokenize the given text (sentence). 
    it converts text input in lowercase, remove stop words and reduce que words with lemmatization
    
    - Input:
        text - string with a sequence of words.
        
    - Output:
        clean_tokens - list of lemmatized words
        
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def count_frequent_words(df, n):
    """ count the top n most frequent words from df['message'].
    
    - Input:
        df - pandas dataframe with the 'message' column
        n - int number of top n words to return
        
    - Output:
        to_return - pandas datafram with the top n words and its frequency
        
    """
    words, frequency = [], []
    # Create counter
    term_count = CountVectorizer(tokenizer=tokenize)
    analized_tokens = term_count.fit_transform(df['message'])
    # Get the name of the words used at transformation
    features=term_count.get_feature_names()
    # Get the matrix of frequency
    words_freq=analized_tokens.toarray()
    # Sum over axis 0 to get que number of times a word is used
    global_freq=words_freq.sum(axis=0)
    # put the most frequent words at the end of sorted_freq
    sorted_freq=np.sort(global_freq)
    top=sorted_freq[-n:]
    for freq in top:
        words.append(features[np.where(global_freq==freq)[0][0]])
        frequency.append(freq)
    # Save results in a Pandas dataframe
    to_return = pd.DataFrame(data={'Word':words,'Frequency':frequency})
    return to_return

def count_messages_per_category(df):
    """ Count the number of messages per category in df
    
    - Input:
        df - pandas dataframe with the messages and categories information
        
    - Output:
        to_return - pandas dataframe with the number of messages per category
        
    """
    
    categories_df=pd.melt(df,id_vars=['id', 'original','message', 'genre'],
                          var_name='Category',value_name='Category true')
    categories_df = categories_df[categories_df['Category true'] == 1]
    freq=categories_df.groupby('Category')['id'].count()
    to_return = pd.DataFrame(data={'Category':freq.index.tolist(),
                                   'Frequency':freq.tolist()})
    return to_return

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, 'labeled_messages')
        
        print('Cleaned data saved to database!')
        
        # Find the most common terms in all messages.
        print('Calculating extra metrics...\n')
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(count_frequent_words(df,100), database_filepath, 'common_terms')
        print('Common terms saved to database!')
        
        # Count the number of messages per category
        print('Calculating extra metrics...\n')
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(count_messages_per_category(df), database_filepath, 'categories')
        print('Number of messages per category saved to database!')
        
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()