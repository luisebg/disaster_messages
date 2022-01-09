import json
import plotly
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('labeled_messages', engine)
words_freq = pd.read_sql_table('common_terms', engine)
words_freq.sort_values('Frequency', inplace=True, ascending=False)
cat_freq = pd.read_sql_table('categories', engine)
cat_freq.sort_values('Frequency', inplace=True)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # Visual 2
    words = words_freq['Word'].iloc[0:10].tolist()
    words_count = words_freq['Frequency'].iloc[0:10].tolist()
    
    # Visual 3
    cat = cat_freq['Category'].tolist()
    cat_count = cat_freq['Frequency'].tolist()
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data':[
                Bar(
                    x=words,
                    y=words_count
                )
            ],
            'layout': {
                'title': 'Top 10 of the most frequent words in messages',
                'yaxis': {
                    'title':'Count'
                },
                'xaxis':{
                    'title':'Word'
                }
            }
        },
        {
            'data':[
                Bar(
                    x=cat,
                    y=cat_count
                )
            ],
            'layout': {
                'title': 'Number of messages per categories',
                'yaxis': {
                    'title':'Count'
                },
                'xaxis':{
                    'title':'Category'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()