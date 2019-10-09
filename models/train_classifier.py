import re
import numpy as np
import pandas as pd
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet'])
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import pickle

def load_data(database_filepath):
    
    """
    input: created SQLite database in data/process_data.py
    output: database content in form of a dataframe and split up in X and Y value
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df =  pd.read_sql_table('Disaster', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):

    """
    input: original message text
    output: Tokenized, cleaned, and lemmatized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Input: None
    Output: Results of GridSearchCV (gs)
    """
    from sklearn.multioutput import MultiOutputClassifier
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
    ])

    parameters = {'clf__estimator__max_depth': [10, 50, None],
             'clf__estimator__min_samples_leaf': [2, 5, 10]
             }

    gs = GridSearchCV(pipeline, parameters)

    return gs

def evaluate_model(model, X_test, Y_test, category_names):
    """
    input: model, X_test, y_test, category_names, list of category strings
    output: Print accuracy and classfication report for each category
    """
    y_pred = model.predict(X_test)
    df_y_pred = pd.DataFrame(y_pred, columns = Y_test.columns)
    for column in category_names:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test[column], df_y_pred[column]))

def save_model(model, model_filepath):
    """
    input: model, model_filepath
    output: A pickle file of saved model
    """
    import pickle
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