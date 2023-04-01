
# import necessary libraries for data manipulation and ML modeling
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
import nltk
import pickle

# download nltk punkt and wordnet packages
nltk.download('punkt')
nltk.download('wordnet')

"""
Loads data from the provided database path and returns X, Y, and category names

Parameters:
    database_filepath (str): File path of the SQLlite database

Returns:
    X (Series): Feature data
    Y (DataFrame): Target data
    category_colnames (list): List of category column names
"""
def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('YourTableName', con=engine)
    df[["related", "aid_related", "infrastructure_related", "weather_related"]] = df[["related", "aid_related", "infrastructure_related", "weather_related"]].astype(np.int64)
    X=df['message']
    Y = df.iloc[:,4:]
    category_colnames=list(df.columns[4:])
    return X,Y, category_colnames

"""
Tokenizes text data

Parameters:
    text (str): Text data

Returns:
    clean_tokens (list): List of tokenized words
"""
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

"""
Builds and returns ML pipeline

Returns:
    cv (GridSearchCV): Grid search cross validation object
"""
def build_model():
    clf = RandomForestClassifier(random_state=0)
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multi_clf', MultiOutputClassifier(clf))
    ])

    parameters = {
        'tfidf__use_idf': (True, False)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

"""
Evaluates model performance and prints accuracy

Parameters:
    model (GridSearchCV): Grid search cross validation object
    X_test (Series): Test feature data
    Y_test (DataFrame): Test target data
    category_names (list): List of category column names
"""
def evaluate_model(model, X_test, Y_test, category_names):
    # model.fit(X_train, y_train)

# Make predictions on the test set
    predictions = model.predict(X_test)

# Calculate the accuracy of the model
    accuracy = grid_search.score(X_test, Y_test)

    print('Model accuracy:', accuracy)
    print(category_names)

"""
Saves model as a pickle file

Parameters:
    model (GridSearchCV): Grid search cross validation object
    model_filepath (str): File path of saved model
"""
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))



"""
Main function for training and saving ML model
"""
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
