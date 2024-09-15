# Import necessary libraries
import sys
import nltk
import re
nltk.download(['punkt', 'wordnet'])
import warnings
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    Load data from the SQLite database
    
    Arguments:
        database_filepath -> Path to the SQLite database
    Output:
        X -> Features (messages)
        y -> Targets (categories)
        category_names -> Names of the target categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = 'disaster_response_table'
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenize text data
    
    Arguments:
        text -> Text message
    Output:
        clean_tokens -> List of tokenized and lemmatized words
    """
    # Remove URLs from the text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline with GridSearchCV
    
    Output:
        model -> GridSearchCV model object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [8, 15],
        'clf__estimator__min_samples_split': [2]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model performance
    
    Arguments:
        model -> Trained model
        X_test -> Test features
        Y_test -> Test target values (DataFrame)
        category_names -> Names of the target categories
    """
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Loop through each category and print evaluation metrics
    for i, category in enumerate(category_names):
        print(f'Category: {category}')
        
        # Evaluate with classification report for each category
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))

        # Print accuracy for each category
        accuracy = accuracy_score(Y_test.iloc[:, i], y_pred[:, i])
        print(f'Accuracy for {category}: {accuracy:.2f}\n')

        
def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file
    
    Arguments:
        model -> Trained model
        model_filepath -> Path to save the pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument.\n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
