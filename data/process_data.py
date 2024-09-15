# Import all the relevant libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined data frame containing messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df 

def clean_data(df):
    """
    Clean Categories Data Function
    
    Arguments:
        df -> Data Frame
    Outputs:
        df -> Combined data containing messages and categories with categories cleaned up
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Select the first row of the categories dataframe to extract column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
    
    # Replace categories column in df with new category columns
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Save Data to SQLite Database
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    table_name = 'disaster_response_table'
    # Save the DataFrame to the SQLite database
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
