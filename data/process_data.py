import sys
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt


def load_data(messages_filepath, categories_filepath):

    """
    input: two csv files
    outputs: both files merged as a dataframe
    """

    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    df_merged = pd.merge(df_messages, df_categories, on='id')
    return df_merged

def clean_data(df, column_name):

    """
    input: merged dataframe and colum_name(string)
    output: cleaned dataset
    """
    categories = df[column_name].str.split(';', expand=True)
    categories.columns = categories.iloc[1]

    row = categories.iloc[1]

    category_colnames = row.apply(lambda x: x[:-2])
    print(category_colnames)

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df.drop(column_name, axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    input: cleaned pandas dataframe
    output: SQLite database
    """
    from sqlalchemy import create_engine
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('Disaster', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, 'categories')
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()