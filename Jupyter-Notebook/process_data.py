import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from a SQLite database.

    Args:
        database_filepath (str): The file path of the SQLite database.

    Returns:
        X (pd.Series): A pandas Series containing the messages.
        Y (pd.DataFrame): A pandas DataFrame containing the categories.
        category_names (numpy.ndarray): An array of category names.
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    '''
    Clean and preprocess the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'categories' column.

    Returns:
        df (pd.DataFrame): The cleaned DataFrame with individual category columns.
    '''
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0].copy()

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.rsplit('-', n=1).str[0]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # delete rows with strange values
    for col in category_colnames:
        if( ~((df[col].min() == 0) or (df[col].min() == 1)) or ~((df[col].max() == 0) or (df[col].max() == 1))): 
            print("    '{}' has min: {}, max: {}. Delete rows rows with strange values.".format(col, str(df[col].min()), str(df[col].max())))
            df = df.query('(0 <= {}) and ({} <= 1) '.format(col, col)).reset_index(drop=True)
    
    # before save the clean dataset, test the dataset
    for x in row:
        name = x.rsplit('-')[0]
        value = int(x.rsplit('-')[1])
        assert value == categories[name].loc[0], f"    [Error] Assertion failed for '{name}': Expected value {categories[name].loc[0]}, Actual value {value}"

    for col in category_colnames:
        assert df[col].isna().sum() == 0, f"    [Error] Assertion failed for '{col}': NaN values found."
        assert df[col].max() <= 1, f"    [Error] Assertion failed for '{col}': Maximum value exceeds 1, that is {df[col].max()}."
        assert df[col].min() >= 0, f"    [Error] Assertion failed for '{col}': Minimum value is less than 0, that is {df[col].min()}."
        
    assert df.duplicated().sum() == 0, "    [Error] Duplicate rows found in the DataFrame."
    
    print("    ---- Tests at cleaning data process were all passed. ----")

    return df


def save_data(df, database_filename):
    '''
    Save the clean DataFrame as a SQLite database file.

    Args:
        df (pd.DataFrame): The cleaned DataFrame to be saved.
        database_filename (str): The file path of the SQLite database.

    Returns:
        None
    '''
    # save the clean dataframe as a database file
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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