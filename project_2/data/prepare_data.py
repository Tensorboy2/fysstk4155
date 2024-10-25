'''The module that gets the data prepared for training and testing'''
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import get_data
data_path = 'project_2/data/kaggle_data/data.csv'

def prepare_data(test_size=0.8):
    '''Prepares the data into a train and load part.

    Args:
        test_size (float): the size of the dataset used for testing.
    
    Returns:
        tuple(x train, x test, y train, y test):

    '''
    if os.path.exists(data_path):
        print(f"Data found at {data_path}.")
    else:
        print(f'No data at {data_path}. Trying to get data through api...')
        get_data()
        if os.path.exists(data_path):
            print(f"Data found at {data_path}.")
        else:
            print('Check api settings or authentication.')
            
    df = pd.read_csv('./project_2/data/kaggle_data/data.csv')
    x = df.drop(['id','diagnosis'],axis=1)
    y = df['diagnosis']

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size)
    return x_train, x_test, y_train, y_test
if __name__ == '__main__':
    prepare_data()
