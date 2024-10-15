'''The module that gets the data from kaggle'''
import os
import kaggle
import pandas as pd

def get_data():
    '''Gets the breast cancer data from kaggle through api'''
    dataset = 'uciml/breast-cancer-wisconsin-data'
    download_dir = './project_2/data/kaggle_data'
    os.makedirs(download_dir, exist_ok=True)
    kaggle.api.dataset_download_files(dataset, path=download_dir, unzip=True)
    csv_file_path = os.path.join(download_dir, 'data.csv')
    pd.read_csv(csv_file_path)

if __name__ == '__main__':
    get_data()
