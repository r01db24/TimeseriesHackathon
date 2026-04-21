import numpy as np
import pandas as pd


def to_df(csv_path):
    df = pd.read_csv(csv_path)
    #df.iloc[2] = df.iloc[2]
    return df

def data_NaN(csv_path):
    'origional dataset with NaN values'
    return  to_df(csv_path)



def data_zeros(csv_path):
    'replaces empty spaces with 0s'
    dataframe = to_df(csv_path)
    dataframe[dataframe.isna()] = 0
    return dataframe



def data_remove_rows(csv_path):
    'removes rows with missing data'
    dataframe = to_df(csv_path)
    dataframe = dataframe.dropna(axis = 'rows')
    return dataframe


def data_mean_val(csv_path):
    'replaces NaN with mean values'
    dataframe = to_df(csv_path)
    df_removed_nan = data_remove_rows(csv_path)
    column = pd.isna(dataframe.iloc[:,2]).name
    means =  df_removed_nan[column].to_numpy()
    means = np.float(means)
    mean = np.mean(means)
    dataframe[column].fillna(mean,inplace = True)
    return dataframe



if __name__ == '__main__':
    
    path = '../rainfall.csv'
    
    data = to_df(path)
    
    zero = data_zeros(path)
    removed = data_remove_rows(path)
    
    mean = data_mean_val(path)