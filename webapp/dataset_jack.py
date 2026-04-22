import numpy as np
import pandas as pd


def to_df(csv_path):
    df = pd.read_csv(csv_path)


    if 'Month' in df:
        df['Year'] = pd.to_datetime(df[['Year','Month']].assign(day=1)).dt.strftime('%m/%Y')
        df = df.drop(columns = 'Month')




    'shaping pesticide data'
    if 'Element' in df:
        df = df.drop(['Domain','Element', 'Item','Unit'], axis = 1)


        'shaping yield data'
        if 'Element Code' in df: #note 'Item' could be a useful column to have
            df = df.drop(['Domain Code','Area Code',
                          'Element Code', 'Item Code','Year Code'], axis = 1)

    df.columns = df.columns.str.replace(' ', '') #gets rid of spaces

    'renaming columns'
    standard_names = {'year': 'Date',
                      'Year': 'Date',
                      'country': 'Country',
                      'avg_temp' : 'Value',
                      'Area': 'Country',
                      'average_rain_fall_mm_per_year': 'Value',
                      'monthly_rainfall_mm' : 'Value'
                      }

    df = df.rename(columns = standard_names)

    're-arranging into standard order country, year , value'
    df = df[['Country', 'Date', 'Value']]




    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')# turn spaces to nan

    return df


def data_NaN(csv_path):
    'origional dataset with NaN values'
    return  to_df(csv_path)



def data_zeros(csv_path):
    'replaces empty spaces with 0s'
    dataframe = to_df(csv_path)
    dataframe = dataframe.fillna(0)
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
    mean = np.mean(means)
    dataframe[column].fillna(mean,inplace = True)
    return dataframe

def data_median_val(csv_path):
    '''replaces NaN with median values (apparently better than mean
     according to a stackoverflow post)'''


    dataframe = to_df(csv_path)
    df_removed_nan = data_remove_rows(csv_path)
    column = pd.isna(dataframe.iloc[:,2]).name
    medians =  df_removed_nan[column].to_numpy()
    median = np.median(medians)
    dataframe[column].fillna(median,inplace = True)
    return dataframe



if __name__ == '__main__':

    path = '../rainfall.csv'

    data_rain = to_df(path)
    data_temp = to_df('../temp.csv')
    data_pest = to_df('../pesticides.csv')
    data_yield = to_df('../yield.csv')

    zero = data_zeros(path)
    removed = data_remove_rows(path)

    mean = data_mean_val(path)

    median = data_median_val(path)


    domi_path = '../Germany_monthly_1950_2013.csv'
    domi_data = to_df(domi_path)



