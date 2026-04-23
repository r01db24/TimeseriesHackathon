import numpy as np
import pandas as pd

def dfs(folder = '..'):
    data_rain = to_df(folder+'/rainfall.csv')
    data_temp = to_df(folder+'/temp.csv')
    data_pest = to_df(folder + '/pesticides.csv')
    data_yield = to_df(folder+ '/yield.csv')
    dataframes = [data_rain, data_temp, data_pest, data_yield]
    return dataframes

def to_df(csv_path):

    df = pd.read_csv(csv_path)

    # 'Shaping Dominiks monthly data'
    # if 'Month' in df:
    #     df['Year'] = pd.to_datetime(df[['Year','Month']].assign(day=1)).dt.strftime('%m/%Y')
    #     df = df.drop(columns = 'Month')


    #'shaping yield data'
    if 'Element Code' in df: #note 'Item' could be a useful column to have
        df = df.drop(['Domain Code','Area Code', 'Domain', 'Unit',
                      'Element Code','Element', 'Item','Item Code','Year Code'], axis = 1)
        df = df.rename(columns = {'Value':'Yield'})

    #'shaping pesticide data'
    if 'Element' in df:
        df = df.drop(['Domain','Element', 'Item','Unit'], axis = 1)
        df = df.rename(columns = {'Value':'Pest'})




    df.columns = df.columns.str.replace(' ', '') #gets rid of spaces

    'renaming columns'
    # standard_names = {'year': 'Date',
    #                   'Year': 'Date',
    #                   'country': 'Country',
    #                   'avg_temp' : 'Value',
    #                   'Area': 'Country',
    #                   'average_rain_fall_mm_per_year': 'Value',
    #                   'monthly_rainfall_mm' : 'Value'
    #                   }

    standard_names = {'year': 'Date',
                      'Year': 'Date',
                      'country': 'Country',
                      'Area': 'Country',
                      'average_rain_fall_mm_per_year': 'avg_rain',
                      }


    df = df.rename(columns = standard_names)

    're-arranging into standard order country, year , value'
    value_name = df.iloc[:,2].name
    df = df[['Country', 'Date',value_name]]




    df[value_name] = pd.to_numeric(df[value_name], errors='coerce')# turn spaces to nan

    return df



def align(dataframes):
    'list of dataframes to align and reseperate'

    df_merged = dataframes[0]
    for i in range(1, len(dataframes)):
        df_merged = pd.merge(df_merged, dataframes[i], on=["Date", "Country"], how="outer")


    return df_merged



    "df_merged = pd.merge(\n",
    "    df_temp, \n",
    "    df_rain, \n",
    "    on=['year', 'month', 'country'], \n",
    "    how='inner'\n",












def data_NaN(folder = '..', value = ''):
    'origional datasets with NaN values'
    df_list = dfs(folder = folder)
    dataframe = align(df_list)

    if value == 'rain':
        dataframe = dataframe['avg_rain'].values

    elif value == 'temp':
        dataframe = dataframe['avg_temp'].values

    elif value == 'pest':
        dataframe = dataframe['Pest'].values

    elif value == 'yield':
        dataframe = dataframe['Yield'].values


    else:
        pass

    return  dataframe



def data_zeros(folder = '..', value = ''):
    'replaces empty spaces with 0s'

    df_list = dfs(folder = folder)
    dataframe = align(df_list)
    dataframe = dataframe.fillna(0)

    if value == 'rain':
        dataframe = dataframe['avg_rain'].values

    elif value == 'temp':
        dataframe = dataframe['avg_temp'].values

    elif value == 'pest':
        dataframe = dataframe['Pest'].values

    elif value == 'yield':
        dataframe = dataframe['Yield'].values


    else:
        pass

    return  dataframe



def data_remove_rows(folder = '..', value = ''):
    'removes rows with missing data'
    df_list = dfs(folder = folder)
    dataframe = align(df_list)
    dataframe = dataframe.dropna(axis = 'rows')

    if value == 'rain':
        dataframe = dataframe['avg_rain'].values

    elif value == 'temp':
        dataframe = dataframe['avg_temp'].values

    elif value == 'pest':
        dataframe = dataframe['Pest'].values

    elif value == 'yield':
        dataframe = dataframe['Yield'].values

    else:
        pass

    return dataframe


def data_mean_val(folder = '..', value = ''):
    'replaces NaN with mean values'
    df_list = dfs(folder = folder)
    dataframe = align(df_list)
    dataframe = dataframe.fillna(dataframe.mean(numeric_only=True))

    if value == 'rain':
        dataframe = dataframe['avg_rain'].values

    elif value == 'temp':
        dataframe = dataframe['avg_temp'].values

    elif value == 'pest':
        dataframe = dataframe['Pest'].values

    elif value == 'yield':
        dataframe = dataframe['Yield'].values

    else:
        pass

    return dataframe

def data_median_val(folder = '..', value = ''):
    '''replaces NaN with median values (apparently better than mean
     according to a stackoverflow post)'''

    df_list = dfs(folder = folder)
    dataframe = align(df_list)
    dataframe = dataframe.fillna(dataframe.median(numeric_only=True))

    if value == 'rain':
        dataframe = dataframe['avg_rain'].values

    elif value == 'temp':
        dataframe = dataframe['avg_temp'].values

    elif value == 'pest':
        dataframe = dataframe['Pest'].values

    elif value == 'yield':
        dataframe = dataframe['Yield'].values

    else:
        pass

    return dataframe



if __name__ == '__main__':
    
    st = ['rain','temp','pest','yield']
    
    zeros = []
    remove = []
    mean = []
    median = []
    for i in range (len(st)):
       zeros.append(data_zeros(value = st[i]))
       file = f'../zeros/{st[i]}_zeros.csv'
       np.savetxt(file, zeros[i])
       
       remove.append(data_remove_rows(value = st[i]))
       file_r = f'../remove/{st[i]}_remove.csv'
       np.savetxt(file_r, remove[i])
       
       mean.append(data_mean_val(value = st[i]))
       file_r = f'../mean/{st[i]}_mean.csv'
       np.savetxt(file_r, mean[i])
       
       median.append(data_median_val(value = st[i]))
       file_r = f'../median/{st[i]}_median.csv'
       np.savetxt(file_r, median[i])
       
       

       
       
        
        
        


    dataframes_aligned = align(dfs())

    nan_df = data_NaN(value = 'rain')

    zero = data_zeros(value = 'temp')
    removed = data_remove_rows()

    mean = data_mean_val(value = 'yield')

    median = data_median_val(value = 'pest')


