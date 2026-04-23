#Random Forest Regressor model

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def run_model(X_file, y_file, df_file,  test_size = 0.3, seed = int(42),  max_depth = None):
    
    X =  pd.read_csv(X_file,  header=None)
    y = pd.read_csv(y_file,  header=None).values.ravel()
    df = pd.read_csv(df_file, index_col=False)

    #Makes our test dataset
                           # n_samples=200, n_features=5, noise=20, seed = int(42)
    #X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=seed)
    #split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    #Create, fit and run out model
    model = RandomForestRegressor(max_depth = max_depth, random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    
    
    
    if 'Date' in df : 
       year = df['Date'].to_numpy()
       month = None
        
    else:
        year = None
        month = df['month'].to_numpy()
        
    country = str(df['Country'].to_numpy())
    
    dictionary = {'year': year,
           'month': month,
           'country': country ,
           'actuals' : y,
           'predictions': preds,
           'temperature': X[0].to_numpy()
           }
    # dictionary = {
    #          "year": np.array([1,2]),
    #          "month": np.array([]),   #Leave it empty if year dataset
    #          "country": np.array([]),
    #          "actuals": np.array([]),
    #          "predictions": np.array([]),
    #         "temperature": np.array([])))
    #     }
        
    
    #returns...
    # {
    #     "year": np.array([...]),    Leave it empty if month dataset.
    #     "month": np.array([...]), Leave it empty if year dataset
    #     "country": np.array([...]),
    #     "actuals": np.array([...]),
    #     "predictions": np.array([...],
    #     "temperature": np.array([...])))
    # }
    
    return dictionary


if __name__ == '__main__':
    
    test = run_model('testX.csv', 'testy.csv', 'testdf.csv', max_depth = None)
    