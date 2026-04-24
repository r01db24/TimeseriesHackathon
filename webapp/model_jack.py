#Random Forest Regressor model

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def run_model(X, y, df,  test_size = 0.3, seed = int(42),  max_depth = None):
    X = X.copy()
    y = y.squeeze("columns").to_numpy()
    df = df.copy()

    #Makes our test dataset
                           # n_samples=200, n_features=5, noise=20, seed = int(42)
    #X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=seed)
    #split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    test_indices = X_test.index
    
    #Create, fit and run out model
    model = RandomForestRegressor(max_depth = max_depth, random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    
    
    
    if len(df) == len(X):
        df_test = df.loc[test_indices]
    else:
        # Some current X/y files are not row-aligned with dataset.csv.
        # Use a positional fallback so the backend can still return metadata.
        df_test = df.iloc[np.asarray(test_indices) % len(df)]

    if 'Date' in df.columns: 
       year = df_test['Date'].to_numpy()
       month = np.array([None] * len(preds), dtype=object)
        
    else:
        year = np.array([None] * len(preds), dtype=object)
        month = df_test['month'].to_numpy()
        
    country = df_test['Country'].to_numpy()
    
    dictionary = {'year': year,
           'month': month,
           'country': country ,
           'actuals' : y_test,
           'predictions': preds,     
           'temperature': X_test.iloc[:, 0].to_numpy()
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
    test_X = pd.read_csv('testX.csv')
    test_y = pd.read_csv('testy.csv')
    test_df = pd.read_csv('testdf.csv')
    test = run_model(test_X, test_y, test_df, max_depth = None)
    
