#Random Forest Regressor model

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd


def run_model(X, y, df,  test_size = 0.3, seed = int(42),  max_depth = None):
        
    #Makes our test dataset
                           # n_samples=200, n_features=5, noise=20, seed = int(42)
    #X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=seed)
    #split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    #Create, fit and run out model
    model = RandomForestRegressor(max_depth = max_depth, random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    dictionary = {
            #  "year": np.array([...]),    #Leave it empty if month dataset.
            #  "month": np.array([...]), #Leave it empty if year dataset
            #  "country": np.array([...]),
            #  "actuals": np.array([...]),
            #  "predictions": np.array([...],
            # "temperature": np.array([...])))
        }
        
    
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
    
    X_testfile =  pd.read_csv('testX.csv',  header=None).values
    y_testfile = pd.read_csv('testy.csv',  header=None).values.ravel()
    test = run_model(X = X_testfile, y = y_testfile, max_depth = None)