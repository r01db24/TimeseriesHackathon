#Random Forest Regressor model

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split



def run_model(n_samples=200, n_features=5, noise=20, seed = int(42), max_depth = None):
        
    #Makes our test dataset
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=seed)
    #split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    #Create, fit and run out model
    model = RandomForestRegressor(max_depth = max_depth, random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    return preds
    

if __name__ == '__main__':
    test = run_model(max_depth = None)