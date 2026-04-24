import numpy as np
import pandas as pd
import sklearn
from pandas import read_csv
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_regression
from numpy import asarray
from pandas import concat

# X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# print(X.shape, y.shape)


# df['Date'] = pd.to_datetime(df['Date'])
# df['year'] = df['Date'].dt.year
# df['month'] = df['Date'].dt.month
# df['quarter'] = df['Date'].dt.quarter


def series_to_supervised_rainfall(df, n_in=6, dropnan=True):
    df = df.copy()
    df = df[["Date", "avg_rain", "avg_temp", "Pest"]]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cols = []

    for i in range(n_in, 0, -1):
        shifted = df.shift(i)
        shifted.columns = [f"{col}(t-{i})" for col in df.columns]
        cols.append(shifted)

    current_X = df[["Date", "avg_temp", "Pest"]].copy()
    current_X.columns = [f"{col}(t)" for col in current_X.columns]
    cols.append(current_X)

    target = df[["avg_rain"]].copy()
    target.columns = ["target_avg_rain"]
    cols.append(target)

    supervised = pd.concat(cols, axis=1)

    if dropnan:
        #supervised.dropna(inplace=True)
        supervised = supervised.dropna()

    return supervised


# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = DataFrame(data)
#     cols = list()
#
#     ### input sequence here
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#     ### forecast sequence (t, t+1 ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#     ### putting it together
#     agg = concat(cols, axis=1)
#
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg.values


# split a univariate dataset into train/test sets
def local_train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

def random_forest_forecast(train, testX):
    train = asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(trainX, trainy)
    #yhat = model.predict(testX)
    #model.fit(trainX, trainy)
    yhat = model.predict([testX])
    return yhat[0]

def walk_forward_validation(data, n_test):
    #predictions = list()
    predictions = []
    train, test = local_train_test_split(data, n_test)
    history = [x for x in train]

    for i in range(len(test)):
        testX, testy = test[i, :-1], test[i, -1]
        yhat = random_forest_forecast(history, testX)
        predictions.append(yhat)
        history.append(test[i])
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, :-1], predictions


series = pd.read_csv("testdf.csv")
country_name = "United Kingdom"
series = series[series["Country"] == country_name].copy()
series["Date"] = pd.to_numeric(series["Date"], errors="coerce")
series = series.sort_values("Date").reset_index(drop=True)
#series = read_csv('testdf.csv', header=0, index_col=0)

#values = series.values
print(f"Country: {country_name}")
print(f"Rows: {len(series)}")
print(series.head())


supervised_df = series_to_supervised_rainfall(series, n_in=6)
data = supervised_df.values ####################################
print("Supervised shape:", supervised_df.shape)
print("Last column name should be target:", supervised_df.columns[-1])

#data = series_to_supervised_rainfall(values, n_in=6)
mae, y, yhat = walk_forward_validation(data, 12)
print("type(y):", type(y))
print("type(yhat):", type(yhat))
print("shape(y):", np.shape(y))
print("shape(yhat):", np.shape(yhat))
print("y first 20:", y[:20] if len(np.shape(y)) == 1 else y[:5])
print("yhat first 20:", yhat[:20] if len(np.shape(yhat)) == 1 else yhat[:5])
print('MAE: %.3f' % mae)

### Flatten for plot
# y = np.asarray(y).ravel()
# yhat = np.asarray(yhat).ravel()
y = np.asarray(y, dtype=float)
yhat = np.asarray(yhat, dtype=float)

print("final shape(y):", y.shape)
print("final shape(yhat):", yhat.shape)
##Temporary?

print("shape(y):", y.shape)
print("shape(yhat):", yhat.shape)
print("y:", y)
print("yhat:", yhat)


assert y.ndim == 1, f"Expected y to be 1-D, got shape {y.shape}"
assert yhat.ndim == 1, f"Expected yhat to be 1-D, got shape {yhat.shape}"
assert len(y) == len(yhat), f"Length mismatch: y={len(y)}, yhat={len(yhat)}"

pyplot.figure(figsize=(12, 5))
pyplot.plot(y, label='Expected', marker='o')
pyplot.plot(yhat, label='Predicted', marker='x')###
pyplot.title(f"Walk-Forward Forecast: avg_rain ({country_name})")
pyplot.xlabel("Test step")
pyplot.ylabel("avg_rain")###
pyplot.legend()
#pyplot.plot(values)
pyplot.grid(True, alpha=0.3)
pyplot.tight_layout()
pyplot.show()

#train = series_to_supervised_rainfall(values, n_in=6)
# split into input and output columns
train = data
trainX, trainy = train[:, :-1], train[:, -1]
# fit model
#model = RandomForestRegressor(n_estimators=1000)
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(trainX, trainy)
# construct an input for a new prediction
#
#row = values[-6:].flatten()
# make a one-step prediction
# yhat = model.predict(asarray([row]))
# print('Input: %s, Predicted: %.3f' % (row, yhat[0]))


last_feature_row = train[-1, :-1]
yhat_last = model.predict(asarray([last_feature_row]))

print("Predicted rainfall from last available feature row: %.3f" % yhat_last[0])


###Bonus for later
# all_supervised = []
#
# for country, group in df.groupby("Country"):
#     group = group.sort_values("Date").reset_index(drop=True)
#     if len(group) > 6:
#         temp = series_to_supervised_rainfall(group, n_in=6)
#         temp["Country"] = country
#         all_supervised.append(temp)
#
# supervised_df = pd.concat(all_supervised, ignore_index=True)