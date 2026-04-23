from ast import mod
import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split
# from dataset_jack import 

import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TODO： Wait for this entrypoint to be implemented in model_?.py and then import it here
from model_jack import run_model as run_model_jack

app = Flask(__name__)

# !!! Absolute path to the project root folder.

#
# !!! The temporary CSV files are expected to be grouped by dataset first, then by preprocessor:
#
#   TimeseriesHackathon/datasets/year/remove_na/dataset.csv
#   TimeseriesHackathon/datasets/year/remove_na/X.csv
#   TimeseriesHackathon/datasets/year/remove_na/y.csv
#
#   TimeseriesHackathon/datasets/month/remove_na/dataset.csv
#   TimeseriesHackathon/datasets/month/remove_na/X.csv
#   TimeseriesHackathon/datasets/month/remove_na/y.csv
#

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# This route "/" is the default page, this will run when the browerser runs the URL
@app.route("/")
def index():
    return render_template("index.html")

# This is called by a acript in the index.html
@app.route("/run")
def handel_request():

    # Get those params from the frontend.
    #
    # Example:
    #   /run?dataset=Year%20Dataset&preprocessor=Remove%20NA&model=Jack%20Model
    #
    dataset = request.args.get("dataset", "Year Dataset") # we have year and month dataset
    model = request.args.get("model", "Jack Model")
    preprocessor = request.args.get("preprocessor", "Remove NA") # Just remove NA for now

    # Dataset switch.
    if dataset is "Yearly":
        dataset = "Year Dataset"
    elif dataset is "Monthly":
        dataset = "Month Dataset"
        
    if dataset is "Year Dataset":
        dataset_folder = "year"
    elif dataset is "Month Dataset":
        dataset_folder = "month"
    else:
        # Unknown dataset return HTTP 400.
        return jsonify({
            "error": f"OOOOOOOOOOOOOOOps! Unsupported dataset????: {dataset}"
        }), 400

    # Preprocessor switch.
    # Every dataset + preprocessor combination has exactly three CSV files:
    #
    #   dataset.csv = the full dataset, including metadata columns such as
    #                 year, month, and country.
    #
    #   X.csv       = the model input features for this dataset/preprocessor.
    #
    #   y.csv       = the target labels for this dataset/preprocessor.
    if preprocessor == "Remove NA":
        preprocessor_folder = "remove_na"
    else:
        # Unknown preprocessor name return HTTP 400.
        return jsonify({
            "error": f"OOOOOOOOOOOOOOOps! Unsupported preprocessor????: {preprocessor}"
        }), 400

    # Build the three CSV path.
    csv_folder = os.path.join(BASE_DIR, "datasets", dataset_folder, preprocessor_folder)
    dataset_path = os.path.join(csv_folder, "dataset.csv")
    x_path = os.path.join(csv_folder, "X.csv")
    y_path = os.path.join(csv_folder, "y.csv")

    # Model switch.
    #
    if model == "Jack Model":
        run_model = run_model_jack
    elif model == "Test Model":
        # ! Hey, this is not implemented yet.
        return jsonify({
            "error": "Test Model is not implemented yet"
        }), 501
    else:
        # Unknown model name return HTTP 400.
        return jsonify({
            "error": f"OOOOOOOOOOOOOOOps! Unsupported model????: {model}"
        }), 400

    # Load the three CSV files selected by dataset + preprocessor.
    df = pd.read_csv(dataset_path)
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    # !!! Every model should expose the same entrypoint shape, PLEASE!
    #
    #   run_model(X: pd.DataFrame, y: pd.DataFrame, df: pd.DataFrame) -> dict
    #
    model_output = run_model(X, y, df)

    # Expected model output:
    #
    # {
    #     "year": np.array([...]),
    #     "month": np.array([...]),
    #     "country": np.array([...]),
    #     "actuals": np.array([...]),
    #     "predictions": np.array([...])
    # }
    #
    actuals = np.asarray(model_output["actuals"])
    predictions = np.asarray(model_output["predictions"])

    # MAE
    # RMSE
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    if dataset == "Year Dataset":
        model_output["month"] = np.array([None] * len(model_output["predictions"]))
    elif dataset == "Month Dataset":
        model_output["year"] = np.array([None] * len(model_output["predictions"]))
        
    return jsonify({
        "dataset": dataset,
        "model_name": model,
        "preprocessor": preprocessor,
        "year": np.asarray(model_output["year"]).tolist(),
        "month": np.asarray(model_output["month"]).tolist(),
        "country": np.asarray(model_output["country"]).tolist(),
        "actuals": actuals.tolist(),
        "predictions": predictions.tolist(),
        "temperature": np.asarray(model_output["temperature"]).tolist(),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
        },
    })

# Start the server if this file is run directly (python app.py)
if __name__ == "__main__":
    # host="0.0.0.0":
    # listen on all network interfaces without it Flask only listens inside the container 
    # and your browser can't reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
