    # Read the seed from the query string e.g. /run?seed=42, default to 42 if not provided
    seed = int(request.args.get("seed", 42))

    #Makes our test dataset
    X, y = make_regression(n_samples=200, n_features=5, noise=20, random_state=seed)
    #split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    #Create, fit and run out model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)