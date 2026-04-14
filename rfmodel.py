import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Hyperparameter_Tuning_RF")


# Hyperparameter options
n_estimators_list = [50, 100]
max_depth_list = [3, 5]

for n in n_estimators_list:
    for depth in max_depth_list:

        with mlflow.start_run():

            # Training model
            model = RandomForestRegressor(n_estimators=n,max_depth=depth,random_state=42)
            model.fit(X_train, y_train)

            # Prediction
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            # Log parameters + metrics
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("rmse", rmse)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            print(f"Run: n={n}, depth={depth}, RMSE={rmse}")