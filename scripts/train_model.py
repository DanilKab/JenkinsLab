import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def scale_features(df: pd.DataFrame):
    X = df.drop(columns=['charges'])
    y = df['charges'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    power_trans = PowerTransformer()
    y_scaled = power_trans.fit_transform(y).ravel()

    return X_scaled, y_scaled, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    df = pd.read_csv('insurance_processed.csv')
    X, y, power_trans = scale_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'l1_ratio': [0.0, 0.15, 0.5, 1.0],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'fit_intercept': [True, False]
    }

    mlflow.set_experiment("insurance_charges_prediction")

    with mlflow.start_run() as run:
        sgd = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
        grid = GridSearchCV(sgd, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred_scaled = best_model.predict(X_val)

        y_val_orig = power_trans.inverse_transform(y_val.reshape(-1, 1)).ravel()
        y_pred_orig = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        rmse, mae, r2 = eval_metrics(y_val_orig, y_pred_orig)

        for param, value in grid.best_params_.items():
            mlflow.log_param(param, value)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri()
        model_path = artifact_uri + "/model"


        with open("best_model_uri.txt", "w") as f:
            f.write(model_path)
