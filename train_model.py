import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    df = pd.read_csv("df_clear.csv")
    X = df.drop(columns=['charges'])
    y = df['charges']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    
    # Пайплайн: масштабирование признаков + регрессор, который работает с преобразованной целевой переменной
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', TransformedTargetRegressor(
            regressor=SGDRegressor(random_state=42),
            transformer=PowerTransformer()
        ))
    ])
    
    # Параметры для GridSearchCV (обратите внимание на двойное вложение: model__regressor__...)
    params = {
        'model__regressor__alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'model__regressor__l1_ratio': [0.001, 0.05, 0.01, 0.2],
        'model__regressor__penalty': ['l1', 'l2', 'elasticnet'],
        'model__regressor__loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'model__regressor__fit_intercept': [False, True]
    }
    
    mlflow.set_experiment("insurance_charges_model")
    with mlflow.start_run():
        clf = GridSearchCV(pipe, params, cv=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_
        
        y_pred = best.predict(X_val)   # теперь это уже доллары
        
        (rmse, mae, r2) = eval_metrics(y_val, y_pred)
        
        # Логируем лучшие параметры
        best_params = clf.best_params_
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, "model", signature=signature)
    
    # Выводим путь к модели для deploy
    dfruns = mlflow.search_runs(experiment_names=["insurance_charges_model"])
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://","") + '/model'
    print(path2model)
