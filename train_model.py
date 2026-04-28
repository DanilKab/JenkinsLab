import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib

def eval_metrics(actual, pred):
    """Метрики в исходной шкале (доллары)."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    # 1. Загрузка очищенных данных
    df = pd.read_csv("./df_clear.csv")

    # 2. Разделение на признаки и целевую переменную
    X = df.drop(columns=['charges'])
    y = df['charges']   # ИСХОДНЫЕ значения, доллары

    # 3. Масштабирование только признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Разделение на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # 5. Сетка гиперпараметров для SGDRegressor
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.1, 0.2],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'fit_intercept': [False, True]
    }

    # 6. Настройка эксперимента MLflow
    mlflow.set_experiment("insurance_charges_model")

    with mlflow.start_run() as run:
        # 6.1. Базовый регрессор + поиск по сетке
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)

        # 6.2. Обёртка TransformedTargetRegressor
        #      Она сама внутри применит PowerTransformer к y,
        #      а при predict() вернёт значения в долларах.
        model = TransformedTargetRegressor(
            regressor=clf,
            transformer=PowerTransformer()
        )

        # 6.3. Обучение на исходных y_train
        model.fit(X_train, y_train)

        # 6.4. Предсказания (уже в исходной шкале!)
        y_pred = model.predict(X_val)

        # 6.5. Метрики в долларах
        rmse, mae, r2 = eval_metrics(y_val, y_pred)

        # 6.6. Логирование параметров (лучшие из GridSearchCV)
        best_params = model.regressor_.best_params_
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # 6.7. Сохранение ОБЁРТКИ (TransformedTargetRegressor) в MLflow
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # 6.8. Запись пути к модели в файл для задачи Deploy
        artifact_uri = run.info.artifact_uri.replace("file://", "") + "/model"
        with open("best_model.txt", "w") as f:
            f.write(artifact_uri)

        # Вывод для логов Jenkins
        print(f"Model saved at: {artifact_uri}")
        print(f"Metrics – RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        print("Example predictions (first 5):", model.predict(X_val)[:5])
