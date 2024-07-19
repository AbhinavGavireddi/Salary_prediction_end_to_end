import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor
import optuna
import pickle
from mlflow.models import infer_signature
from utilities import experiment_name, artifacts, logger, config

try:
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
except Exception as e:
    experiment_id = mlflow.create_experiment(experiment_name)

# enabling autologging
mlflow.sklearn.autolog()

# reading the training dataset
dataset = pd.read_csv(
    artifacts / "dataset.csv",
    low_memory=False,
).drop_duplicates()

for col in config.cat_cols:
    dataset[col] = pd.Categorical(dataset[col])

for col in config.target_col:
    dataset[col] = pd.to_numeric(dataset[col])

# feature engineeering
dataset = dataset[config.cat_cols + config.target_col]

dataset["experience_level"] = (
    dataset["experience_level"].map(config.experience_level_map).astype("int64")
)
dataset["company_size"] = dataset["company_size"].map(config.company_size_map).astype("int64")
dataset['job_title'] = dataset['job_title'].apply(lambda x : x if x in config.req_job_titles else 'Others')
dataset['employee_residence'] = dataset['employee_residence'].apply(lambda x : x if x in config.req_emp_residences else 'Others' )
dataset['company_location'] = dataset['company_location'].apply(lambda x : x if x in config.req_company_locations else 'Others' )
dataset['emp_residence_company_location'] = dataset['employee_residence']+'_'+dataset['company_location']
dataset.drop(columns = ['employee_residence','company_location'],inplace=True)

# modelling
X = dataset.drop(columns=["salary_in_usd"])
y = dataset[["salary_in_usd"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train.shape, X_test.shape
num_cols = X_train.select_dtypes(include="number").columns
cat_cols = X_train.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer(
    [("encoder", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
)

def objective(trial):
    params = {
        "criterion": trial.suggest_categorical(
            "criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"]
        ),
        "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        "max_depth": trial.suggest_int("max_depth", 1, 32, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_weight_fraction_leaf": trial.suggest_float(
            "min_weight_fraction_leaf", 0.0, 0.5
        ),
        "max_features": trial.suggest_float(
            "max_features", 0, 1
        ),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 1000),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 1.0),
    }
    model = DecisionTreeRegressor(**params)

    with mlflow.start_run(nested=True) as run:
        pipeline_steps = [("processor", preprocessor), ("regressor", model)]
        reg = Pipeline(pipeline_steps)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        r2_score_ = r2_score(y_test, preds)

        # logging metrics and saving model
        mlflow.log_params(params, run_id=run.info.run_id)
        mlflow.log_metric("r2_score", r2_score_)
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds)))
        mlflow.log_metric("mse", mean_squared_error(y_test, preds))
        mlflow.log_metric("mae", mean_absolute_error(y_test, preds))
        # mlflow.sklearn.log_model(sk_model=reg, artifact_path=f"model_{run.info.run_id}")

    return r2_score_


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, show_progress_bar=True, n_jobs=-1)

best_trial = study.best_trial
best_params = best_trial.params

print(f"Best trial R2 score: {best_trial.value}")
print(f"Best trial parameters: {best_params}")

# Registering the best model
best_model = DecisionTreeRegressor(**best_params)
pipeline_steps = [("processor", preprocessor), ("regressor", best_model)]
best_pipeline = Pipeline(pipeline_steps)
best_pipeline.fit(X_train, y_train)

mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="sklearn-model",
        signature=infer_signature(X_test, y_test),
        registered_model_name=config.model_name,
    )
# saving locally
with open(artifacts / "final_model.pkl", "wb") as file:
    pickle.dump(best_pipeline, file)
    
mlflow.end_run()
