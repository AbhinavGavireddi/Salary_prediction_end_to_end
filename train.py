import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from datetime import date
from pathlib import Path

root = Path().cwd()
artifacts = root / "artifacts"

experiment_name = "salary_prediction"
experiment_id = None
try:
    experiment =mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
except:
    experiment_id = mlflow.create_experiment(experiment_name)

print(f"{experiment_id}")
mlflow.autolog()
dataset = pd.read_csv(
    artifacts / "dataset.csv",
    low_memory=False,
)
dataset.head()
cat_cols = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "remote_ratio",
    "company_size",
    "emp_residence_company_location",
]
num_cols = ["salary_in_usd"]

for col in cat_cols:
    dataset[col] = pd.Categorical(dataset[col])

for col in num_cols:
    dataset[col] = pd.to_numeric(dataset[col])
experience_level_map = {"Senior": 3, "Middle": 2, "Junior": 1, "Executive": 0}

company_size_map = {
    "Large": 2,
    "Medium": 1,
    "Small": 0,
}

dataset["experience_level"] = (
    dataset["experience_level"].map(experience_level_map).astype("int64")
)
dataset["company_size"] = dataset["company_size"].map(company_size_map).astype("int64")
dataset.dtypes
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
models = {
    "Linear Regression": LinearRegression(),
    "decision Tree": DecisionTreeRegressor(),
    "random forest": RandomForestRegressor(),
}

with mlflow.start_run():
    for name, model in models.items():
        with mlflow.start_run(nested=True):
            pipeline_steps = [("processor", preprocessor)]
            pipeline_steps.append(("regressor", model))
            reg = Pipeline(pipeline_steps)
            reg.fit(X_train, y_train)
            preds = reg.predict(X_test)
            r2_score_ = r2_score(y_test, preds)
            mlflow.log_param('test_r2_score',r2_score_)
            print(name, r2_score_)
