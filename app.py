import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import logging
from pathlib import Path
from ordered_set import OrderedSet
from scipy.stats import chi2_contingency
import os
import mlflow
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    PolynomialFeatures,
    QuantileTransformer,
)
from utiities import read_yaml_from_path
import warnings
import pickle
warnings.filterwarnings("ignore")


root = Path("../")
artifacts = root / "artifacts"
config_path = root / "config/config.yaml"

config = read_yaml_from_path(config_path)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

experiment_name = "salary_prediction"
experiment_id = None
try:
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
except Exception as e:
    experiment_id = mlflow.create_experiment(experiment_name)

mlflow.autolog()

dataset_path = artifacts / "ds_salaries.csv"

model = pickle.load(artifacts / 'model.pkl')

def read_dataset(path: Path):
    dataset = pd.read_csv(path, low_memory=False)
    logger.info(f"dataset is read from {artifacts}")
    logger.info(f"Total rows available {dataset.shape[0]}")
    return dataset


def preprocessor(dataset):
    dataset = dataset.drop_duplicates()
    cat_cols = config.cat_cols
    num_cols = config.target_col

    for col in cat_cols:
        dataset[col] = pd.Categorical(dataset[col])

    for col in num_cols:
        dataset[col] = pd.to_numeric(dataset[col])

    dataset = dataset[cat_cols + num_cols]

    dataset['experience_level'] = dataset['experience_level'].map(config.experience_level_map)
    dataset['employment_type'] = dataset['employment_type'].map(config.employment_type_map)
    dataset['company_size'] = dataset['company_size'].map(config.company_size_map)
    dataset['remote_ratio'] = dataset['remote_ratio'].map(config.remote_ratio_map)


    dataset["job_title"] = dataset["job_title"].apply(
        lambda x: x if x in config.req_job_titles else "Others"
    )
    dataset["employee_residence"] = dataset["employee_residence"].apply(
        lambda x: x if x in config.req_emp_residences else "Others"
    )
    dataset["company_location"] = dataset["company_location"].apply(
        lambda x: x if x in config.req_company_locations else "Others"
    )

    dataset["emp_residence_company_location"] = (
        dataset["employee_residence"] + "_" + dataset["company_location"]
    )
    return dataset


def predict(*values):
    return model.predict(values)


