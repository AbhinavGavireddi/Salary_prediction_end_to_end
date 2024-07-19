from pathlib import Path
from box import ConfigBox
import yaml
from dotenv import load_dotenv
import warnings
import logging
import os
import sys


load_dotenv()

warnings.filterwarnings("ignore")
root = Path().cwd()
artifacts = root / "artifacts"
experiment_name = "salary_prediction"

os.makedirs(artifacts, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s: %(module)s: %(message)s: [in %(pathname)s:%(lineno)d]",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("niroBRE")
logger.info(f"working in {root} directory")
logger.info(f"loading artifacts from {artifacts}")

def read_yaml_from_path(path: Path):
    with open(path, "r") as j:
        content = yaml.safe_load(j)
    config = ConfigBox(content)
    return config

config_path = root / 'config.yaml'

config = read_yaml_from_path(config_path)
logger.info(f'loaded config from {config_path}.')