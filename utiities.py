from box import ConfigBox
from pathlib import Path
import yaml

def read_yaml_from_path(path: Path):
    with open(path, "r") as j:
        content = yaml.safe_load(j)
    config = ConfigBox(content)
    return config
