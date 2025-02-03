import yaml
import json
from pathlib import Path

def read_file(filepath):
    "read contents of yaml file"
    with open(filepath, "r") as f:
        data = filepath.readlines()
        f.close()
    return data

def read_yaml(filepath):
    "read contents of yaml file"
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return data

def read_json(filepath):
    "read contents of yaml file"
    with open(filepath, "r") as f:
        data = json.load(f)
        f.close()
    return data

if __name__=="__main__":
    project_folder = Path(__file__).parent.parent
    yaml_file =  project_folder / "info.yaml"
    readme_file = project_folder / "README.md"
    data = read_yaml(filepath=yaml_file.absolute())
    readme = read_file(filepath= readme_file)
    print(data)
    print(f"readme content \n {readme}")