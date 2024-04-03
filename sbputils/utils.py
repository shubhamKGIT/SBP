import yaml
import json
import pathlib
import os
import glob

def read_file(filepath):
    "read contents of yaml file"
    with open(filepath, "r") as f:
        data = filepath.readlines()
        f.close()
    return data

def read_yaml(filepath):
    "read contents of yaml file"
    with open(filepath, "r") as f:
        data = yaml.safe_load(filepath)
        f.close()
    return data

def read_json(filepath):
    "read contents of yaml file"
    with open(filepath, "r") as f:
        data = json.load(filepath)
        f.close()
    return data

if __name__=="__main__":
    yaml_file = os.path.join(pathlib.Path(__file__).parent.parent, "info.yaml")
    readme_file = os.path.join(pathlib.Path(__file__).parent.parent, "README.md")
    data = read_yaml(filepath=yaml_file)
    readme = read_file(filepath=read_file)
    print(data)
    print(f"readme content \n {readme}")