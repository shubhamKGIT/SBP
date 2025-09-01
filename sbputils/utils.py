import yaml
import json
import os
from pathlib import Path
from typing import Optional
import numpy as np

FileList = Optional[list[str]]

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


def read_mraw_file(file_path, num_frames, height, width):
    "function to read .mraw file"
    with open(file_path, 'rb') as f:
        # Read the binary data
        raw_data = np.fromfile(f, dtype=np.uint16)
    # Reshape raw data into frames
    frames = raw_data.reshape((num_frames, height, width))
    return frames

def get_filename_with_ext(filelist: FileList, ext: str) -> str:
    "gets first file which matches extension or file with certain filename and extension"
    for f in filelist:
        if os.path.splitext(f)[-1].lower() == ext:
            return f 
        
def get_file_from_filelist(filelist: FileList, file: str) -> str:
    "returns path of file with specific filename"
    for f in filelist:
        if os.path.split(f)[-1] == file:
            return f

if __name__=="__main__":
    project_folder = Path(__file__).parent.parent
    yaml_file =  project_folder / "info.yaml"
    readme_file = project_folder / "README.md"
    data = read_yaml(filepath=yaml_file.absolute())
    readme = read_file(filepath= readme_file)
    print(data)
    print(f"readme content \n {readme}")