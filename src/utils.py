import yaml
import json
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

FileList = Optional[list[str]]

def read_csv(filepath: Optional[str] = None, 
            delimiter: str = ",", 
            cols: list[str] = ["col1", "col2", "col3"]
            ) -> Optional[pd.DataFrame]:
    "reading csv - designed for spectral data but can read any csv file"
    try:
        if filepath is not None:
            csv_dataframe = pd.read_csv(filepath, delimiter=delimiter, usecols= cols)
        else: 
            print(f"not filepath passed, nothing to read, try again!")
            csv_dataframe = None
    except:
        raise Exception("Could not read the csv")
    return csv_dataframe

def read_file(filepath):
    "read contents of yaml file"
    with open(filepath, "r") as f:
        data = f.readlines()
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
        
def get_file_from_filelist(filelist: FileList, filename: str) -> str:
    "returns path of file with specific filename"
    for f in filelist:
        if os.path.split(f)[-1] == filename:
            return f

def get_experiment_folder(experiment_number):
    # Get the directory of the current script being executed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the parent folder
    parent_dir = os.path.dirname(script_dir)
    # Path to the "data" subfolder
    data_dir = os.path.join(parent_dir, 'data')
    # Find the folder for the specified experiment number
    experiment_folder = os.path.join(data_dir, f"{experiment_number:03d}") 
    # Check if the experiment folder exists
    return experiment_folder

def find_video_files_in_experiment_folder(experiment_folder):
    """ helper function to find the raw files in experiment folder, assumes one file written only

    ARGS
    ----
        experiment_folder: str | Path
            path to the folder having raw video files

    RETURNS
    -------
        mrwa_file, cihx_file: path as strings
            paths to first raw and cihx files
    """
    mraw_file = None
    cihx_file = None
    for root, dirs, files in os.walk(experiment_folder):
        for file in files:
            if file.endswith('.mraw'):
                mraw_file = os.path.join(root, file)
            elif file.endswith('.cihx'):
                cihx_file = os.path.join(root, file)
    
    return mraw_file, cihx_file

if __name__=="__main__":
    project_folder = Path(__file__).parent.parent.resolve()
    yaml_file =  os.path.join(project_folder, "info.yaml")
    readme_file = os.path.join(project_folder, "README.md")
    data = read_yaml(filepath=yaml_file)
    print(data)
    print(f"readme file path: {readme_file}")
    readme = read_file(filepath=readme_file)
    print(f"readme content \n {readme}")