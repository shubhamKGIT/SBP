
""" above line is for when you wanna make this file executable and needs its to know which python version to launch it with,
    select the right interpreter using Crtl + Shift + P
"""
import sys
import os
import pathlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2
import pandas as pd

class Pyrodata():
    def __init__(self, filePaths: list[str]):
        self.filepaths = filePaths
    
    def files(self):
        return f"[]"

class Files():
    def __init__(self, exp_number: int = 0):
        self.exp_num = f"{exp_number:03d}"

    def files(self, refNames = ["sample_video_file.mp4", "spectra.csv"]):
        "get the file locations and pointers"
        #file = sys.argv[0]
        filepath = pathlib.Path(__file__).resolve()
        self.fileRefNames: list[str] = refNames
        self.dir = filepath.parent
        self.dataDir, _ = os.path.split(self.dir)
        self.dataFiles = [os.path.join(self.dataDir, "data", self.exp_num, _) for _ in self.fileRefNames]
        return self.dataFiles
    
    def read_csv(self, filepath, delimiter = ",", cols = ["col1", "col2", "col3"]):
        "reading csv for spectral data"
        try:
            if filepath is not None:
                csv_dataframe = pd.read_csv(filepath, delimiter=delimiter, usecols= cols)
            else:
                csv_dataframe = pd.read_csv(self.dataFiles[1], delimiter=delimiter, usecols= cols)
        except:
            raise Exception("the filepath was not given or not established in object, call files method")
        return csv_dataframe
    
    def read_video(self, filepath):
        pass
    
def test_files():
    "basic test for first time code test"
    EXP_No = 1
    #getting the class instance
    exp = Files(EXP_No)
    exp_files = exp.files()
    print(f"{exp_files}")
    #also returning filenames
    return exp_files

def test_csv():
    "basic test for testing csv read"
    EXP_No = 1
    #getting the class instance
    exp = Files(EXP_No)
    exp_files = exp.files()
    print(f"{exp_files}")
    #reading the csv data
    spectral_data = exp.read_csv(None)
    print(spectral_data.head())
    return spectral_data

def test_video():
    "basic test for video"
    EXP_No = 1
    #getting the class instance
    exp = Files(EXP_No)
    exp_files = exp.files()
    print(f"{exp_files}")
    #reading the video data
    video_path = exp_files[0]
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        print(frame, ret)
        if ret:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    "getting all data files and registering it in objects to open camera data and spectra"
    #DATA_FILES = ["some_camFile.mp4", "spectra.csv"]
    #DIR, FILE = os.path.split(os.path.__file__)
    #DATA_DIR = os.path.join(DIR, "data")
    #MY_FILES = [os.path.join(DIR, "data", file) for file in DATA_FILES]
    #print(f"file to open : {MY_FILES}")
    exp_files = test_files()
    csvFile = exp_files[1]
    print(f"csv file: {csvFile}")
    #print(pd.read_csv(csvFile).head())
    #dataframe = pd.read_csv(exp_files[1])
    spectra_frames = test_csv()
    spectra_frames.head(1)
    test_video()