
import sys
import os
import pathlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
from PIL import Image
from mraw import load_video
from typing import Type, TypeVar, Union, Optional

Folder = Optional[pathlib.Path]
FileList = Optional[list[str]]
DataFrame = Optional[pd.DataFrame]
Data = Optional[Union[np.array, np.ndarray]]
from utils import get_file_from_filelist, get_filename_with_ext

class Files():
    "Object to return the file names to the pyro data object; takes exp_number, pathlib.Path for data folder and filenames as list of string"
    def __init__(self, exp_number: int = 1):
        self.exp_num = f"{exp_number:03d}"
    def files(self, 
              basepath: Folder = None, 
              files: FileList = None
              ):
        "generate the filepaths"
        try:
            if basepath is not None:
                self.basedir = basepath
            else:
                src_filepath = pathlib.Path(__file__).resolve()
                self.basedir = src_filepath.parent.parent   # goes from file -> sbputils -> SBP
        except:
            raise Exception("Unable to resolve base path, give data directory as pathlib.Path object")
        self.dataDir = self.basedir.joinpath("data")
        if files is None:
            self.fileNames = [f for f in self.dataDir.joinpath(self.exp_num).iterdir() if f.is_file()]
        else:
            self.fileNames = files
        self.expFolder = os.path.join(self.dataDir, self.exp_num)
        self.dataFiles = [os.path.join(self.dataDir, self.exp_num, f) for f in self.fileNames]
        return self.dataFiles
    
    def read_csv(self, 
                 filepath: Optional[str] = None, 
                 delimiter: str = ",", 
                 cols: list[str] = ["col1", "col2", "col3"]
                 ) -> pd.DataFrame:
        "reading csv - designed for spectral data but can read any csv file"
        try:
            if filepath is not None:
                csv_dataframe = pd.read_csv(filepath, delimiter=delimiter, usecols= cols)
            else:
                csv_file = get_filename_with_ext(self.dataFiles, ".csv")
                csv_dataframe = pd.read_csv(csv_file, delimiter=delimiter, usecols= cols)
        except:
            raise Exception("the filepath was not given or not established properly in object, call files method to check csv filename")
        return csv_dataframe
    
    def check_video(self, filepath: Optional[str] = None) -> None:
        try:
            if filepath is not None:
                #play_video(filepath)
                pass #TODO # implement mraw read function call here
            else:
                video_file = get_filename_with_ext(self.dataFiles, ".mp4")
                play_mp4(video_file)
        except:
            raise Exception("the filepath was not given or not established properly in object, call files method to check video filename")

def play_mp4(video_path: str):
    "plays to test a video, takes absolute filepath of video as string (can play mp4) or can take Files object video path"
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
        
def test_files_obj(exp: int = 0):
    "test files class and its methods"
    EXP_No = exp
    #getting the class instance
    exp = Files(EXP_No)
    exp_files = exp.files()    # called without basepath and filenames
    print(f"{exp_files}")
    #also returning filenames
    return exp_files

def test_csv_read(exp: int = 0, cols: list[str] = ["Frame", "Wavelength", "Intensity"]):
    "basic test for testing csv read"
    EXP_No = exp    # should load video_file.mp4 and spectra.csv from folder 000
    #getting the class instance
    #FILES = ["video.mp4", "spectra.csv"]
    FILES = None
    exp = Files(EXP_No)
    exp_files = exp.files(FILES)
    print(f"files found: {exp_files} \n")
    csv_file = get_filename_with_ext(exp_files, ".csv")
    #reading the csv data
    spectral_data = exp.read_csv(filepath=csv_file, cols = cols)
    print(f"csv data read using Files.read_csv(), returned as pd.Dataframe, printing below sone lines:\n")
    print(spectral_data.head())
    return spectral_data

def test_video_read(exp: int = 0, extension: str = None):
    "play the .mp4 file"
    EXP_No = exp   # should load video_file.mp4 and spectra.csv from folder 000
    #getting the class instance
    #FILES = ["video_file.mp4", "spectra.csv"]
    exp = Files(EXP_No)
    FILES = None
    exp_files = exp.files(files= FILES)
    print(f"files found: {exp_files} \n")
    if extension == ".mp4":
        video_file = get_filename_with_ext(exp_files, ".mp4")
        exp.check_video(video_file)
    else:
        video_file = get_filename_with_ext(exp_files, ".mraw")
        print(f"reading video file named: {video_file}")
        pass

if __name__=="__main__":
    #test_files_obj()
    test_csv_read(12)
    test_video_read(exp=12, extension=".mraw")
    #play_video(Files(0).files()[0])
    