
""" select the right interpreter using Crtl + Shift + P, use Python 3.9.x
"""
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
import json

Folder = Optional[pathlib.Path]
FileList = Optional[list[str]]
DataFrame = Optional[pd.DataFrame]
Data = Optional[Union[np.array, np.ndarray]]

from files import Files, get_filename_with_ext

class Pyrodata():
    "object for an experiment with both video and spectral data, video as mp4/mraw and spectra as csv"
    def __init__(self,
                 exp_number: int,    # passes experiment number 
                 basepath: Folder = None, 
                 filenames: FileList = None,
                 ):
        if exp_number is not None:
            print(f"Data object being initialied for experiment no.: {exp_number}")
        else:
            raise Exception(f"Please provide experiment number of data as integer to intialise data object.\n")
        self._exp = exp_number
        try:
            if filenames is None:
                # filename is not given, use defaults
                print(f"no filename passed, using default names video_file.mp4 and spectra.csv\n")
                DEFAULT_FILES = ["video_file.mp4", "spectra.csv"]
                self.file_holder = Files(exp_number)   # using dataset with default file name
                self.filepaths = self.file_holder.files(basepath=basepath, files= DEFAULT_FILES)
            else:
                self.file_holder = Files(exp_number)
                self.filepaths = self.file_holder.files(basepath, filenames)
        except:
            raise Exception(f"Data not initialised.")
        finally:
            print(f"Check file paths used to initialise data object:\n {self.filepaths}")
        info_file = get_filename_with_ext(self.filepaths, ".json")
        self._info = read_json(info_file)
    
    @property
    def experiment_number(self):
        return self._exp

    @property
    def info(self):
        return self._info
    
    def read_spectral_data(self, columns: Optional[list[str]] = None):
        if columns is None:
            cols = ["Frame","Row","Column","Wavelength","Intensity"]
        else:
            cols = columns
        spectra = self.file_holder.read_csv(
                                            get_filename_with_ext(self.filepaths, ".csv"), 
                                            delimiter=",", 
                                            cols=cols
                                            )
        return spectra
    
    def plot_spectra(self, args=["Wavelength", "Intensity"]):
        fig = plt.figure(figsize = (8, 8))
        #plt.plot(self.spectra["Wavelength"], self.spectra["Intensity"])
        sns.lineplot(data=self.read_spectral_data(), x=args[0], y=args[1], hue="Frame")
        plt.show()
    
    def read_video_data(self):
        analyse_video(self.filepaths[0])

def read_json(json_file: str):
    "return data from json file"
    with open(json_file) as f:
        data = json.load(f)
        f.close()
    return data

def test_pyrodata_obj():
    "to test the Pyrodata class and its methods"
    mydata = Pyrodata(exp_number= 1)
    mydata.read_spectral_data()
    mydata.plot_spectra()
    mydata.read_video_data()

def analyse_video(video_path: str, num_frame: Optional[int] = 10) -> np.ndarray:
    "read video file, return some number of frame data (prefer small number) as numpy array"
    if num_frame is None:
        num_frame = 10
    else:
        num_frame = num_frame
    frame_list = []
    i=0
    print(f"opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps = {fps}")
    while(cap.isOpened()):
        ret, frame = cap.read()
        #print(frame, ret)
        if ret and i<num_frame:
            frame_list.append(frame)
            i= i+1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    frame_array = np.array(frame_list)
    print(f"no. of frames = {i}")
    print(f"array shape: {frame_array.shape}")
    return frame_array

if __name__== "__main__":
    "getting all data files and registering it in objects to open camera data and spectra"
    #DATA_FILES = ["some_camFile.mp4", "spectra.csv"]
    #DIR, FILE = os.path.split(os.path.__file__)
    #DATA_DIR = os.path.join(DIR, "data")
    #MY_FILES = [os.path.join(DIR, "data", file) for file in DATA_FILES]
    #print(f"file to open : {MY_FILES}")
    #exp_files = test_files()
    #csvFile = exp_files[1]
    #print(f"csv file: {csvFile}")
    #print(pd.read_csv(csvFile).head())
    #dataframe = pd.read_csv(exp_files[1])
    # Use this section to test spectra part
    """spectra_frames = test_csv()
    spectra_frames.head(1)
    test_data_obj()"""

    """b_i, b_0 = test_video()
    #mp4_array_accum = np.sum(mp4_array, axis=2)
    print(f"accumulated array shape: {b_0.shape}")
    factor = 520./1440000.    #lambda_filter/c2
    inv_T = np.multiply(factor, np.log(np.divide(b_0, b_i[8,:,:,:])))
    print(inv_T)
    T_0s = test_data_obj()
    T_0 = T_0s[0]    # taking the first value here
    print(f"T_0: {T_0s}")
    T = np.reciprocal(np.add(1/T_0, inv_T))
    print(f"Elementwise temperature:\n {T}")
    plt.imshow(np.reciprocal(inv_T), cmap="magma_r")
    plt.show()
    plt.imshow(np.divide(T, 2500.), cmap="magma_r")
    plt.show()"""

    test_pyrodata_obj()
    
   