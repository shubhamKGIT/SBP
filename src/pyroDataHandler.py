

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
from mraw import load_video
from typing import Union, Optional
import json
import re

from fileHandler import Files, get_filename_with_ext
from utils import read_csv, read_json

Folder = Optional[pathlib.Path]
FileList = Optional[list[str]]
DataFrame = Optional[pd.DataFrame]
Data = Optional[Union[np.array, np.ndarray]]

class CustomDataHnadler():
    def __init__(self,
                 fileHandler: Optional[Files] = None,
                 exp_num: Optional[int] = None,
                 expDataFileList: Optional[list[str]] = None,
                 video_format: str = ".mraw"
                 ):
        self.fileHandler = fileHandler

    def read_exp_data(self):
        pass

    def __repr__(self):
        return f"the experiment data {self.fileHandler} will be used"

class PyroData(CustomDataHnadler):
    """experiment specific spectral and data files holder, video as mp4/mraw and spectra as csv
    
    ARGS
    ----
    fileHandler: Files object
    or 
    exp_num + expDataFileList

    METHODS
    -------
    read_spectral_data
    read_brightness_data
    plot_spectra
    """
    def __init__(self,
                 fileHandler: Optional[Files] = None,
                 exp_num: Optional[int] = None,
                 expDataFileList: Optional[list[str]] = None,
                 video_format: str = ".mraw"
                 ):
        self.fileHandler = fileHandler
        if fileHandler is None:
            self.dataFiles = expDataFileList
        else:
            self.dataFiles = fileHandler.expFiles
        info_file = get_filename_with_ext(self.dataFiles, ".json")
        self.spectraFile = get_filename_with_ext(self.dataFiles, ".csv")
        if video_format == ".mraw":
            self.videoFile = get_filename_with_ext(self.dataFiles, ".cihx")
        else: 
            self.videoFile = get_filename_with_ext(self.dataFiles, ".mp4")
        self.videoFormat = video_format
        self._exp_num = exp_num
        self._info = read_json(info_file)
    
    @property
    def experiment_number(self):
        return self._exp_num   # only reads here

    @property
    def info(self):
        return self._info
    
    def read_spectral_data(self, columns: Optional[list[str]] = None):
        if columns is None:
            cols = ["Frame","Row","Column","Wavelength","Intensity"]
        else:
            cols = columns
        print(f"reading {self.spectraFile}")
        spectra = read_csv( filepath=self.spectraFile, 
                            delimiter=",", 
                            cols=cols
                            )
        return spectra
    
    def read_brightness_data(self):
        if self.videoFormat==".mraw":
            frame_array, cih_info = load_video(self.videoFile)
        elif self.videoFormat==".mp4":
            frame_array = analyse_video(self.videoFile)
        else:
            print(f"the video file format is not supported to be read here, pass .mraw or .mp4 only")
        return frame_array
    
    def plot_spectra(self, args=["Wavelength", "Intensity"]):
        plt.figure(figsize = (8, 8))
        #plt.plot(self.spectra["Wavelength"], self.spectra["Intensity"])
        sns.lineplot(data=self.read_spectral_data(), x=args[0], y=args[1], hue="Frame")
        plt.show()

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

def test_pyrodata_obj():
    "to test the Pyrodata class and its methods"
    fileHandelr = Files(exp_number=12)
    mydata = PyroData(fileHandler=fileHandelr, video_format=".mraw")
    mydata.read_spectral_data()
    mydata.plot_spectra()
    #_ = mydata.read_brightness_data()
    frames, cih_info = load_video(mydata.videoFile)
    print(frames.shape)
    print(cih_info["Total Frame"])

if __name__=="__main__":
    test_pyrodata_obj()