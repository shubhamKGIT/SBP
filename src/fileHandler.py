

import os
import pathlib
import numpy as np
import cv2
import pandas as pd
from mraw import load_video
from typing import Union, Optional
from utils import get_filename_with_ext
import glob
from videoUtils import show_image

Folder = Optional[pathlib.Path]
FileList = Optional[list[str]]
DataFrame = Optional[pd.DataFrame]
Data = Optional[Union[np.array, np.ndarray]]

class Files():
    """Object to return the file names to the pyro data object; takes exp_number, 
    pathlib.Path for data folder and filenames as list of string

    PARATMETERS
    -----------
    baseDir: directory of parent folder
    dataDir: data folder path
    expFolder: path to exp folder
    expFiles: files for an exp

    METHODS
    -------
    read_csv:  reads the spectra to test;
    read_mp4: can test mp4, first implementation was based on mp4
    """

    def __init__(self, exp_number: int, basepath: Folder = None):
        self.exp_num = f"{exp_number:03d}"
        self.baseDir, self.dataDir, self.expFolder, self.expFiles = self._files(basepath)

    def _files(self, basepath: Folder = None):  # constructor to initalize
            "generate the filepaths"
            try:
                if basepath is not None:
                    baseDir = basepath
                else:
                    script_dir = pathlib.Path(__file__)
                    baseDir = script_dir.parent.parent   # goes from file -> sbputils -> SBP
            except:
                raise Exception("Unable to resolve base path, give data directory as pathlib.Path object")
            dataDir = baseDir/ "data"
            expFolder = dataDir / self.exp_num
            expFiles = [os.path.join(expFolder, f) for f in expFolder.iterdir()]
            return baseDir, dataDir, expFolder, expFiles
    
    def __repr__(self):
        return f"the files from experiment data folder are: {self.expFiles}"

class FilesTester():
    def __init__(self):
        pass # nothing to do here
    def check_csv(self, 
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
        print(csv_dataframe.head())
    
    def check_video(self, filepath: Optional[str] = None, mp4:bool = False) -> None:
        "filepath is cihx filepath in case of mraw video"
        try:
            if filepath is not None:
                if not mp4:
                    print(f"Reading the .cihx and .mrwa files")
                    images, cih = load_video(cih_file=filepath)
                    show_image(image_data=images[20])
                    print(f'Bit rate: {cih["Color Bit"]} bit, Frame rate: {cih["Record Rate(fps)"]}')
                else:
                    print(f"Reding the mp4 file")
                    video_file = get_filename_with_ext(self.expFiles, ".mp4")
                    play_mp4(video_file)
            else:
                print(f"pass a video file name either the .chix (has .mraw located with it) or the .mp4")
        except:
            raise Exception("Files object might be setup properly check again!")

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
        