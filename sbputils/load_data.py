
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

class Pyrodata():
    def __init__(self, 
                 basepath: pathlib.Path = None, 
                 filenames: list[str] = ["sample_video_high-speed.mp4", "sample-spectra-with-lambda.csv",],
                 experiment: int = 1
                 ):
        "give path as parent folder of data which has nested folders with expreriment number"
        if filenames is not None:
            self.video_spectra = Files(experiment)
            self.filepaths = self.video_spectra.files(basepath, filenames)
        else:
            print("no files passed, using sample files to run")
    
    def read_spectral_data(self, columns: list[str] = None):
        if columns is None:
            cols = ["Frame","Row","Column","Wavelength","Intensity"]
        else:
            cols = columns
        self.spectra = self.video_spectra.read_csv(self.filepaths[1], delimiter=",", cols=cols)

    def add_radiation_cols(self):
        self.C2 = 14400
        self.spectra["y"] = np.log(self.spectra["Intensity"]*(self.spectra["Wavelength"]**5))
        self.spectra["y_smooth"] = self.spectra["y"].rolling(window=30).mean()
        self.spectra["x"] = self.C2/(self.spectra["Wavelength"])
    
    def calc_T(self):
        self.spectra["del_y"] = self.spectra["y_smooth"].diff()
        self.spectra["del_x"] = self.spectra["x"].diff()
        self.spectra["T_0"] = - self.spectra["del_y"]/ self.spectra["del_x"]
        self.x_0 = self.C2/520.0
    
    def plot_spectra(self, args=["Wavelength", "Intensity"]):
        fig = plt.figure(figsize = (8, 8))
        #plt.plot(self.spectra["Wavelength"], self.spectra["Intensity"])
        sns.lineplot(data=self.spectra, x=args[0], y=args[1], hue="Frame")
        plt.show()

class Files():
    def __init__(self, exp_number: int = 0):
        self.exp_num = f"{exp_number:03d}"

    def files(self, basepath: pathlib.Path = None, refNames = ["video_file.mp4", "spectra.csv"]):
        "get the file locations and pointers"
        #file = sys.argv[0]
        try:
            if basepath is not None:
                self.dir = basepath.resolve()
            else:
                filepath = pathlib.Path(__file__).resolve()
                self.dir = filepath.parent
        except:
            raise Exception("unable to resolve base path, give data in nested file directory or give data directory as pathlib.Path object")
        
        self.fileRefNames: list[str] = refNames
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

def test_data_obj():
    "to test the Pyrodata object"
    mydata = Pyrodata(experiment=1)
    mydata.read_spectral_data()
    mydata.add_radiation_cols()
    mydata.calc_T()
    #mydata.spectra["grad"].head(5)
    mydata.plot_spectra(["x", "T_0"])

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
    #exp_files = test_files()
    #csvFile = exp_files[1]
    #print(f"csv file: {csvFile}")
    #print(pd.read_csv(csvFile).head())
    #dataframe = pd.read_csv(exp_files[1])
    spectra_frames = test_csv()
    spectra_frames.head(1)
    #test_video()
    test_data_obj()