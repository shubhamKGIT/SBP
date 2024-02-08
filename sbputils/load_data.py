
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
    
    def plot_spectra(self, args=["Wavelength", "Intensity"]):
        fig = plt.figure(figsize = (8, 8))
        #plt.plot(self.spectra["Wavelength"], self.spectra["Intensity"])
        sns.lineplot(data=self.spectra, x=args[0], y=args[1], hue="Frame")
        plt.show()

    def add_radiation_cols(self):
        """ y = Intesnity /lambda**5 
            x = C2/ lambda
            - need to chunk the spectra wrt frames before doing smoothing
            - might explore smoothing before scaling with powers of lambda
            - can be plotted with plot_spectra() method
        """
        self.C2 = 14400*1000    # multiplied by 1K because we have lambda in nm
        self.x_0 = self.C2 / 520.0     # getting reference variable value for filter
        self.spectra["y"] = np.log(self.spectra["Intensity"]*(self.spectra["Wavelength"]**5))
        self.spectra["x"] = self.C2/(self.spectra["Wavelength"])

    def get_spectral_frames(self):
        "y_smoothing will require chunking the spectra with frames and filling the NA values"
        self.frames: dict[int, pd.DataFrame] = {k: v for _, (k,v) in enumerate(self.spectra.groupby("Frame"))}

    def analyse_spectra(self):
        " check the functional variation of the intensity registered vs. intensity from surface"
        pass
    
    def calc_framewise_rad_vars(self, use_smoothed_y: bool = True, smooth_window = 30):
        "calculate the y_diff, x_diff, T_Os and exact T_0 for X_0 for all spectral frames"
        
        try:
            if "y" not in self.spectra.columns:
                self.add_radiation_cols(smooth_window=30)
                self.get_spectral_frames()
            else:
                pass
        except:
            raise Exception("radiation colums missing")
        #print(f"length of T_0 array is {len(self.frames.keys())}")
        self.T_0: list[float] = np.zeros(shape = len(self.frames.keys()))
        #print(self.T_0)
        for k in self.frames.keys(): 
            print(f"length of {k}_th frame is {len(self.frames[k].index)}\n")
            print(f"columns: {self.frames[k].columns}")
            self.frames[k]["y_smooth"] = self.frames[k]["y"].rolling(window=smooth_window).mean()     # get the smooth_y for the given frame data
            if use_smoothed_y:
                self.frames[k]["del_y"] = self.frames[k]["y_smooth"].diff()
            else:
                self.frames[k]["del_y"] = self.frames[k]["y"].diff()
            self.frames[k]["del_x"] = self.frames[k]["x"].diff()    # should be in nm (close to 1 nm)
            mean_x_diff = self.frames[k]["del_x"].mean()
            mean_y_diff = self.frames[k]["del_y"].mean()
            self.frames[k]["del_x"].fillna(mean_x_diff)
            self.frames[k]["del_y"].fillna(mean_y_diff)
            self.frames[k]["T_0s"] = - (self.frames[k]["del_y"]/ self.frames[k]["del_x"])**-1
        self.T_0 = self._lookup_T0(self.x_0, self.frames)    # index from k-1 since frames start from 1 in csv data

    def _lookup_T0(self, x_0: float, df_dict: dict[int, pd.DataFrame]):
        "based on x_0 and dataframe with T_0s, lookup the nearest location where you can find the T_0, assumes df has colum x and T_0s"
        T_0 = []
        if not x_0:
            x_0 = self.C2/520.0    # using 520 nm
        for k in df_dict.keys():
            result_index = df_dict[k]["x"].sub(x_0).abs().idxmin() - (k-1)*1340    # this is a workaround patch as index was getting increased during lookup
            print(f"index of {k}th frame inside get_T0 is {len(df_dict[k].index)}")
            print(f"index for T_0 in frame {k} is {result_index}\n")
            T_0.append(df_dict[k]["T_0s"].iloc[result_index])
        return T_0
    
    def plot_T0s(self, which_frame:int = 1):
        fig = plt.figure(figsize = (8, 8))
        #plt.plot(self.spectra["Wavelength"], self.spectra["Intensity"])
        sns.lineplot(data=self.frames[which_frame], x="x", y="T_0s")
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
    mydata.get_spectral_frames()
    mydata.calc_framewise_rad_vars(use_smoothed_y=True)
    #mydata.plot_spectra(["x", "y_smooth"])
    #mydata.plot_spectra(["x", "T_0s"])
    mydata.plot_spectra(["x", "y"])
    print(mydata.x_0, mydata.T_0)
    #print(mydata.spectra["del_x"])
    mydata.plot_T0s(which_frame=3)
    return mydata.T_0

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

def play_video(video_path):
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

def analyse_video(video_path):
    "check aspects of hwo video data structure is handled"
    frame_list = []
    i=0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps = {fps}")
    while(cap.isOpened()):
        ret, frame = cap.read()
        #print(frame, ret)
        if ret and i<10:
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

def test_video():
    "basic test for video"
    EXP_No = 1
    #getting the class instance
    exp = Files(EXP_No)
    exp_files = exp.files()
    print(f"{exp_files}")
    #reading the video data
    video_path = exp_files[0]
    #play_video(video_path=video_path)
    b_i = analyse_video(video_path=video_path)
    #mp4_brightness_accum = np.sum(mp4_brightness, axis = 0)
    b_0 = get_b0(b_i)
    return b_i, b_0

def get_b0(b_i: np.ndarray) -> np.ndarray:
    "calculating b_0 from the b_i s of the accumuated, b_i is 4 D array"
    a = np.sum(np.multiply(np.log(b_i), b_i), axis=0)
    print(f"a shape: {a.shape}\n")
    b = np.sum(b_i, axis=0)
    print(f"b shape: {b.shape}\n")
    c = np.divide(a, b)
    print(f"c shape: {c.shape}\n")
    d = np.exp(c)
    print(f"d shape: {d.shape}\n")
    return d

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
    b_i, b_0 = test_video()
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
    plt.show()