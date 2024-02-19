
from pyrodata import Pyrodata, Folder, FileList, analyse_video
from files import Files, get_filename_with_ext, get_file_from_filelist
from video_management import show_image
from typing import Type, TypeVar, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib
import cv2

class SBP():
    "gets pyrodata object and uses files there to read spectra, analyse spectra and video, process them for SBP algo"
    def __init__(self, myExperiment: Optional[Pyrodata]):
        if myExperiment is None:
            self.data_holder = Pyrodata(exp_number=1)
        else:
            self.data_holder = myExperiment
        self.spectra = myExperiment.read_spectral_data()

    def plot_raw_spectra(self, args=["Wavelength", "Intensity"]):
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
    
    def subtract_background_spectra(self):
        "read background as separate data and subtract from spectra"
        #TODO
        pass

    def get_spectral_frames(self):
        "y_smoothing will require chunking the spectra with frames and filling the NA values"
        self.frames: dict[int, pd.DataFrame] = {k: v for _, (k,v) in enumerate(self.spectra.groupby("Frame"))}

    def analyse_spectra(self):
        """ 
            - check the functional variation of the intensity registered vs. blackbody expectations and find divergence. C
            - can do least square fit too
            - easier to do after calling get_spectral_frames
        """
        #TODO
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
            raise Exception("radiation columns missing")
        #print(f"length of T_0 array is {len(self.frames.keys())}")
        self.T_0: list[float] = np.zeros(shape = len(self.frames.keys()))
        #print(self.T_0)
        for k in self.frames.keys(): 
            "loop overa all frames"
            print(f"length of {k}_th frame is {len(self.frames[k].index)}\n")
            print(f"columns: {self.frames[k].columns}")
            self.frames[k]["y_smooth"] = self.frames[k]["y"].rolling(window=smooth_window).mean()     # get the smooth_y for the given frame data
            self.frames[k]["y_smooth"].fillna(0)
            # Getting delta x and delta y
            if use_smoothed_y:
                self.frames[k]["del_y"] = self.frames[k]["y_smooth"].diff()
            else:
                self.frames[k]["del_y"] = self.frames[k]["y"].diff()
            self.frames[k]["del_x"] = self.frames[k]["x"].diff()    # should be in nm (close to 1 nm)
            # Filling NA values
            mean_x_diff = self.frames[k]["del_x"].mean()
            mean_y_diff = self.frames[k]["del_y"].mean()
            self.frames[k]["del_x"].fillna(mean_x_diff)
            self.frames[k]["del_y"].fillna(mean_y_diff)
            # Getting T0s with del_y and del_x for each frame
            self.frames[k]["T_0s"] = - (self.frames[k]["del_y"]/ self.frames[k]["del_x"])**-1
        #For all frames, check where TO lies
        self.T_0 = self._lookup_T0(self.x_0, self.frames)    # index from k-1 since frames start from 1 in csv data

    def _lookup_T0(self, x_0: float, df_dict: dict[int, pd.DataFrame]):
        "based on x_0 and dataframe with T_0s, lookup the nearest location where you can find the T_0, assumes df has colum x and T_0s"
        T_0s = []
        if not x_0:
            x_0 = self.C2/520.0    # using 520 nm
        for k in df_dict.keys():
            result_index = df_dict[k]["x"].sub(x_0).abs().idxmin() - (k-1)*1340    # this is a workaround patch as index was getting increased during lookup
            print(f"index of {k}th frame inside get_T0 is {len(df_dict[k].index)}")
            print(f"index for T_0 in frame {k} is {result_index}\n")
            T_0s.append(df_dict[k]["T_0s"].iloc[result_index])
        return T_0s
    
    def plot_T0s(self, which_frame:int = 1):
        fig = plt.figure(figsize = (8, 8))
        #plt.plot(self.spectra["Wavelength"], self.spectra["Intensity"])
        sns.lineplot(data=self.frames[which_frame], x="x", y="T_0s")
        plt.show()
    
    def plot_framewise_spectra(self, args = ["x", "y_smooth"]):
        fig = plt.figure(figsize = (8, 8))
        #plt.plot(self.spectra["Wavelength"], self.spectra["Intensity"])                           
        for k in self.frames.keys():
            sns.lineplot(data=self.frames[k], x=args[0], y=args[1])
        plt.show()
    
    def video_brightness_data(self, type: Optional[str]):
        datafiles = self.data_holder.file_holder.files()
        print(f"filelist used for calling video processing (mp4 selected with extension) {datafiles}")
        self.bi, self.b0 = process_video(Exp_Num=None, filename=datafiles, ext=".mp4", num_frames=20)  # read 20 frames of the mp4 file in data folder
    
    def plot_brightness(self, data: Optional[Union[np.array, np.ndarray]]):
        show_image(data)

def read_video(video_path: str, num_frame: Optional[int] = 10) -> np.ndarray:
    "read video file, return some number of frame data (prefer small number) as numpy array"
    if num_frame is None:
        num_frame = 10   # keeping it small
    else:
        num_frame = num_frame
    frame_list = []    # get all frames here as list of np.array
    i=0
    print(f"READING VIDEO DATA ...")
    print(f"opening this video file to read data: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps of {video_path}= {fps}")
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
    print(f"VIDEO DATA READ !")
    print(f"array list of frames turned to ndarray with shape: {frame_array.shape}")
    print(f"no. of frames = {i}")
    return frame_array

def process_video(Exp_Num: Optional[int], filename: Optional[Union[list[str], str]], ext: str = ".mp4", num_frames: int = 20):
    "selecting/opening video from a list of files, running the averageing"
    if filename is None:
        Exp_Num = 1
        #getting the class instance
        exp = Files(Exp_Num)
        exp_files = exp.files()
        print(f"{exp_files}")
    elif type(filename) is str:
        exp_files = list(filename)
    else:
        exp_files = filename
    print(f"expriment files considered for processing video: {exp_files}")
    #reading the video data
    #video_path = get_filename_with_ext(exp_files, ext)   # getting data from first .mp4 file found
    video_path = get_file_from_filelist(exp_files, "video_file.mp4")
    print(f"file received with file selector {video_path}")
    #play_video(video_path=video_path)
    b_i = read_video(video_path=video_path, num_frame= num_frames)    # getting frames in an array, keep small number
    #mp4_brightness_accum = np.sum(mp4_brightness, axis = 0)
    b_0 = get_b0(b_i)
    return b_i, b_0

def get_b0(b_i: np.ndarray) -> np.ndarray:
    "calculating b_0 from the b_i s of the accumuated, b_i is 4 D array"
    print(f"Intermediate matrices for b_0 calculation:")
    a = np.sum(np.multiply(np.log(b_i), b_i), axis=0)
    print(f"a shape: {a.shape}\n")
    b = np.sum(b_i, axis=0)
    print(f"b shape: {b.shape}\n")
    c = np.divide(a, b)
    print(f"c shape: {c.shape}\n")
    d = np.exp(c)
    print(f"d shape: {d.shape}\n")
    return d

def test_sbp_obj(mydata: Optional[Pyrodata]):
    if mydata is None:
        mysbp = SBP(Pyrodata(exp_number=1))
    else:
        mysbp(SBP(myExperiment=mydata))
    mysbp.add_radiation_cols()
    mysbp.get_spectral_frames()
    mysbp.plot_spectra(["x", "y"])
    mysbp.calc_framewise_rad_vars(use_smoothed_y=True)
    print(mysbp.x_0, mysbp.T_0)
    mysbp.plot_T0s(which_frame=3)
    return mysbp.T_0

if __name__=="__main__":
    EXP_No: int = 1
    FILES: FileList = ["video_file.mp4", "spectra.csv"]
    myData = Pyrodata(exp_number=EXP_No, filenames=FILES)
    raw_spectra = myData.read_spectral_data()
    print(raw_spectra.head())
    #myData.plot_spectra()
    sbp1 = SBP(myData)
    sbp1.add_radiation_cols()
    print(sbp1.spectra)
    sbp1.plot_raw_spectra(["x", "y"])
    sbp1.get_spectral_frames()
    sbp1.calc_framewise_rad_vars(use_smoothed_y=True, smooth_window=10)
    sbp1.plot_framewise_spectra(["x", "y_smooth"])
    #sbp1.plot_T0s()
    print(f"framewise T0s: {sbp1.T_0}")
    sbp1.video_brightness_data(None)
    sbp1.plot_brightness(sbp1.b0)
    print(f"brightness processed data sizes: {sbp1.bi[0].shape, sbp1.b0.shape}")