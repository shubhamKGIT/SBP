
from pyrodata import Pyrodata, Folder, FileList, analyse_video
from files import Files, get_filename_with_ext, get_file_from_filelist
from video import show_image, frame_sync_vid_seq, show_masked_image
from mraw import load_video
from typing import Type, TypeVar, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
                self.add_radiation_cols(smooth_window=smooth_window)
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
            if use_smoothed_y is True:
                self.frames[k]["del_y"] = self.frames[k]["y_smooth"].diff(-400)
            else:
                self.frames[k]["del_y"] = self.frames[k]["y"].diff().diff(-400)
            self.frames[k]["del_x"] = self.frames[k]["x"].diff(-400)    # should be in nm (close to 40 nm)
            # Filling NA values
            mean_x_diff = self.frames[k]["del_x"].mean()
            mean_y_diff = self.frames[k]["del_y"].mean()
            self.frames[k]["del_x"].fillna(mean_x_diff)
            self.frames[k]["del_y"].fillna(mean_y_diff)
            # Getting T0s with del_y and del_x for each frame
            self.frames[k]["T_0s"] = - (self.frames[k]["del_y"]/ self.frames[k]["del_x"])**-1
            self.frames[k]["T_0s_avg"] = self.frames[k]["T_0s"].rolling(window=10).mean()
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
            #T_0s.append(df_dict[k]["T_0s"].iloc[result_index])
            T_0s.append(df_dict[k]["T_0s_avg"].iloc[result_index])    # selecting from smoothed columns
        return T_0s
    
    def plot_T0s(self, which_frame:int = 1, T_lim: float = 1e5):
        fig = plt.figure(figsize = (8, 8))
        #plt.plot(self.spectra["Wavelength"], self.spectra["Intensity"])
        sns.lineplot(data=self.frames[which_frame], x="x", y="T_0s")
        plt.ylim(0, T_lim)
        plt.show()
    
    def plot_framewise_spectra(self, args = ["x", "y_smooth"], ylimit: Optional[float] = None):
        fig = plt.figure(figsize = (8, 8))
        #plt.plot(self.spectra["Wavelength"], self.spectra["Intensity"])                           
        for k in self.frames.keys():
            sns.lineplot(data=self.frames[k], x=args[0], y=args[1])
        if ylimit is not None:
            plt.ylim(0, ylimit)
        else:
            pass
        plt.show()
    
    def video_brightness_data(self, type: Optional[str]):
        "handles various type of video data files and return the brightness, also a way to test if files read"
        datafiles = self.data_holder.file_holder.files()
        print(f"filelist used for calling video processing (mp4 selected with extension) {datafiles}")
        self.bi, self.b0, _ = process_video(Exp_Num=None, 
                                         filename=datafiles, 
                                         vid_file="video_file", 
                                         ext=".mp4",
                                         start_frame= 0,
                                         num_frames=20)  # read 20 frames of the mp4 file in data folder
    
    def plot_brightness(self, data: Optional[Union[np.array, np.ndarray]]):
        show_image(data)
    
    def vid_seq_temperature(self, spectral_frame_num: int, test: bool = True):
        "get video data holder and call the build_frame_temperature passing T_0, b_i dataset and the b_0"
        try:
            T_0 = self.T_0[spectral_frame_num -1]
            print(f"T_0 used for spectral frame {spectral_frame_num} is :{T_0} K")
        except: 
            raise Exception(f"Could not pick reference temeperature for the spectral frames")

        if test is True:
            b_i, b_0, _ = process_video(Exp_Num=None, 
                                    filename=self.data_holder.file_holder.files(), 
                                    vid_file="video_file",
                                    ext=".mp4",
                                    start_frame= 0,
                                    num_frames= 20)
        else:
            start_at_vid_frame, num_of_frames = frame_sync_vid_seq(info= self.data_holder.info, spectral_frame_num= spectral_frame_num)
            num_of_frames = 50 # to keep is manageable # TODO: change this later
            b_i, b_0, b_0_over_t, cih_info = process_video(Exp_Num=None, 
                                    filename=self.data_holder.file_holder.files(), 
                                    vid_file=None,
                                    ext=".mraw",
                                    start_frame= start_at_vid_frame,
                                    num_frames= num_of_frames)
            print(f"calc T_i called...")
        T_i = np.zeros(b_i.shape)
        # temporal integration decoupled here
        T_j_framewise = np.zeros(b_0.shape)
        T_j_framewise = self.calc_Ti(bi_frame=b_0, b0=b_0_over_t, T0=T_0)
        # spatial integration decoupled here
        for k in range(b_i.shape[0]):
            "camera framewise temperature"
            T_i[k] = self.calc_Ti(bi_frame=b_i[k], b0=b_0[k], T0=T_j_framewise[k])
        return T_i, T_j_framewise, b_i, b_0, b_0_over_t, cih_info

    def calc_Ti(self, bi_frame: np.ndarray, b0: float, T0: float):
        "calculates T_i from T_0, b_i and b_0"
        b_0 = np.zeros(shape= bi_frame.shape, dtype = float)
        b_0[:] = b0   # populate with same b0s
        temp1 = np.zeros(shape= bi_frame.shape, dtype = float)
        temp1[:] = 1/T0
        temp2 = (1/self.x_0)*np.log(b_0)
        bi_frame_log =  np.log(bi_frame)
        bi_frame_log = np.nan_to_num(bi_frame_log, copy=False, nan=-9999, neginf=-33333333)
        temp3 = - (1/self.x_0)*bi_frame_log
        temp12 = np.add(temp1, temp2)
        temp_inv = np.add(temp12, temp3)
        T_i = np.reciprocal(temp_inv)
        print(f"T_i retuned.")
        return T_i

def read_video(video_path: str, start_frame: Optional[int], num_frame: Optional[int] = 100) -> np.ndarray:
    "read video file, return some number of frame data (prefer small number) as numpy array"
    if num_frame is None:
        num_frame = 10   # keeping it small
    else:
        num_frame = num_frame
    frame_list = []    # get all frames here as list of np.array
    if start_frame is None:
        start_frame = 0    # read from beginning of video
    i= 0
    print(f"READING VIDEO DATA ...")
    print(f"opening this video file to read data: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps of {video_path}= {fps}")
    while(cap.isOpened()):
        ret, frame = cap.read()
        #print(frame, ret)
        if ret and i < (start_frame + num_frame):
            if i >= start_frame:
                frame_list.append(frame)
            else:
                pass   # don't add frames from earlier
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

def process_video(Exp_Num: Optional[int], 
                  filename: Optional[Union[list[str], str]],
                  vid_file: Optional[str],
                  ext: str = ".mp4", 
                  start_frame: int = 0, 
                  num_frames: int = 20
                  ):
    """ reading video and selecting data from starting frame for given number of frames
        inputs:
        filename: list of all files in Pyrodata.Files, normally holds all files from expreiment folder
        my_vid: exact name of video file
        ext: extension of video file - for now .mp4
        start_frame: to read starting from
        num_frame: no. of video frames to read
        output:
        b_i: np.ndarray data from the video for number of frames(N) as (N, h, w, c) data format
        b_0: apatial average of the single frames based on SBP formaula as np.array[num_frames] format
    """
    if filename is None:
        # filenames have to be generated here
        Exp_Num = 1
        exp = Files(Exp_Num)
        exp_files = exp.files()
        print(f"{exp_files}")
    elif type(filename) is str:
        exp_files = list(filename)    # only one file passed
    else:
        exp_files = filename
    #print(f"expriment files considered for processing video: {exp_files}")
    if ext == ".mp4":
        #reading the video data
        my_video_file: str = vid_file + ext   # example video_file.mp4
        #video_path = get_filename_with_ext(exp_files, ext)   # getting data from first .mp4 file found
        video_path = get_file_from_filelist(exp_files, my_video_file)
        print(f"file received with file selector {video_path}")
        #play_video(video_path=video_path)
        b_i = read_video(video_path=video_path, start_frame=start_frame, num_frame= num_frames)    # getting frames in an array, keep small number
        info = None
    elif ext ==".mraw":
        cihx_file = get_filename_with_ext(filelist= filename, ext=".cihx")
        print(f'Reading data from video file: {os.path.splitext(cihx_file)[0] + ".mraw"}')
        b_i, info = load_video(cih_file= cihx_file)
        print(f"b_i received from .mraw file, matrix shape: {b_i.shape}")
    else:
        raise Exception("reading video format of this extension not implemented")
    b_i_s = np.nan_to_num(b_i[:num_frames], copy=True, nan=1)  # reducing size
    b_0s = np.zeros(num_frames)
    print(f"trimmed b_i shape: {b_i_s.shape}")
    print(f"b_0s shape: {b_0s.shape}")
    for snap in range(num_frames):
        print(f"For snapshot: {snap+1}")
        b_0s[snap] = get_b0(b_i_s[snap])   # passing a single frame here, updated code
    b_0_over_t = get_b0(b_0s)
    return b_i_s, b_0s, b_0_over_t, info

def get_b0(b_i: np.ndarray):
    "calculating b_0 from the b_i (i = surface element) of the camera frame, b_i is 2D (h, w) or 3D (h, w, c) array"
    print(f"Intermediate matrices for b_0 calculation:")
    #a = np.sum(np.multiply(np.log(b_i), b_i), axis=0)
    bi_log = np.log(b_i)   # log of brightness
    print()
    bi_log = np.nan_to_num(bi_log, copy=False, nan=-9999, neginf=-33333333)   # getting rig of NA and -inf
    bi_log = np.ceil(bi_log)
    a = np.sum(np.multiply(bi_log, b_i))   # sigma(bi*log(bi))
    print(f"a: {a}\n")
    #b = np.sum(b_i, axis=0)
    b = np.sum(b_i)
    print(f"b: {b}\n")
    #c = np.divide(a, b)
    c = a/b
    print(f"c: {c}\n")
    d = np.exp(c)
    print(f"d: {d}\n")
    print(f"b_0 returned !")
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
    FILES: FileList = ["video_file.mp4", "spectra.csv", "info.json"]
    myData = Pyrodata(exp_number=EXP_No, filenames=FILES)
    raw_spectra = myData.read_spectral_data()
    #print(raw_spectra.head())
    #myData.plot_spectra()
    sbp1 = SBP(myData)
    sbp1.add_radiation_cols()
    #print(sbp1.spectra)
    sbp1.plot_raw_spectra(["Wavelength", "Intensity"])
    sbp1.plot_raw_spectra(["x", "y"])
    sbp1.get_spectral_frames()
    sbp1.calc_framewise_rad_vars(use_smoothed_y=True, smooth_window=10)
    sbp1.plot_framewise_spectra(["x", "y_smooth"])
    sbp1.plot_T0s(3, 10000)
    print(f"framewise T0s: {sbp1.T_0}")
    #sbp1.video_brightness_data(None)
    #sbp1.plot_brightness(sbp1.b0)
    #print(f"brightness processed data sizes for b_i, b_0: {sbp1.bi.shape, sbp1.b0.shape}")
    #print(f'pyrodata experiment info: {myData.info["video_channel"]}')
    T_i, T_j, b_i, b_0, b_0_over_t, info = sbp1.vid_seq_temperature(spectral_frame_num= 3, test= False)
    show_image(image_data=b_i[10])
    print(f"max in b_i[10]: {np.max(b_i[10])}")
    #sbp1.plot_brightness(T_i[10])
    Ti_bias = np.zeros(T_i.shape)
    Ti_bias[:] = 1500.
    T_i_corrected = np.subtract(T_i, Ti_bias)
    show_masked_image(image_data=T_i[10], cmap='plasma', vmin_max=(2500, 3000))
    print(f"Overall brightness over time t: {b_0_over_t}")
    print(f"Individual frame b_0: {b_0}")
    print(f"Framewise reference temperaure: {T_j}")