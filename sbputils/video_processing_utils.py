from files import Files, get_filename_with_ext
from mraw import load_video
from videoUtils import read_frames_from_pkl_dump, write_video_from_frames, test_write_mp4, write_color_vid_to_file
import PIL.Image as Image
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import os
import pickle
import numpy as np
import cv2
import scipy.ndimage as ndimage
import skimage
from skimage import exposure, util
import imageio

def show_image(some_image, cmap="gray", vmin =0, vmax=2**12, title = "image"):
    "default values assuming 12 bit color depth in video brightness data, pass custom"
    if cmap is not None:
        plt.imshow(some_image, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(some_image)
    plt.title(title)
    plt.show()

def invert_image(image):
    "can use np.invert or skimage.util.invert"
    return util.invert(image)

def zoom_image(image, factor: int):
    return ndimage.zoom(image, factor)

class VideoFilesBase:
    def __init__(self, exp: int = 3, format = None):
        self.exp_num = exp
        self.experiment_holder = Files(self.exp_num)
        self.exp_files = self.experiment_holder.files()
        self.data_folder = self.experiment_holder.dataDir.joinpath(self.experiment_holder.exp_num)
        if format is None:
            self.vid_file = get_filename_with_ext(self.exp_files, ".cihx")
        else:
            raise NotImplementedError("Implemeted only for mraw")
    
    def read_and_plot(self):
        frames, cihx_dict = load_video(self.vid_file)
        bit_depth = cihx_dict["Color Bit"]
        max_brightness = 2**bit_depth
        fps = cihx_dict["Record Rate(fps)"]
        snap = frames[800, :, :]
        print(f'bit depth: {cihx_dict["Color Bit"]}')
        print(f'frame rate: {cihx_dict["Record Rate(fps)"]}')
        print(f"video data read shape: {frames.shape}")
        print(f"probing data, raw frames max brightness: {snap.max()}, with gain of 12: {(snap*12).max()}")
        show_image(snap, vmin=0, vmax=max_brightness)
        show_image(snap*12, vmin=0, vmax=max_brightness)

    def read_and_save(self, filename: str = "frames_dump.pkl"):
        "dumping to a pickel file to be read later"
        frames, cihx_dict = load_video(self.vid_file)
        self.file_path = os.path.join(self.data_folder, filename)
        print(f"saving only some frames as there was memory error")
        frames[400:600].dump(self.file_path)
    
def main_read_video_and_dump(exp_num = 3):
    "wrapper function to read data for a given experiment and dump the data in pickle file to be read later"
    exp_3_vid = VideoFilesBase(exp=exp_num)
    print(f"video_file: {exp_3_vid.vid_file}, data folder: {exp_3_vid.data_folder}")
    #exp_3_vid.read_and_plot()
    FILE_DUMP = "frames_dump.pkl"
    print(f"data folder: {exp_3_vid.data_folder}")
    exp_3_vid.read_and_save(FILE_DUMP)



class FramesHolder:
    "to hold brighness data and conducts operation like normalising, increasing brightness, contrast etc."
    def __init__(self, frames: np.array, is_Normalised: bool = False):
        "assume frames has monocrome brightness data as N X H X W"
        self.frames = frames
        self.is_normalised = is_Normalised
    
    def __repr__(self):
        return f"frames size: {self.frames.shape}; is it normalised: {self.is_normalised}, \n item min: {self.frames.min()}, item max: {self.frames.max()}"
    
    def scale_brightness(self, factor: float = 1.5):
        """assumes frames are already normalised to 0-255 range,
        this helps in contrast too, directly value of factor = 2.5 gives best value of video visualisation compared to other methods
        """
        print(f"Before increasing brigthenss, max item value: {self.frames.max()}, dtype: {self.frames.dtype}")
        self.frames = self.frames*factor
        self.frames[self.frames>255] = 255
        print(f"After increasing brigthenss, max item value: {self.frames.max()}, dtype: {self.frames.dtype}")
        self.frames = np.array(self.frames, dtype=np.uint8)  # changing float to uint8 again
        #print(f"Final after changing type to uint8, max item value: {self.frames.max()}, dtype: {self.frames.dtype}")

    def normalise_brigthess(self):
        "changing uint16 to uint8 and values ranging from 0 - 255, standard for svi, mp4"
        try:
            if not self.is_normalised:
                print(f"Before normalising brightness, max item value: {self.frames.max()}, dtype: {self.frames.dtype}")
                self.frames = np.array((self.frames/self.frames.max())*255, dtype=np.uint8)
                print(f"After normalising brightness, max item value: {self.frames.max()}, dtype: {self.frames.dtype}")
                self.is_normalised = True
            else:
                print(f"Items were already normalised, not doing anything for now, item value range: {self.frames.min()} - {self.frames.max()}")
        except:
            raise Exception("Something went wrong in normalising")
    
    def sharpen(self):
        alpha = 30
        for i in range(frames.shape[0]):
            blurred_frame = ndimage.gaussian_filter(self.frames[i, :, :], 3)
            filter_blurred = ndimage.gaussian_filter(blurred_frame, 1)
            sharpened = blurred_frame + alpha*(blurred_frame - filter_blurred)
            self.frames[i, :, :] = sharpened
    
    def denoise(self):
        for i in range(frames.shape[0]):
            self.frames[i, :, :] = ndimage.median_filter(self.frames[i, :, :], 3)
    
    def alpha_beta_enhance(self, alpha: float = 2, beta: float = 50):
        "alpha is increase brightness by scaling, beta is brightness shift"
        self.frames = np.array(np.clip(self.frames*alpha + beta, 0, 255), np.uint8)

    def gamma_correction(self, gamma):
        """raised normalised intensity to power of gamma before scaling back to 0 - 255 range,
        check source:
        https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        """
        self.frames = np.array(np.power(self.frames/255., gamma) *255, dtype=np.uint8)

    def increase_contrast_with_f_factor(self, C):
        """using method from: https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
        C is the contrast (range = -255 to 255)
        """
        f: float =  ((255 + C)/255.)*(259./ (259 - C))  # contrast correction factor, f
        print(f"f factor value: {f}")
        self.frames = np.array(np.multiply((self.frames - 128), f) + 128, dtype=np.uint8)
        print(f"After f factor, frame item value are -  min: {self.frames.min()}, max: {self.frames.max()}, dtype: {self.frames.dtype}")

    def enhance_contrast(self):
        "enhance contrast frame by frame using standard or adaptive histogram enhancement (CLAHE)"
        for i in range(frames.shape[0]):
            #self.frames[i, :, :] = exposure.equalize_adapthist(self.frames[i, :, :], clip_limit=0.03)
            self.frames[i, :, :] = exposure.equalize_hist(util.invert(self.frames[i, :, :]))
    
    def enhance_contrast2(self):
        "was done for image to image, trying here for whole video"
        percentiles = np.percentile(self.frames, (0.5, 99.5))
        # array([ 1., 28.])
        self.frames = exposure.rescale_intensity(self.frames,
                                            in_range=tuple(percentiles))


    def plot_histogram(self):
        "plot histogram for sample image from the frames"
        plt.hist(self.frames[99, :, :].flatten(), bins=256, range=(0,256))
        plt.title('Histogram of a Low Contrast Image:')
        plt.xlabel('Pixel Value')
        plt.ylabel('Number of Pixels')
        plt.show()
    
    def zoom(self):
        "zooms around flame but spoils sharpness quite a bit"
        FACTOR = 2.0   # with this changing, the range of pizels passed will have to be changes
        for i in range(frames.shape[0]):
            self.frames[i, :, :] = zoom_image(self.frames[i, 256:(1024-256), 256:(1024-256)], FACTOR)
    
    def colorfy(self, custom_map=cv2.COLORMAP_JET):
        #self.frames = cv2.cvtColor(self.frames, cv2.COLOR_GRAY2BGR)  # cvtColor just makes copy of grey brightness to r, g, b
        self.frames_color = np.array(np.empty(shape=(self.frames.shape + (3,))), dtype = np.uint8)
        print(f"making RGB frames from gray brightnes data using selected colormap")
        for i in range(self.frames.shape[0]):
            self.frames_color[i] = cv2.applyColorMap(self.frames[i], custom_map)
        #self.frames_color = np.array(self.frames_color, dtype=np.uint8)
        print(f"colored frames made, use {__class__.__name__}.frames_color to access it")
        print(f"""info about coloured frames:
              size = {self.frames_color.shape},
              min: {self.frames_color.min()},
              max: {self.frames_color.max()}"""
              )
        
    @property
    def brightenss(self):
        return self.frames

def test_imshow_range():
    a = np.random.randint(-128, 128, (1280, 720))
    plt.imshow(a)
    plt.title("[-128, 128 range]")
    plt.show()

    b = np.random.random((1280, 720))
    plt.imshow(b)
    plt.title("[0, 1 range]")
    plt.show()
    
    c = np.random.randint(0, 255, (1280, 720))
    plt.imshow(b)
    plt.title("[0, 255 range]")
    plt.show()

def check_thresholded(my_frames: FramesHolder):
    _, thresh = cv2.threshold(my_frames.brightenss[99], 50, 255, cv2.THRESH_TOZERO)
    #cv2.imshow("using cv2", my_frames.brightenss[99])
    cv2.imshow("using cv2", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    # reading data and saving numpy in a file
    EXP_NUM = 3
    #main_read_video_and_dump(EXP_NUM)
    # REading data from file
    exp_vid_file_holder = VideoFilesBase(exp=EXP_NUM)
    FILE_DUMP = "frames_dump.pkl"
    data_filepath = os.path.join(exp_vid_file_holder.data_folder, FILE_DUMP)
    frames = read_frames_from_pkl_dump(data_filepath)
    print(f"read data from pickle: \n : {frames} \n shape: {frames.shape}")
    print(f"dtype in frames: {frames.dtype}")
    snap = frames[99, 32:32+960, 152:152+720]
    #snap_norm = (frames[99, 32:32+960, 152:152+720]/ (2**12-1))*(2**8 -1)
    plt.imshow(snap/snap.max(), cmap="hot")
    plt.title("0 - 1 values range in gray image data, colormap: hot")
    plt.show()
    imageio.imsave(os.path.join(exp_vid_file_holder.data_folder, "pyro_img.png"), snap)
    print(f"max item (brightness) value in video frames: {(frames).max()}")
    print(f"max item value in snap: {(snap).max()}")

    # writing gray mp4 after processing
    OUT_TYPE = ".mp4"
    TEST_OUT = "test_output.mp4"
    FILE_OUT = "frames_output.mp4" if OUT_TYPE==".mp4" else "frames_output.avi"
    test_out_filepath = os.path.join(exp_vid_file_holder.data_folder, TEST_OUT)
    output_filepath = os.path.join(exp_vid_file_holder.data_folder, FILE_OUT)
    #writing file
    test_write_mp4(frames, test_out_filepath)   # frames not used for now
    NORMALISED = False    # normalised to 8 bit to avoid weird artifacts
    if NORMALISED:   # are array items between 0 and 255
        write_video_from_frames(frames, 25, output_filepath, OUT_TYPE)
    else:
        print(f"Normalising the frames from 0 - 255 before senting to output writer")
        frames_norm = np.array((frames/frames.max())*255, dtype=np.uint8)  # getting values from 0 - 255 of uint dtype for compatability
        print(f"normalised frames maximum item: {frames_norm.max()}, min: {frames_norm.min()}, dtype: {frames_norm.dtype}")
        write_video_from_frames(frames_norm, 25, output_filepath, OUT_TYPE)
    
    # working with Frames class and using it to change brightness
    my_frames = FramesHolder(frames)
    print(my_frames)
    my_frames.normalise_brigthess()
    # test the enhancement methods here
    my_frames.scale_brightness(factor=2.5)
    #my_frames.gamma_correction(1.5)
    #my_frames.increase_contrast_with_f_factor(C=20)
    #my_frames.alpha_beta_enhance(1.5, 50)
    #my_frames.plot_histogram()
    #my_frames.zoom()  # spoils sharpness a bit
    #my_frames.enhance_contrast2()  # did not work with CLAHE, maybe needs colour reversal
    #my_frames.plot_histogram()
    #my_frames.sharpen()
    # testing what happend to frames data
    FILE_OUT = "frameholder_test_video.avi"
    out_path = os.path.join(exp_vid_file_holder.data_folder, FILE_OUT)
    palette = plt.cm.gray.with_extremes(over='r', under='g', bad='b')  # using palette as in SBP
    show_image(my_frames.brightenss[99, :, :], cmap=palette, vmin=30, vmax=250, title="0 - 255 range values gray image with gray custom palette")
    #show_image(zoom_image(my_frames.brightenss[99, 256:(1024-256), 256:(1024-256)], 2.0), vmin=0, vmax=255)
    #zoomed_snap = zoom_image(my_frames.brightenss[99, 256:(1024-256), 256:(1024-256)], 2.0)
    #print(f"shape of zoomed snapshot: {zoomed_snap.shape}")
    #show_image(invert_image(my_frames.brightenss[99, :, :]), vmin=0, vmax=255)
    print(f"shape of frames: {my_frames.brightenss.shape}")
    #write_video_from_frames(my_frames.brightenss, fps=10, output_filepath=out_path, out_type=".avi")
    
    COLOR_VID = "flame_in_color_2.avi"
    RED_CNL = "red_channel.avi"
    color_out_path = os.path.join(exp_vid_file_holder.data_folder, COLOR_VID)
    only_red_out_path = os.path.join(exp_vid_file_holder.data_folder, RED_CNL)
    my_frames.colorfy(cv2.COLORMAP_HOT)
    #show_image(my_frames.frames_color[99, :, :], cmap=None, title="colorfield with cmap: hot, all channels")
    #show_image(my_frames.frames_color[99, :, :, 2], cmap="hot", title="colorfield with cmap: hot, single channels, plotted as hot")
    write_color_vid_to_file(my_frames.frames_color, fps=10, output_filepath=color_out_path, out_type=".avi")
    #check_thresholded(my_frames)
    
    
