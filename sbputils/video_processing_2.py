from files import Files, get_filename_with_ext
from mraw import load_video
import PIL.Image as Image
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import os
import pickle
import numpy as np
import cv2

def show_image(some_image, cmap="gray", vmin =0, vmax=2**12):
    plt.imshow(some_image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.show()

class Video():
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
    
def read_and_dump(exp_num = 3):
    "main content abstracted here"
    exp_3_vid = Video(exp=exp_num)
    print(f"video_file: {exp_3_vid.vid_file}, data folder: {exp_3_vid.data_folder}")
    #exp_3_vid.read_and_plot()
    FILE_DUMP = "frames_dump.pkl"
    print(f"data folder: {exp_3_vid.data_folder}")
    exp_3_vid.read_and_save(FILE_DUMP)

def read_frames_from_pkl_dump(filename):
    "serialised data read here using pickle.load till file is having info"
    data = []
    with open(filename, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    """uint16 assigned to dtype, values can be from 0 to (2**16 -1) as we do not have np.uint12
    video data is either native camera bit depth 12 bit ot 16 bit based on export here
    """
    return np.array(np.squeeze(data, axis = 0), dtype=np.uint16)   # data has extra dimention so correcting it here

def test_write_mp4(frames, output_filepath):
    size = 720*16//9, 720
    duration = 2
    fps = 25
    out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for _ in range(fps * duration):
        data = np.random.randint(0, 256, size, dtype='uint8')
        out.write(data)
    out.release()

def write_mp4_from_frames(frames, output_filepath, out_type):
    #size = frames.shape[1], frames.shape[2]   # this sequence is reversed when writing the mp4
    #size =  size = 720*16//9, 720
    size = 960, 720   # aspect ratio of 4:3 used here
    fps = 25    # we had saved 200 frames so viewo will be of 8 sec
    w_start = int(frames.shape[1] - size[1])/2
    print(f"frame write width starts at: {w_start}")
    if out_type == ".mp4":
        out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    elif out_type == ".avi":
        out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MJPG'), fps, (size[1], size[0]), False)
    else:
        NotImplementedError("Type not implemented")
    for i in range(frames.shape[0]):
        #out.write(frames[i, 32:32+size[0], 152:152+size[1]].astype(np.int8))  # when writing 12 bit video, convert to 8 bit
        out.write(frames[i, 32:32+size[0], 152:152+size[1]])  # when frames are already modified to 0-255 range
    out.release()

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


if __name__=="__main__":
    # reading data and saving numpy in a file
    EXP_NUM = 3
    #read_and_dump(EXP_NUM)
    # REading data from file
    exp_3_vid = Video(exp=EXP_NUM)
    FILE_DUMP = "frames_dump.pkl"
    data_filepath = os.path.join(exp_3_vid.data_folder, FILE_DUMP)
    frames = read_frames_from_pkl_dump(data_filepath)
    print(f"read data from pickle: \n : {frames} \n shape: {frames.shape}")
    print(f"dtype in frames: {frames.dtype}")
    snap = frames[99, 32:32+960, 152:152+720]
    #snap_norm = (frames[99, 32:32+960, 152:152+720]/ (2**12-1))*(2**8 -1)
    plt.imshow(snap/snap.max(), cmap="hot")
    plt.show()
    print(f"max item (brightness) value in video frames: {(frames).max()}")
    print(f"max item value in snap: {(snap).max()}")

    # writing mp4 after processing
    OUT_TYPE = ".mp4"
    TEST_OUT = "test_output.mp4"
    FILE_OUT = "frames_output.mp4" if OUT_TYPE==".mp4" else "frames_output.avi"
    test_out_filepath = os.path.join(exp_3_vid.data_folder, TEST_OUT)
    output_filepath = os.path.join(exp_3_vid.data_folder, FILE_OUT)
    #writing file
    test_write_mp4(frames, test_out_filepath)   # frames not used for now
    NORMALISED = False    # normalised to 8 bit to avoid weird artifacts
    if NORMALISED:   # are items between 0 and 255
        write_mp4_from_frames(frames, output_filepath, OUT_TYPE)
    else:
        print(f"Normalising the frames from 0 - 255 before senting to output writer")
        frames_norm = np.array((frames/frames.max())*255, dtype=np.uint8)  # getting values from 0 - 255 of uint dtype for compatability
        print(f"normalised frames maximum item: {frames_norm.max()}, min: {frames_norm.min()}, dtype: {frames_norm.dtype}")
        write_mp4_from_frames(frames_norm, output_filepath, OUT_TYPE)

    