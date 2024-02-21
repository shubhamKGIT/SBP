from sbp import SBP, show_image
from pyrodata import Pyrodata
from video import frame_sync_vid_seq
from unittest import TestCase
from mraw import load_video, get_cih, load_images
from files import get_filename_with_ext


my_sbp = SBP(Pyrodata(1, None, ["video_file.mp4", "spectra.csv", "info.json"]))
#print(my_sbp.data_holder.file_holder.files())

start_frame, num_frames = frame_sync_vid_seq(my_sbp.data_holder.info, 3)
print(start_frame, num_frames)
files = my_sbp.data_holder.file_holder.files()
cih_file = get_filename_with_ext(files, ".cihx")
#print(cih_file)
cih = get_cih(cih_file)
print(cih["Date"])
print(cih.keys())
N = cih["Total Frame"]
images = load_images(get_filename_with_ext(files, ".mraw"), 1024, 1024, N, 16, False)
print(images[300].shape)
show_image(images[500])