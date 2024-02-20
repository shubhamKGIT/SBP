from sbp import SBP
from pyrodata import Pyrodata
from video import frame_sync_vid_seq
from unittest import TestCase


my_sbp = SBP(Pyrodata(1, None, ["video_file.mp4", "spectra.csv", "info.json"]))
#print(my_sbp.data_holder.file_holder.files())

start_frame, num_frames = frame_sync_vid_seq(my_sbp.data_holder.info, 3)
print(start_frame, num_frames)