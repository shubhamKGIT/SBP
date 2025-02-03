
from sbp import SBP
from pyrodata import Pyrodata
from files import Files
from video_processing_utils import VideoFilesBase

EXP = 21
files = Files(exp_number= EXP).files()
pyro_handle = Pyrodata(exp_number= EXP, filenames= files)
sbp_handle = SBP(myExperiment=pyro_handle)
sbp_handle.add_radiation_cols()
sbp_handle.get_spectral_frames()
# sbp_handle.plot_raw_spectra()
sbp_handle.calc_framewise_rad_vars(use_smoothed_y=True, smooth_window=50)
sbp_handle.plot_framewise_spectra()
print(sbp_handle.T_0[30:])
sbp_handle.plot_T0s(which_frame=40, T_lim=4000)

video = VideoFilesBase(exp=21, format=None)
video.read_and_plot(300)