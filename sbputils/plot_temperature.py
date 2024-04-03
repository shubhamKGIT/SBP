
from videoUtils import read_frames_from_pkl_dump, write_video_from_frames, apply_color, show_image, show_masked_image, write_color_vid_to_file
from gray2color import gray_2_color, get_mpl_cmap_custom_palette, get_mpl_colormap
from video_processing_utils import FramesHolder
import matplotlib as mpl
from matplotlib import colors
from matplotlib import pyplot as plt
from files import Files
import os
import numpy as np
import cv2

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)
    
def show_colored(T_i):
    img = apply_color(T_i[10], get_mpl_colormap("hot", is_custom=False, norm=None, vmin=2500, vmax=3000))
    cv2.imshow("with palette colormap", img)
    #vid_frames = gray_2_color(T_i, get_mpl_colormap(cmap=None, is_custom = True, norm=None, vmin=2500, vmax=3000))
    #cv2.imshow("with palette colormap", vid_frames[10])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    EXP_NUM = 1
    TEMP_DUMP = "temperature.pkl"
    TEMP_VIDEO = "temperature.avi"
    TEMP_COLOR_VIDEO = "temperature_colored.avi"
    exp = Files(EXP_NUM)
    exP_files = exp.files()
    exp_data_folder = exp.expFolder
    temp_dump_file = os.path.join(exp_data_folder, TEMP_DUMP)
    T_i = read_frames_from_pkl_dump(temp_dump_file)
    Ti_holder = FramesHolder(frames = T_i, is_Normalised=False)
    Ti_holder.normalise_brigthess()
    print(Ti_holder)
    show_masked_image(T_i[10], mask_threshhold=300, vmin_max=(2400, 3000))
    # writing temperaure video here
    Ti_output_filepath = os.path.join(exp_data_folder, TEMP_VIDEO)
    write_video_from_frames(Ti_holder.brightenss, fps = 10, output_filepath=Ti_output_filepath, out_type=".avi")
    norm = colors.Normalize(vmin=1800, vmax=3000, clip=1700)  # can build a custom norm and pass here
    midnorm = MidpointNormalize(vmin=120, vcenter=180, vmax=255)
    Ti_holder.colorfy(get_mpl_colormap("hot", is_custom=False, norm = midnorm, vmin=0, vmax=255))
    Ti_colored_output_filepath = os.path.join(exp_data_folder, TEMP_COLOR_VIDEO)
    write_color_vid_to_file(Ti_holder.frames_color, fps=10, output_filepath=Ti_colored_output_filepath, out_type=".avi")
   