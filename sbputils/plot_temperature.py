
from videoUtils import read_frames_from_pkl_dump, write_video_from_frames, apply_color, show_image, show_masked_image, write_color_vid_to_file
from gray2color import gray_2_color, get_mpl_cmap_custom_palette, get_mpl_colormap
from video_processing_utils import FramesHolder
from files import Files
import os
import numpy as np
import cv2

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
    show_masked_image(T_i[10], mask_threshhold=1000, vmin_max=(2000, 3000))
    # writing temperaure video here
    Ti_output_filepath = os.path.join(exp_data_folder, TEMP_VIDEO)
    write_video_from_frames(Ti_holder.brightenss, fps = 10, output_filepath=Ti_output_filepath, out_type=".avi")
    Ti_holder.colorfy(get_mpl_colormap("hot", is_custom=False, norm = None, vmin=0, vmax=255))
    Ti_colored_output_filepath = os.path.join(exp_data_folder, TEMP_COLOR_VIDEO)
    write_color_vid_to_file(Ti_holder.frames_color, fps=10, output_filepath=Ti_colored_output_filepath, out_type=".avi")
   