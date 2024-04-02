import numpy as np
import scipy.ndimage as ndimage
import skimage
from PIL import Image
import cv2
from video_processing_utils import show_image
import matplotlib.pyplot as plt
import os
import glob
import pathlib
import matplotlib as mpl
import cmapy
from videoUtils import get_mpl_cmap_custom_palette, get_mpl_colormap, apply_color


def gray_2_color(frames, custom_mpl_cmap = None):
    "converts gray image or video to color one"
    if len(frames.shape) == 2:
        frames_color = cv2.applyColorMap(frames, cv2.COLORMAP_JET)
    elif len(frames.shape)  == 3:
        frames_color = np.empty(shape=(frames.shape + (3,)))
        for i in range(frames.shape[0]):
            if custom_mpl_cmap is None:
                frames_color[i] = cv2.applyColorMap(frames[i], cv2.COLORMAP_HOT)
            else:
                frames_color[i] = cv2.applyColorMap(frames[i], custom_mpl_cmap)
    else:
        raise NotImplementedError
    return frames_color

def test_gray_2_color(my_cmap):
    "test implementation of gray to color"
    gray_frames = np.array(np.random.rand(100, 16, 16)*255, dtype=np.uint8)   # 16 x 16 pixels used here
    print(f"synthetic frames shape: {gray_frames.shape}, dtype: {gray_frames.dtype}, min item: {gray_frames.min()}, max item: {gray_frames.max()}")
    show_image(gray_frames[10, : , :], vmin=0, vmax=255)
    color_img = gray_2_color(gray_frames[10])
    color_vid = gray_2_color(gray_frames)
    print(f"shape of colored frame : {color_img.shape}")
    print(f"shape of colored video : {color_vid.shape}")
    plt.imshow(color_img[1, :, :], cmap=my_cmap, vmin = 50, vmax=200)   # shows only read
    plt.show()
    #plt.imshow(color_img[:, :, :])
    #plt.show()

def colormapify(frame, colormap, vmin, vmax):
    "apply colormap and norm, gives RGBS output"
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(frame))

def random_vid_array():
    frames = np.array(np.random.rand(10, 16, 16)*255, np.uint8)
    print(f"length of array of 10 x 16 x 16: {len(frames)}")
    return frames

if __name__=="__main__":
    frames = random_vid_array()
    CMAP = "hot"
    palette = plt.cm.hot.with_extremes(over='r', under='g', bad='b')
    print(palette)
    gray = np.array(np.random.rand(1600, 1600)*255 + 50, dtype=np.uint8)
    print(f"shape of gray array using trial cmap:{gray.shape}, min: {gray.min()}, max: {gray.max()}")
    #print(gray)
    #colormap = plt.cm.hot
    norm = plt.Normalize(0, 255)
    # trying matplpotlib colormap with cv2
    color_mapped_img = palette(norm(gray))
    print(f"size of colormapped image: {color_mapped_img.shape}")
    #img = cv2.applyColorMap(gray, cmapy.cmap("viridis"))
    #img = cv2.applyColorMap(gray, get_mpl_colormap(vmin=0, vmax=255))
    #img = cv2.applyColorMap(gray, get_mpl_cmap_custom_palette())
    img = apply_color(gray_array=gray, cmap="hot", vmin=100, vmax=200)
    print(f"shape of converted array using trial cmap:{img.shape}, min: {img.min()}, max: {img.max()}")
    cv2.imshow("color image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(img, cmap=palette)
    plt.show()
    

    #test_gray_to_colour(palette)

    

