import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle
import cv2
import subprocess
import os
import glob
import pickle

def show_image(image_data: Optional[Union[np.array, np.ndarray]] = None, 
               cmap: str = 'hot'
               ):
    "early implmentation with default random image"
    if image_data is None:
        image_data = np.array([(1,2,3,4,5),(4,5,6,7,8),(7,8,9,10,11)])
    else:
        pass
    im = plt.imshow(image_data, 
                    cmap= cmap
                    )
    plt.colorbar(im)
    plt.show()

def show_masked_image(image_data: Optional[Union[np.array, np.ndarray]] = None,
                    mask_threshhold: float = 800.,
                    cmap: str = 'hot', 
                    vmin_max: tuple = (None, None)
                    ):
    "plotting images for SBP Temperature outputs"
    if image_data is None:
        image_data = np.array([(1,2,3,4,5),(4,5,6,7,8),(7,8,9,10,11)])
        mask_threshhold = 3
    else:
        pass
    fig, ax = plt.subplots()
    colormap = mpl.colormaps[cmap]
    palette = colormap.with_extremes(over='r', under='g', bad='k')
    array_masked = np.ma.masked_where(image_data < mask_threshhold, image_data)
    print(f"masked array info: {array_masked.min()}, {array_masked.max()}")
    vmin = vmin_max[0]
    vmax = vmin_max[1]
    bounds = np.linspace(vmin, vmax, 10)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=palette.N)
    #patch = Rectangle((300, 300), 200, 200)
    #patch = Circle((460, 200), radius=200, transform=ax.transData)
    im = ax.imshow(X = array_masked,
                    cmap=palette,
                    #norm=colors.Normalize(vmin=vmin, vmax=vmax),
                    norm = norm
                    )
    plt.colorbar(im)
    plt.show()

def write_frames_to_pkl_dump(frames: np.array, filepath: str):
    print(f"write data with size: {frames.shape} to {filepath}")
    try:
        frames.dump(filepath)
        print(f"frames dumped to pickle file")
    except:
        raise Exception("Could not write the data to the specified file")

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

def write_video_from_frames(frames, fps, output_filepath, out_type):
    "writes a gray video from the pyrolant expriment data of 1024 x 1024 pixels size to custom one"
    #size = frames.shape[1], frames.shape[2]   # this sequence is reversed when writing the mp4
    #size =  size = 720*16//9, 720
    # if FPS is 25 => if we had saved 200 frames, viewtime will be of 8 sec
    size = 960, 720   # aspect ratio of 4:3 used here   
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

def write_color_vid_to_file(frames, fps, output_filepath, out_type):
    "gets colour frames and writes it a video here"
    size = 960, 720   # custom size with width and height
    if out_type == ".mp4":
        out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    elif out_type == ".avi":
        out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MJPG'), fps, (size[1], size[0]), True)
    else:
        NotImplementedError("Type not implemented")
    
    for i in range(frames.shape[0]):
        out.write(frames[i, 32:32+size[0], 152:152+size[1], :])
    out.release()

def generate_video_from_pyplot_figures(frames, cmap, folder, vid_filename):
    "makes a video of colormapped N x w x h x c numpy array to video just, writes plotted image to folder and then uses ffmepeg"
    for i in range(len(frames)):
        plt.imshow(frames[i], cmap=cmap)
        plt.savefig(folder + "/file%02d.png" % i)

    os.chdir(folder)
    subprocess.call([
        'ffmpeg', '-framerate', '10', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)


def get_mpl_colormap(cmap_name = None, is_custom = False, norm = None, vmin = 0, vmax = 255):
    "using standard matplotlib cmap, get the colormap for applying color to gray frames in cv2"
    
    if is_custom:
        colormap = mpl.colormaps[cmap_name]
        palette = colormap.with_extremes(over='r', under='g', bad='b')
        bounds = np.linspace(vmin, vmax, 10)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors= palette.N)
    else:
        if isinstance(cmap_name, str):
            palette = plt.get_cmap(cmap_name)
            print(f"name of colormap passed, retrieving the color range here: {palette}, type: {type(palette)}")
        """elif isinstance(cmap_name, type(mpl.colors.LinearSegmentedColormap)):
            palette = cmap_name
        else:
            raise TypeError"""
        if norm is None:
            print(f"norm is not provided, calculating it using vmin, vmain; range passed for colormap: {vmin} to {vmax}")
            norm = plt.Normalize(vmin, vmax)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=palette)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]
    return color_range.reshape(256, 1, 3)

def get_mpl_cmap_custom_palette(cmap = None, norm = None):
    "using custom pallete of matplotlib cmap, get the colormap for applyting color to gray in cv2"
    if cmap is None:
        "default implementation is gray"
        cmap = plt.cm.gray.with_extremes(over='r', under='g', bad='b')  # same as pa letter
    sm = plt.cm.ScalarMappable(cmap = cmap)
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]
    return color_range.reshape(256, 1, 3)

def apply_color(gray_array, cmap = None, customized = False, norm = None, vmin = 0, vmax = 255):
    "apply color to gray frame in cv2 using matplotlib colormaps"
    img = cv2.applyColorMap(gray_array, get_mpl_colormap(cmap, customized, norm, vmin, vmax))
    return img


if __name__=="__main__":
    # compute some interesting data
    x0, x1 = -5, 5
    y0, y1 = -3, 3
    x = np.linspace(x0, x1, 500)
    y = np.linspace(y0, y1, 500)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 2
    print(Z.max, Z.min)
    #show_image(Z, "hot")
    # Set up a colormap:
    palette = plt.cm.gray.with_extremes(over='r', under='g', bad='b')
    # Alternatively, we could use
    # palette.set_bad(alpha = 0.0)
    # to make the bad region transparent.  This is the default.
    # If you comment out all the palette.set* lines, you will see
    # all the defaults; under and over will be colored with the
    # first and last colors in the palette, respectively.
    Zm = np.ma.masked_where(Z > 1.2, Z)

    # By setting vmin and vmax in the norm, we establish the
    # range to which the regular palette color scale is applied.
    # Anything above that range is colored based on palette.set_over, etc.

    # set up the Axes objects
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 5.4))

    # plot using 'continuous' colormap
    im = ax1.imshow(Zm, interpolation='bilinear',
                    cmap=palette,
                    norm=colors.Normalize(vmin=-1.0, vmax=1.0),
                    aspect='auto',
                    origin='lower',
                    extent=[x0, x1, y0, y1])
    ax1.set_title('Green=low, Red=high, Blue=masked')
    cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax1)
    cbar.set_label('uniform')
    ax1.tick_params(axis='x', labelbottom=False)

    # Plot using a small number of colors, with unevenly spaced boundaries.
    im = ax2.imshow(Zm, interpolation='nearest',
                    cmap=palette,
                    norm=colors.BoundaryNorm([-1, -0.5, -0.2, 0, 0.2, 0.5, 1],
                                            ncolors=palette.N),
                    aspect='auto',
                    origin='lower',
                    extent=[x0, x1, y0, y1])
    ax2.set_title('With BoundaryNorm')
    cbar = fig.colorbar(im, extend='both', spacing='proportional',
                        shrink=0.9, ax=ax2)
    cbar.set_label('proportional')

    fig.suptitle('imshow, with out-of-range and masked data')
    plt.show()

    #show_image()
    
    """a = np.random.random(size=(24, 24))*2000.
    print(a)
    show_masked_image(None, 2, "hot", (3, 11))
    show_masked_image(a, 100, "viridis", (400, 1900))"""
    print(f"values in Z: {Z.min()} to {Z.max()}")
    Z_norm = (Z - Z.min())/ (Z.max() - Z.min())
    print(f"values in Z_norm: {Z_norm.min()} to {Z_norm.max()}")
    Z_new = np.array(Z_norm*255., dtype=np.uint8)
    print(f"shape of new_Z = {Z_new.shape}, dtype= {Z_new.dtype}")
    show_image(Z_new, "gray")

    show_masked_image(Z_new, mask_threshhold=50, cmap="hot", vmin_max=(20, 200))
    CMAP="hot"
    colormap = mpl.colormaps[CMAP]
    palette = colormap.with_extremes(over='r', under='g', bad='b')
    print(f"palette: {palette}")
    vmin, vmax = 0, 255
    bounds = np.linspace(vmin, vmax, 10)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=palette.N)
    print(f"norm: {norm}")
    Z_colored = apply_color(gray_array = Z_new, customized=True, cmap="hot", vmin=vmin, vmax=vmax)
    print(f"shape of colored_Z: {Z_colored.shape}, min: {Z_colored.min()}, max: {Z_colored.max()}")
    plt.imshow(Z_colored, cmap=palette, norm=norm)
    plt.show()
