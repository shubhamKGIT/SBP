import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle

def show_image(image_data: Optional[Union[np.array, np.ndarray]] = None, 
               cmap: str = 'hot'
               ):
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
    if image_data is None:
        image_data = np.array([(1,2,3,4,5),(4,5,6,7,8),(7,8,9,10,11)])
        mask_threshhold = 3
    else:
        pass

    fig, ax = plt.subplots()
    colormap = mpl.colormaps[cmap]
    palette = colormap.with_extremes(over='r', under='g', bad='k')
    array_masked = np.ma.masked_where(image_data < mask_threshhold, image_data)
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

def show_zoomed_part(Z):
     #x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original image
    fig, ax = plt.subplots()
    
    Z2 = np.zeros((150, 150))
    ny, nx = Z.shape
    Z2[30:30+ny, 30:30+nx] = Z
    axins = ax.inset_axes(
                        [0.1, 0.1, 0.4, 0.4],
                        #xlim=(x1, x2), 
                        #ylim=(y1, y2), 
                        xticklabels=[], 
                        yticklabels=[]
                        )
    axins.imshow(Z2, origin="lower")

    ax.indicate_inset_zoom(axins, edgecolor="black")

def frame_sync_vid_seq(info: dict, 
                       spectral_frame_num: int
                       ):
        video_channel = info["video_channel"]
        spectral_channel = info["spectral_channel"]
        if spectral_frame_num in [i+1 for i in range(spectral_channel["num_of_frames"])]:
            pass
        else:
            raise Exception("spectral frame number passed needs to be checked")
        fps = video_channel["hs_vid_fps"]
        vid_frames_per_ms = video_channel["hs_vid_fps"]/1000.
        vid_frames_per_spectra = spectral_channel["gap_between_spectral_frames"]*vid_frames_per_ms
        integration_time = spectral_channel["integration_time"]
        no_of_vid_frames = integration_time*vid_frames_per_ms
        vid_saved_from_frame = video_channel["video_saved_from_frame"]
        rel_vid_acquisition_frame = (spectral_channel["rel_aquisition_starts_at"]/1000.)*fps
        integration_starts_at_vid_frame = rel_vid_acquisition_frame + (spectral_frame_num - 1)*vid_frames_per_spectra - vid_saved_from_frame
        if integration_starts_at_vid_frame < 0:
             raise Exception("video data not saved for this spectra")
        else:
             pass
        return integration_starts_at_vid_frame, no_of_vid_frames

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
    #show_masked_image()
    a = np.random.random(size=(24, 24))*2000.
    print(a)
    show_masked_image(None, 2, "hot", (3, 11))
    show_masked_image(a, 100, "viridis", (400, 1900))


    