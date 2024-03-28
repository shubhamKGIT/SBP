from mraw import load_video, save_mraw
from files import Files, get_filename_with_ext
import matplotlib.pyplot as plt
import os
import pathlib
import numpy as np
import cv2
import PIL.Image as Image
import seaborn as sns
import skimage


def save_video(frames, savepath):
    print(savepath)
    SIZE = (480, 640)  #w, h selected
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(savepath, fourcc, 20., SIZE)
    for i in range(250, 750):
        #print(i)
        #grey = (frames[i, 300:940, 360:840]/256).astype(np.uint8)
        cropped = frames[i, 300:940, 360:840]
        cm = plt.get_cmap('twilight')
        coloured_img = cm(cropped)
        #image = cv2.applyColorMap(grey, cv2.COLORMAP_HOT)
        #print(f"image_shape: {image.shape}")
        writer.write(coloured_img[:, :,:3])
        #print("written")

    writer.release()
    plt.imshow(coloured_img)
    plt.show()
    #save_mraw(frames[1000:1100, :, :])

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def contrast():
    # Open a typical 24 bit color image. For this kind of image there are
    # 8 bits (0 to 255) per color channel
    img = cv2.imread('mandrill.png')  # mandrill reference image from USC SIPI
    s = 128
    img = cv2.resize(img, (s,s), 0, 0, cv2.INTER_AREA)  
    font = cv2.FONT_HERSHEY_SIMPLEX
    fcolor = (0,0,0)
    blist = [0, -127, 127,   0,  0, 64] # list of brightness values
    clist = [0,    0,   0, -64, 64, 64] # list of contrast values
    out = np.zeros((s*2, s*3, 3), dtype = np.uint8)
    for i, b in enumerate(blist):
        c = clist[i]
        print('b, c:  ', b,', ',c)
        row = s*int(i/3)
        col = s*(i%3)  
        print('row, col:   ', row, ', ', col)   
        out[row:row+s, col:col+s] = apply_brightness_contrast(img, b, c)
        msg = 'b %d' % b
        cv2.putText(out,msg,(col,row+s-22), font, .7, fcolor,1,cv2.LINE_AA)
        msg = 'c %d' % c
        cv2.putText(out,msg,(col,row+s-4), font, .7, fcolor,1,cv2.LINE_AA)
        cv2.putText(out, 'OpenCV',(260,30), font, 1.0, fcolor,2,cv2.LINE_AA)
    cv2.imwrite('out.png', out)

def process_pyro_mraw():
    pass

def show_image(snap):
    plt.imshow(snap, cmap="Reds_r", vmin=0)    # N, h, w
    plt.show()

def show_images_as_heatmap(snap):
    GAIN = 2
    sns.heatmap(snap*GAIN, vmin = 1, vmax = 1024)
    plt.show()

def enhance_using_F_factor(img, max_scale):
    scaled = (snap + 255)/ 255
    scaling = (max_scale - snap)/ max_scale
    f = scaled/ scaling
    enhanced = f*(snap - 128) + 128
    return enhanced

if __name__=="__main__":
    
    EXP = Files(exp_number= 3)
    exp_files = EXP.files()
    print(exp_files)
    cihx_file = get_filename_with_ext(exp_files, ".cihx")
    print(cihx_file)
    frames, cihx = load_video(cih_file=cihx_file)
    print(f"bit depth = {cihx['Color Bit']}")
    print(f'frame rate: {cihx["Record Rate(fps)"]}')
    print(f"size of array: {frames.shape}")

    #Showing here
    GAIN = np.zeros(shape=(1024, 1024))  # not used for now
    #print(GAIN)
    snap = frames[500, 200:600, 400:800]
    #print(frames[999, 500:510, 500:502])
    print(f"range of values in snap = {(snap.min(), snap.max())}") 
    print(f"range of values in all frames = {(frames.min(), frames.max())}")
    
    # covert snap to 0 - 255 
    #snap = 255*(snap/snap.max())
    # adjusting brightenss
    bright_copy = snap + 200
    dark_copy = snap - 400
    dark_copy[dark_copy < 0 ] = 0

    # adjusting contrast
    high_contrast_copy = snap*8
    high_contrast_copy[high_contrast_copy > 1800] = 1800
    high_contrast_copy[high_contrast_copy == 0] = -1
    low_contrast_copy = snap*1.5

    # linear 
    linear_mapped = snap*8 + 200
    linear_mapped[linear_mapped > 1800] = 1800

    # f_enhanced
    f_enhanced = enhance_using_F_factor(snap, 1400)

    fig, ax = plt.subplots(3, 2, figsize = (8, 8))
    VMIN = 0
    VMAX = 1300*4
    ax[0, 0].imshow(high_contrast_copy, vmin = VMIN, vmax = VMAX/4)
    ax[0, 1].imshow(low_contrast_copy, vmin = VMIN, vmax = VMAX/4)
    ax[1, 0].imshow(bright_copy, vmin = VMIN, vmax = VMAX/6)
    ax[1, 1].imshow(snap*14, vmin = VMIN*2, vmax = VMAX*2, cmap="gray")
    ax[2, 0].imshow(f_enhanced, vmin = 0, vmax = 255)
    ax[2, 1].imshow(linear_mapped, vmin = VMIN, vmax = VMAX)
    plt.legend()
    plt.show()
    Image.Image.show(f_enhanced)

    #show_images_as_heatmap(snap)
    """
    with sns.axes_style('dark'):
        skimage.io.imshow(snap, cmap=plt.cm.gray)
        plt.show()"""

    #plt.hist(snap.ravel(), bins=range(256), fc='k', ec='k')
    #plt.imshow(snap, cmap="gray", clim=(50, 500))
    #-plt.show()

    # enhanced
    alpha = 8
    beta = 0
    enhanced = cv2.convertScaleAbs(snap, alpha=alpha, beta=beta)
    plt.imshow(enhanced)
    plt.show()

    """grey = (frames[1000, 0:800, 400:800]).astype(np.uint8)
    cm = plt.get_cmap('twilight')
    coloured_img = cm(grey)
    #image = cv2.applyColorMap(grey, cv2.COLORMAP_HOT)
    plt.imshow(coloured_img)
    #plt.colorbar()
    plt.show()
    Image.fromarray((coloured_img[:, :, :3] * 255).astype(np.uint8)).save('test.png')"""

    #saving
    SAVE_PATH = os.path.join(EXP.dataDir, "001", "saved.avi",)
    #save_video(frames, SAVE_PATH)   # has issues with CODECS, needs work
    

