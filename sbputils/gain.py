import cv2
import numpy as np
from files import Files, get_filename_with_ext
from mraw import load_video

# Function to read monochrome .mraw file
def read_mraw(filename):
    with open(filename, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint16)
        # Reshape raw data into image format (assuming image dimensions)
        width = 1024  # example width
        height = 1024  # example height
        num_frames = len(raw_data) // (width * height)
        video_array = raw_data.reshape((num_frames, height, width))
    return video_array

def apply_gain(video_data, gain_factor): 
    # Apply artificial gain
    gain_factor = 8.0
    image_data_gain = np.dot(video_data, gain_factor)
    # Clip pixel values to [0, 65535] range, 16 bit data
    image_data_gain = np.clip(image_data_gain, 0, 65535)
    # Optionally, rescale pixel values to [0, 1] range for visualization
    image_data_gain_norm = image_data_gain / 65535.0
    return image_data_gain_norm

def display_with_gain(image_data_gain_norm, frame = 0):
    """Display or save the adjusted image data with gain
    Optionally, save the adjusted image to a new .mraw file
    Be cautious when saving the adjusted data back to 16-bit .mraw format
    as the gain may cause pixel values to exceed the valid range.
    """
    cv2.imshow('Adjusted Image', image_data_gain_norm[frame])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    # Load .mraw file
    EXP = 21
    FRAME_TO_VIEW = 10
    video_file = Files(EXP)
    print(video_file.files())
    raw_file = get_filename_with_ext(video_file.files(), ".cihx")
    filename = raw_file
    video_data = load_video(filename)
    # apply again
    iamge_data_gain_norm = apply_gain(video_data=video_data, gain_factor=8.0)
    # view with gain
    display_with_gain(image_data_gain_norm=iamge_data_gain_norm, frame=10)