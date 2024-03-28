import cv2
import numpy as np
from files import Files, get_filename_with_ext

# Function to read monochrome .mraw file
def read_mraw(filename):
    with open(filename, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint16)
    return raw_data

# Load .mraw file
video_file = Files(2)
#print(video_file.files())
RAW_FILE = get_filename_with_ext(video_file.files(), ".mraw")
print("raw files")
print(RAW_FILE)
filename = RAW_FILE
raw_data = read_mraw(filename)

# Reshape raw data into image format (assuming image dimensions)
width = 1024  # example width
height = 1024  # example height
num_frames = len(raw_data) // (width * height)
image_data = raw_data.reshape((num_frames, height, width))

# Apply artificial gain
gain_factor = 8.0
image_data_gain = image_data * gain_factor

# Clip pixel values to [0, 65535] range
image_data_gain = np.clip(image_data_gain, 0, 65535)

# Optionally, rescale pixel values to [0, 1] range for visualization
image_data_gain_norm = image_data_gain / 65535.0

# Display or save the adjusted image data
# For example, to display the first frame:
cv2.imshow('Adjusted Image', image_data_gain_norm[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the adjusted image to a new .mraw file
# Be cautious when saving the adjusted data back to 16-bit .mraw format
# as the gain may cause pixel values to exceed the valid range.
