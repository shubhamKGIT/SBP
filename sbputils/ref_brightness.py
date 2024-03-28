import numpy as np
import cv2

def calculate_reference_brightness(frames):
    reference_brightness = []
    for frame in frames:
        # Convert frame to floating point for accurate calculations
        frame_float = frame.astype(float)

        # Calculate brightness at each pixel
        brightness = frame_float.flatten()
        log_brightness = np.log(brightness + 1)  # Add 1 to avoid log(0)

        # Sum of brightness multiplied by logarithm of brightness
        brightness_sum = np.sum(brightness * log_brightness)

        # Normalize by total sum of brightness values
        reference_brightness.append(np.exp(brightness_sum / np.sum(brightness)))

    return reference_brightness

# Function to read .mraw file
def read_mraw_file(file_path, num_frames, height, width):
    with open(file_path, 'rb') as f:
        # Read the binary data
        raw_data = np.fromfile(f, dtype=np.uint16)
    # Reshape raw data into frames
    frames = raw_data.reshape((num_frames, height, width))
    return frames

# Read .mraw file
mraw_file_path = 'your_video.mraw'  # Replace with the path to your .mraw file
height = 1024
width = 1024
num_frames = 5000
frames = read_mraw_file(mraw_file_path, num_frames, height, width)

# Calculate reference brightness for each frame
reference_brightness = calculate_reference_brightness(frames)

# Print or store the reference brightness values
for i, brightness in enumerate(reference_brightness):
    print(f"Frame {i+1}: Reference Brightness = {brightness}")

# Return exponential of reference brightness values as B_0
B_0 = np.exp(reference_brightness)