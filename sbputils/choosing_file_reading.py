import os
import xml.etree.ElementTree as ET
import numpy as np
import xmltodict

def find_files_in_experiment_folder(experiment_folder):
    mraw_file = None
    cihx_file = None
    
    for root, dirs, files in os.walk(experiment_folder):
        for file in files:
            if file.endswith('.mraw'):
                mraw_file = os.path.join(root, file)
            elif file.endswith('.cihx'):
                cihx_file = os.path.join(root, file)
    
    return mraw_file, cihx_file

def read_cihx_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            first_last_line = [i for i in range(len(lines)) if '<cih>' in lines[i] or '</cih>' in lines[i]]
            xml = ''.join(lines[first_last_line[0]:first_last_line[-1]+1])

        cihx_info = xmltodict.parse(xml)
        return cihx_info

    except Exception as e:
        print(f"Error reading .cihx file {file_path}: {e}")
        return None

"""def read_mraw_file(mraw_file):
    with open(mraw_file, 'rb') as f:
        # Read the binary data
        raw_data = np.fromfile(f, dtype=np.uint16)
        
    return raw_data"""

def read_mraw_file(file_path, frame_shape, dtype=np.uint16):
    # Calculate the size of each frame in bytes
    frame_size = np.prod(frame_shape) * dtype.itemsize

    # Calculate the number of frames in the file
    file_size = np.fromfile(file_path, dtype=np.uint32, count=1)[0]
    num_frames = file_size // frame_size

    shape = (num_frames,) + frame_shape
    print(shape)
    # Create a memory-mapped array to access the file contents
    memmap_array = np.memmap(file_path, dtype=dtype, mode='r', shape= shape)

    return memmap_array

def process_experiment_data(experiment_number):
    # Get the directory of the current script being executed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the parent folder
    parent_dir = os.path.dirname(script_dir)
    
    # Path to the "data" subfolder
    data_dir = os.path.join(parent_dir, 'data')
    
    # Find the folder for the specified experiment number
    experiment_folder = os.path.join(data_dir, f"{experiment_number:03d}")
    
    # Check if the experiment folder exists
    if not os.path.exists(experiment_folder):
        print(f"Experiment folder {experiment_number:03d} does not exist.")
        return None, None, None
    
    # Find .mraw and .cihx files
    mraw_file, cihx_file = find_files_in_experiment_folder(experiment_folder)
    
    # Read .cihx file
    cihx_info = read_cihx_file(cihx_file)
    
    # Read .mraw file
    mraw_data = read_mraw_file(mraw_file, (1024, 1024))
    
    return cihx_info, mraw_data, mraw_file

# Example usage:
experiment_number = 1  # Replace with the specific experiment number

chix_info, mraw_data, mraw_file = process_experiment_data(experiment_number)

print(mraw_data.shape)