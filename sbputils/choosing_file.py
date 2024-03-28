import os

def find_mraw_file(experiment_number):
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
        return None
    
    # Iterate through the experiment folder and its subfolders
    for root, dirs, files in os.walk(experiment_folder):
        # Search for .mraw files
        for file in files:
            if file.endswith('.mraw'):
                return os.path.join(root, file)

# Example usage:
experiment_number = 1  # Replace with the specific experiment number

mraw_file_path = find_mraw_file(experiment_number)
if mraw_file_path:
    print("Path of .mraw file:", mraw_file_path)
else:
    print("No .mraw file found for the specified experiment number.")