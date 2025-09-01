from fileHandler import Files, FilesTester
from utils import get_filename_with_ext

def test_files_obj(exp: int = 0):
    "test files class and its methods"
    EXP_No = exp
    #getting the class instance
    exp = Files(EXP_No)
    exp_files = exp.expFiles    # called without basepath and filenames
    print(f"{exp_files}")
    #also returning filenames
    return exp_files

def test_csv_read(exp: int = 0, cols: list[str] = ["Frame", "Wavelength", "Intensity"]):
    "basic test for testing csv read"
    EXP_No = exp    # should load video_file.mp4 and spectra.csv from folder 000
    #getting the class instance
    exp = Files(EXP_No)
    exp_files = exp.expFiles
    print(f"files found: {exp_files} \n")
    csv_file = get_filename_with_ext(exp_files, ".csv")
    #reading the csv data
    spectral_data = exp.read_csv(filepath=csv_file, cols = cols)
    print(f"csv data read using Files.read_csv(), returned as pd.Dataframe, printing below sone lines:\n")
    print(spectral_data.head())
    return spectral_data

def test_video_read(exp: int = 0, extension: str = None):
    "play the .mp4 file"
    EXP_No = exp   
    exp = Files(EXP_No)
    exp_files = exp.expFiles
    print(f"files found: {exp_files} \n")
    if extension == ".mp4":
        video_file = get_filename_with_ext(exp_files, ".mp4")
        exp.check_video(video_file)
    else:
        video_file = get_filename_with_ext(exp_files, ".cihx")
        print(f"reading video file named: {video_file}")
        pass

if __name__=="__main__":
    file_handler = Files(exp_number=12, basepath=None)
    print(file_handler)
    print(f"args avaiable for tester: {file_handler.expFiles}")
    fileTester = FilesTester()
    fileTester.check_csv(get_filename_with_ext(file_handler.expFiles, ext=".csv"), cols = ["Frame", "Intensity"])
    fileTester.check_video(get_filename_with_ext(file_handler.expFiles, ext=".cihx"), mp4=False)
    