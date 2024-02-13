import pathlib
import os
import glob
from load_data import Files
import matplotlib.pyplot as plt
from PIL import Image
import rawpy
import cv2
import rasterio
from rasterio.plot import show
import tifffile



"""user to create image sequences from industry-standard file formats, 
    including AVI (Windows video), MOV (Apple QuickTime video), 
    TIFF ( 8-bit or 26-bit file formats), JPEG, BMP (Bitmap), PNG (Portable Network Graphics), 
    RAW and RAWW (native camera file formats)
"""

def get_data_path() -> pathlib.Path:
    dir = pathlib.Path(__file__).resolve().parent
    parent_dir, _ = os.path.split(dir)
    data_dir = pathlib.Path(os.path.join(parent_dir, "data"))
    return data_dir

def read_directory(data_dir: pathlib.Path) -> None:
    for dir, dirs, files in os.walk(data_dir):
        print(f"files in {data_dir} are {files}")

def find_files(data_dir: pathlib.Path, file_extension: str = ".raww") -> list[pathlib.Path]:
    files_with_extension = []
    for file in data_dir.iterdir():
        print(type(file))
        if file.suffix == file_extension:
            files_with_extension.append(file)
    print(f"file with extension {file_extension} are {files_with_extension}")
    return files_with_extension

print(f"data file path is : {get_data_path()}")
read_directory(get_data_path())

def raw():
    """
    .RAW (byte) : 8bit unsigned
    .RAWW (word) : 16bit unsigned
    *.RAF (float) : 32bit real
    """
    "reading raw file - a sample file of .raw format using rawpy"
    raww_files = find_files(data_dir=get_data_path(), file_extension=".raww")
    for file in raww_files:
        with open(file, 'rb') as f:
            #data = f.read()
            img = rawpy.imread(f)    #using rawpy
            data = img.raw_image
            print(f"data from raw file: \n {data}")
        plt.imshow(data)
        plt.show()
        f.close()
    return data

def tiff():
    tiff_files = find_files(data_dir=get_data_path(), file_extension=".tif")
    for file in tiff_files:
        img = Image.open(file)   #using PIL.Image 
        #img1 = cv2.imread(file)    #using cv2
        plt.imshow(img)
        plt.show()
        with rasterio.open(file) as image:    #using rasterio
            image_array = image.read()
            #print(f"tiff image array : {image_array.shape}")
            show(image_array)

def mp4():
    mp4_files = find_files(data_dir=get_data_path(), file_extension=".mp4")
    for file in mp4_files:
        #mp4_file = Image.open(file)
        print(file)
    return mp4_files

def read_many_images():
    basepath = get_data_path()
    path = glob.glob(f"{basepath}+*.jpg")
    cv_img = []
    for img in path:
        n = cv2.imread(img)
        cv_img.append(n)

def mp4_to_jpeg(filepath):
    cap = cv2.VideoCapture(filepath)
    fps = cv2.get(cv2.CAP_PROP_FPS)
    print(f"fps: {fps}\n")
    i=0
    while(cap.isOpened()):
        avail, frame = cap.read()
        if avail:
            cv2.imwrite(os.path.join(get_data_path(), "mp4_extracts") + "frame%d.jpg" % i, frame)
            i=i+1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


data = raw()
print(data.shape)
#print(f"reading multiple files and saving in list")
#read_many_images()
#sample_mp4 = mp4()
#mp4_to_jpeg(sample_mp4)
