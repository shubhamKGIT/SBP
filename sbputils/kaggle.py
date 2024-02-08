import kaggle
import pandas as pd


TRAIN_CSV_DIR = '/kaggle/input/hubmap-organ-segmentation/train.csv'
data = pd.read_csv(TRAIN_CSV_DIR)
print(data)