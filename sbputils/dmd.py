import numpy as np
from numpy.linalg import svd
from videoUtils import read_frames_from_pkl_dump
from files import Files
import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

def flatten_and_make_Vs(orig_frames):
    "assuming here that the original array is 3D, in image sense its a gray video with N x w x h dimentions"
    flattened = orig_frames.reshape((orig_frames.shape[0], -1))
    n_1_size = frames.shape[0]-1
    V_n_1 = flattened[:n_1_size, :]
    V_n = flattened[1:frames.shape[0], :]
    return V_n_1, V_n

if __name__=="__main__":
    EXP_NUM=1
    TEST = True
    exp_holder = Files(EXP_NUM)
    expFiles = exp_holder.files()
    expFolder = exp_holder.expFolder
    dump_file = os.path.join(expFolder, "temperature.pkl")
    if TEST:
        frames = np.random.rand(10, 16, 16)
    else:
        frames = np.array(read_frames_from_pkl_dump(dump_file), dtype=np.uint8)
    orig_array_shape = frames.shape
    print(f"frames shape: {frames.shape}")
    frames = frames.reshape((frames.shape[0], -1))
    print(f"size of flattened frames: {frames.shape}")
    # Step 1
    V_N_minus, V_N= flatten_and_make_Vs(frames)
    print(f"shape of V_n_minus_1: {V_N_minus.shape}, shape of V_n: {V_N.shape}")
    # Step 2
    U, E, Wh = svd(V_N_minus, full_matrices=True)
    W = np.transpose(Wh)
    print(f"shape of U: {U.shape}, shape of E: {E.shape}, shape of W_T: { Wh.shape}")
    print(f"eigenvalues pf V_n_minus: {np.sqrt(np.diag(E))}")
    # Step 3
    inverted_E = np.diag(np.reciprocal(np.diag(E)))
    print(f"inverted E shape: {inverted_E.shape}")
    S = np.transpose(np.multiply(np.transpose(np.matmul(np.matmul(np.transpose(U), V_N), W)), inverted_E))
    print(f"shape of S: {S.shape}")
    Y, L, Zh = svd(S, full_matrices=True)
    print(f"shape of Y: {U.shape}, shape of L: {L.shape}, shape of Zh: {Zh.shape}")
    
    
