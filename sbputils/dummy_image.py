import numpy as np
import matplotlib.pyplot as plt

def build_dummy_frames(n_frames = 100, n_pixel = 1024):
    """dispaly a set of frames, show one and return frame data as numpy array

    PARAMETERS
    ----------
        n_frames: int
            number of frames to build
        n_pixel: int
            number of pixels in each side
    RETURNS
    -------
        frames: np.ndarray, dtype = np.uint8
            video array holding n_frames of random frame @ (n_pixel, n_pixel) each i.e., say (100, 1024, 1024) shape
    """
    frames = np.random.randint(0, 256, (n_frames, n_pixel, n_pixel), dtype=np.uint16)
    plt.imshow(frames[10, :, :])
    plt.colorbar()
    plt.show()
    return frames

if __name__=="__main__":
    frames = build_dummy_frames()
    print(f"frames min, max: {frames.min()}, {frames.max()}")