import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union

def show_image(image_data: Optional[Union[np.array, np.ndarray]] = None, cmap: str = 'hot'):
    if image_data is None:
        image_data = np.array([(1,2,3,4,5),(4,5,6,7,8),(7,8,9,10,11)])
    else:
        pass
    im = plt.imshow(image_data, cmap=cmap, interpolation='none')
    plt.colorbar(im)
    plt.show()

def frame_sync_vid_seq(info: dict, spectral_frame_num: int):
        video_channel = info["video_channel"]
        spectral_channel = info["spectral_channel"]
        if spectral_frame_num in [i+1 for i in range(spectral_channel["num_of_frames"])]:
            pass
        else:
            raise Exception("spectral frame number passed needs to be checked")
        fps = video_channel["hs_vid_fps"]
        vid_frames_per_ms = video_channel["hs_vid_fps"]/1000.
        vid_frames_per_spectra = spectral_channel["gap_between_spectral_frames"]*vid_frames_per_ms
        integration_time = spectral_channel["integration_time"]
        no_of_vid_frames = integration_time*vid_frames_per_ms
        vid_saved_from_frame = video_channel["video_saved_from_frame"]
        rel_vid_acquisition_frame = (spectral_channel["rel_aquisition_starts_at"]/1000.)*fps
        integration_starts_at_vid_frame = rel_vid_acquisition_frame + (spectral_frame_num - 1)*vid_frames_per_spectra - vid_saved_from_frame
        if integration_starts_at_vid_frame < 0:
             raise Exception("video data not saved for this spectra")
        else:
             pass
        return integration_starts_at_vid_frame, no_of_vid_frames
