import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union

def show_image(image_data: Optional[Union[np.array, np.ndarray]]):
    if image_data is None:
        image_data = np.array([(1,2,3,4,5),(4,5,6,7,8),(7,8,9,10,11)])
    else:
        pass
    im = plt.imshow(image_data, cmap='hot', interpolation='none')
    plt.colorbar(im)
    plt.show()