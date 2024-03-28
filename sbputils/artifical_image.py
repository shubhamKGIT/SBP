import numpy as np
import matplotlib.pyplot as plt

frames = np.random.randint(1, 2e10-1, (100, 1024, 1024))
plt.imshow(frames[10, :, :])
plt.show()