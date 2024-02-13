import numpy as np
import os

a = np.arange(1, 4)
b = np.arange(500, 800, 100)
c = np.log(b)

print(a)
print(b)

d = np.multiply(b, c)
e = np.divide(d, a)
print(e)

f = np.exp(e/1000.)
print(f)
print(os.listdir())