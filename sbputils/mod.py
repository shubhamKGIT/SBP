import numpy as np
import os
import pandas as pd

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

data = pd.DataFrame(data = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 200, 40, 80], columns = ["vals"])
print(data.head(10))
data1 = data.rolling(4).mean()
print(data1)