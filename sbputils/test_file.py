import json
import numpy as np


def test_json():
    dict = {
        "talk": "lalala",
        "walk": [100, 300, 1]
    }

    with open("myjson.json") as f:
        f.dumps(dict)
        f.close()

a =  np.arange(9.0).reshape((3,3))
c = np.add(a, 1.0)
print(np.reciprocal(c))