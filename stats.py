import numpy as np
import time
from tqdm import tqdm
from imos_cython import imos_cython

shape = np.array([200, 30, 70], dtype=np.int16)
n_index, d_index, _ = np.where(np.ones(shape=np.array([200, 30, 70]), dtype=np.int16))

st = np.random.randint(0, shape[2] // 2, size=shape, dtype=np.int16)
ed = st + np.random.randint(0, shape[2] // 2, size=shape, dtype=np.int16)

time_st = time.time()
for i in tqdm(range(1000)):
    imos_cython(shape, st, ed)
print("imos_cython:", (time.time() - time_st), "s")
