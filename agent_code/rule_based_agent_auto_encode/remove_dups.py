import numpy as np
# careful this will probably fill your RAM if you have less than 16GB

with open("states.npy", "rb") as f:
    data = np.load(f, allow_pickle=False)

data_unique = np.unique(data, axis=0)
with open("states_no_dup.npy", "wb") as f:
    np.save(f, data_unique, allow_pickle=False)
