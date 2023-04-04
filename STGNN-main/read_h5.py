import h5py
import numpy as np

# convert the .h5 file into easier operatable .npy file
f = h5py.File('pems-bay.h5', 'r')
print(f.keys())
print([key for key in f.keys()])
print(f['speed'].keys())
print(f['speed']['block0_values'][:].shape)
np.save('traffic_Data', f['speed']['block0_values'][:])
