import numpy as np

class StereoSystem:
    def __init__(self, npz_path):
        S = np.load(npz_path,allow_pickle=True)