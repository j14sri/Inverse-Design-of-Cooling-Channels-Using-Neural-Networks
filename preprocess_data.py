# cnn/preprocess_data.py
import os
import numpy as np
from glob import glob

def load_simulation_files(data_root="data"):
    file_paths = glob(os.path.join(data_root, "t_load_*/temp_map.npz"))
    print(f"Found {len(file_paths)} simulations.")
    return file_paths

def load_data(file_path):
    npz = np.load(file_path)
    data = npz['data']  # shape: (num_vertices, 3) → [x, y, temperature]
    t_load = npz['thermal_load'].item()  # scalar
    return data, t_load
