import bpy
import numpy as np

def draw_channel(mask_path):
    arr = np.load(mask_path)
    s = arr.shape
    for x in range(s[0]):
        for y in range(s[1]):
            if arr[x, y] > 0.5:
                bpy.ops.mesh.primitive_cube_add(size=0.01, location=(x*0.01, y*0.01, 0))

if __name__ == "__main__":
    import sys
    mask_path = sys.argv[-1]  # pass path via command line
    draw_channel(mask_path)
