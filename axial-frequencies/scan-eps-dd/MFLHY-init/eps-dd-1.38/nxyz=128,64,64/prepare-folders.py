import os

os.system(f"cp -r small-box large-box")
# in the run.py, change the value using sed: "box_size=np.array([800, 500, 500]) * 2 * 1.5" -> "box_size=np.array([800, 500, 500]) * 2 * 1.5 * 1.5"
os.system(f"sed -i 's/800, 500, 500/1200, 750, 750/g' large-box/run.py")