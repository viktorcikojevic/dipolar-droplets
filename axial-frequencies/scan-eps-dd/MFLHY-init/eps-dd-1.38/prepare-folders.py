import os

# I want to make folders with nxyz=128,64,64, 256,64,64, 512,64,32, 128,32,32.

# 256,64,64
os.system(f"cp -r nxyz=128,64,64 nxyz=256,64,64")
# in nxyz=256,64,64/small-box/run.py, change using sed: "128, 64, 64" -> "256, 64, 64"
os.system(f"sed -i 's/128, 64, 64/256, 64, 64/g' nxyz=256,64,64/small-box/run.py")

# 512,64,32
os.system(f"cp -r nxyz=128,64,64 nxyz=512,64,32")
# in nxyz=512,64,32/small-box/run.py, change using sed: "128, 64, 64" -> "512, 64, 32"
os.system(f"sed -i 's/128, 64, 64/512, 64, 32/g' nxyz=512,64,32/small-box/run.py")

# 128,32,32
os.system(f"cp -r nxyz=128,64,64 nxyz=128,32,32")
# in nxyz=128,32,32/small-box/run.py, change using sed: "128, 64, 64" -> "128, 32, 32"
os.system(f"sed -i 's/128, 64, 64/128, 32, 32/g' nxyz=128,32,32/small-box/run.py")



# list all directories in a variable
dirs = os.listdir()
print(dirs)
for dir in dirs:
    # if dir is not a directory, skip it
    if not os.path.isdir(dir):
        continue
    
    os.chdir(f"{dir}")
    os.system(f"python3 prepare-folders.py")
    os.chdir("..")