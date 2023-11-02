
import os
# I want to make folders with eps-dd 1.38, 1.39, up to 1.45.



# 1.39
os.system(f"cp -r eps-dd-1.38 eps-dd-1.39")
# in eps-dd-1.39/nxyz=128,64,64/small-box/run.py, change using sed: "eps_dd=1.38" -> "eps_dd=1.39"
os.system(f"sed -i 's/eps_dd=1.38/eps_dd=1.39/g' eps-dd-1.39/nxyz=128,64,64/small-box/run.py")

# 1.40
os.system(f"cp -r eps-dd-1.38 eps-dd-1.40")
# in eps-dd-1.40/nxyz=128,64,64/small-box/run.py, change using sed: "eps_dd=1.38" -> "eps_dd=1.40"
os.system(f"sed -i 's/eps_dd=1.38/eps_dd=1.40/g' eps-dd-1.40/nxyz=128,64,64/small-box/run.py")

# 1.41
os.system(f"cp -r eps-dd-1.38 eps-dd-1.41")
# in eps-dd-1.41/nxyz=128,64,64/small-box/run.py, change using sed: "eps_dd=1.38" -> "eps_dd=1.41"
os.system(f"sed -i 's/eps_dd=1.38/eps_dd=1.41/g' eps-dd-1.41/nxyz=128,64,64/small-box/run.py")

# 1.42
os.system(f"cp -r eps-dd-1.38 eps-dd-1.42")
# in eps-dd-1.42/nxyz=128,64,64/small-box/run.py, change using sed: "eps_dd=1.38" -> "eps_dd=1.42"
os.system(f"sed -i 's/eps_dd=1.38/eps_dd=1.42/g' eps-dd-1.42/nxyz=128,64,64/small-box/run.py")

# 1.43
os.system(f"cp -r eps-dd-1.38 eps-dd-1.43")
# in eps-dd-1.43/nxyz=128,64,64/small-box/run.py, change using sed: "eps_dd=1.38" -> "eps_dd=1.43"
os.system(f"sed -i 's/eps_dd=1.38/eps_dd=1.43/g' eps-dd-1.43/nxyz=128,64,64/small-box/run.py")

# 1.44
os.system(f"cp -r eps-dd-1.38 eps-dd-1.44")
# in eps-dd-1.44/nxyz=128,64,64/small-box/run.py, change using sed: "eps_dd=1.38" -> "eps_dd=1.44"
os.system(f"sed -i 's/eps_dd=1.38/eps_dd=1.44/g' eps-dd-1.44/nxyz=128,64,64/small-box/run.py")

# 1.45
os.system(f"cp -r eps-dd-1.38 eps-dd-1.45")
# in eps-dd-1.45/nxyz=128,64,64/small-box/run.py, change using sed: "eps_dd=1.38" -> "eps_dd=1.45"
os.system(f"sed -i 's/eps_dd=1.38/eps_dd=1.45/g' eps-dd-1.45/nxyz=128,64,64/small-box/run.py")

# Enter each folder and run the prepare-folders.py script
eps_dd_values = ["1.38", "1.39", "1.40", "1.41", "1.42", "1.43", "1.44", "1.45"]
for eps_dd_value in eps_dd_values:
    os.chdir(f"eps-dd-{eps_dd_value}")
    os.system(f"python3 prepare-folders.py")
    os.chdir("..")