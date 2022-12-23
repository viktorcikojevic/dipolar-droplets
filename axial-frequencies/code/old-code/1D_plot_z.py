import matplotlib.pyplot as plt
import numpy as np
import pylab 
from scipy import interpolate
import os

'''
import matplotlib.pylab as pylab
params = {'legend.fontsize': 12.,
          'figure.figsize': (     7.6*0.8     , 4.8),
         'axes.labelsize': 25,
         'axes.titlesize': 25,
         'xtick.labelsize': 20,
         'ytick.labelsize': 20}
pylab.rcParams.update(params)
'''


r_0 = 130 * 3 # 387.672168 
mu_m = 1.E-06 / (r_0 * 0.529E-10)


main_dir = "snapshots_time_evolution_0"
t=np.array([1, 1, 1])
t[0] = np.shape(np.load(f"{main_dir}/npy_files/psi_equil_0_x.npy"))[0]
t[1] = np.shape(np.load(f"{main_dir}/npy_files/psi_equil_0_y.npy"))[0]
t[2] = np.shape(np.load(f"{main_dir}/npy_files/psi_equil_0_z.npy"))[0]
nxyz = np.array(t, dtype=np.int32)    #if you want 1D or 2D, just change this array

x = np.load(f"{main_dir}/npy_files/x_lin.npy")
y = np.load(f"{main_dir}/npy_files/y_lin.npy")
z = np.load(f"{main_dir}/npy_files/z_lin.npy")
#x, y = np.meshgrid(x, y, indexing='ij')





timestep = 0

output_dir = "1D_snapshots_z"
if not os.path.exists(output_dir): os.makedirs(output_dir)


while timestep < 2000:
	#print(timestep)
	z1 = np.load(f"{main_dir}/npy_files/psi_equil_{timestep}_z.npy")
	

	zmaxc = np.max(z1) # * 0.01

	plt.plot(z, z1, ls='-', marker='o')
	plt.ylim(0, zmaxc)
	#plt.xlim(np.min(x), np.max(x))
	
	pylab.savefig(f"{output_dir}/1D_snapshot_{timestep}_z.png",  bbox_inches='tight')
	#plt.show()
	pylab.clf()		
	timestep += 1
	plt.close()
	'''
	if(timestep >= 50):
		break
	'''




