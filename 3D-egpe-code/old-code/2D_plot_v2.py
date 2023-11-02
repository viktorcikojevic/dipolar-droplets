import matplotlib.pyplot as plt
import numpy as np
import pylab 

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

main_dir = "snapshots_time_evolution_0"
nxyz = np.array([256, 128,  128])     #if you want 1D or 2D, just change this array
#L_ok = np.array([9000., 4500., 4500.]) #you add on top of L an interface which is an absorbing area. L_ok is area without the interface
L = 10000. * np.array([1.4, 1., 1.]) #if you want 1D or 2D, just change this array. This means between -L/2 and L/2

LHalf = L/2.

x = np.linspace(-LHalf[0], LHalf[0], nxyz[0], endpoint=False)
y = np.linspace(-LHalf[1], LHalf[1], nxyz[1], endpoint=False)
z = np.linspace(-LHalf[2], LHalf[2], nxyz[2], endpoint=False)
x, y = np.meshgrid(x, y, indexing='ij')

timestep = 0
zmaxc = -1.;

minl = np.min(L)
params = {'figure.figsize': (7.8*L[0]/minl, 7.8*L[1]/minl),
 'xtick.labelsize': 22,
         'ytick.labelsize': 22,
         'axes.titlesize': 22, }
pylab.rcParams.update(params)
pylab.xlim(-LHalf[0], LHalf[0])
pylab.ylim(-LHalf[1], LHalf[1])

while(True):
	#print(timestep)
	z1 = np.load(main_dir+"/npy_files/psi_prod_" + str(timestep) + "_xy.npy")
	z = z1
	if(zmaxc < 0.):
		zmaxc = np.max(z) / 20.
	z_min, z_max = 0, zmaxc #0., 0.002
	fig, ax = plt.subplots()
	c = ax.pcolormesh(x, y, z, cmap='RdBu' ,vmin=z_min, vmax=z_max)
	#ax.set_title("t = %.2f ms" % ((timestep+1) * 1000 * 200. * 9.23649E-06))
	ax.set_title("t = %.2f ms" % ((timestep+1) * 40 * 200. * 6.96414E-06))
	# set the limits of the plot to the limits of the data
	ax.axis([x.min(), x.max(), y.min(), y.max()])
	#fig.colorbar(c, ax=ax)
	pylab.xlabel("x", fontsize=20)
	pylab.ylabel("y", fontsize=20)
	#pylab.axis('equal')
	
	
	
	
	
	
	pylab.savefig("2D_snapshots_v2/2D_snapshot_%i_xy" % timestep,  bbox_inches='tight')
	#plt.show()
	pylab.clf()		
	timestep += 1
	plt.close()
	'''
	if(timestep >= 50):
		break
	'''




