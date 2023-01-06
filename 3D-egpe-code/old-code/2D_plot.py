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

r_0 = 387.672168
sclen = 88. / r_0 # units of a0
fx = 33  # Hz
fz = 167 # Hz
alpha = 0.39 #variable
fy = fx / alpha
fun_unit = 7.85057874E-6 / (r_0*0.529E-10)
#print(fun_unit)
# sqrt(hbar / ((164 atomic mass unit) * 2 * pi * Hertz)) = 7.85057874×10^-6 meters
# taken from: https://www.wolframalpha.com/input/?i=sqrt%28hbar+%2F+%28%28164+atomic+mass+unit%29+*+2+*+pi+*+Hertz%29%29
w_ho = 2 * np.pi * np.array([fx, fy, fz])
a_ho = fun_unit / np.sqrt(w_ho)
#print(f"a_ho = {a_ho}")

alpha = 2 * np.pi * sclen
g = 4*np.pi*sclen
a_dd = 1./3
eps_dd = a_dd / sclen
beta  = 32*g*sclen**1.5 /(3 * np.sqrt(np.pi)) * (1+1.5*eps_dd**2) * 2./5
gamma = 3./2



main_dir = "snapshots_time_evolution_0"
t=np.array([1, 1, 1])
t[0] = np.shape(np.load(f"{main_dir}/npy_files/psi_equil_0_x.npy"))[0]
t[1] = np.shape(np.load(f"{main_dir}/npy_files/psi_equil_0_y.npy"))[0]
t[2] = np.shape(np.load(f"{main_dir}/npy_files/psi_equil_0_z.npy"))[0]
nxyz = np.array(t, dtype=np.int32)    #if you want 1D or 2D, just change this array
#L_ok = np.array([9000., 4500., 4500.]) #you add on top of L an interface which is an absorbing area. L_ok is area without the interface
L = 40. * np.array([2*a_ho[0], a_ho[1], a_ho[2]]) #if you want 1D or 2D, just change this array. This means between -L/2 and L/2


LHalf = L/2.

x = np.linspace(-LHalf[0], LHalf[0], nxyz[0], endpoint=False)
y = np.linspace(-LHalf[1], LHalf[1], nxyz[1], endpoint=False)
z = np.linspace(-LHalf[2], LHalf[2], nxyz[2], endpoint=False)
#x, y = np.meshgrid(x, y, indexing='ij')




timestep = 0
zmaxc = -1.;

minl = np.min(L)
params = {'figure.figsize': (16, 9),
 'xtick.labelsize': 22,
         'ytick.labelsize': 22,
         'axes.titlesize': 22, }
pylab.rcParams.update(params)
pylab.xlim(-LHalf[0], LHalf[0])
pylab.ylim(-LHalf[1], LHalf[1])

timestep = 0
while timestep < 100:
	#print(timestep)
	z1 = np.load(f"{main_dir}/npy_files/psi_equil_{timestep}_xy.npy")
	z = z1
	print(timestep, np.shape(x), np.shape(y), np.shape(z))
	f = interpolate.interp2d(x, y, np.swapaxes(z, 0, 1), kind='cubic')
	xx = np.linspace(-LHalf[0], LHalf[0], 256, endpoint=False)
	yy = np.linspace(-LHalf[1], LHalf[1], 256, endpoint=False)
	
	z = f(xx, yy)

	zmaxc = np.max(z) *0.7

	z_min, z_max = 0, zmaxc #0., 0.002
	fig, ax = plt.subplots()
	c = ax.pcolormesh(xx, yy, z, cmap='magma' ,vmin=z_min, vmax=z_max)
	#ax.set_title("t = %.2f ms" % ((timestep+1) * 1000 * 200. * 9.23649E-06))
	#ax.set_title(f"t = {timestep*0.1:.2f}"+r'$\tau$')	# set the limits of the plot to the limits of the data
	ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])
	#fig.colorbar(c, ax=ax)
	pylab.xlabel("x", fontsize=20)
	pylab.ylabel("y", fontsize=20)
	#pylab.axis('equal')
	
	
	pylab.savefig(f"2D_snapshots/2D_snapshot_{timestep}_xy.png",  bbox_inches='tight')
	#plt.show()
	pylab.clf()		
	timestep += 1
	plt.close()
	'''
	if(timestep >= 50):
		break
	'''

timestep = 0
main_dir = "snapshots_time_evolution_1"

while timestep < 100:
	#print(timestep)
	z1 = np.load(f"{main_dir}/npy_files/psi_equil_{timestep}_xy.npy")
	z = z1
	print(timestep, np.shape(x), np.shape(y), np.shape(z))
	f = interpolate.interp2d(x, y, np.swapaxes(z, 0, 1), kind='cubic')
	xx = np.linspace(-LHalf[0], LHalf[0], 256, endpoint=False)
	yy = np.linspace(-LHalf[1], LHalf[1], 256, endpoint=False)
	
	z = f(xx, yy)

	zmaxc = np.max(z) *0.7

	z_min, z_max = 0, zmaxc #0., 0.002
	fig, ax = plt.subplots()
	c = ax.pcolormesh(xx, yy, z, cmap='magma' ,vmin=z_min, vmax=z_max)
	#ax.set_title("t = %.2f ms" % ((timestep+1) * 1000 * 200. * 9.23649E-06))
	#ax.set_title(f"t = {timestep*0.1:.2f}"+r'$\tau$')	# set the limits of the plot to the limits of the data
	ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])
	#fig.colorbar(c, ax=ax)
	pylab.xlabel("x", fontsize=20)
	pylab.ylabel("y", fontsize=20)
	#pylab.axis('equal')
	
	
	pylab.savefig(f"2D_snapshots/2D_snapshot_{timestep+100}_xy.png",  bbox_inches='tight')
	#plt.show()
	pylab.clf()		
	timestep += 1
	plt.close()
	'''
	if(timestep >= 50):
		break
	'''



