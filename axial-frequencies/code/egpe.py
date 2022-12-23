# Import dependencies

import numpy as np
#from numpy import linalg
import math, sys, os, time


print("SIM. started at ", time.ctime())
start_time = time.time()

scan_indx = 0
eps_dd_arr = np.linspace(1.3, 1.52, num = 20)
eps_dd = 1.35 # eps_dd_arr[scan_indx] #variable


r_0 = 387.672168  # 387.672168 
a_dd = 1./3 
sclen = a_dd / eps_dd * r_0
alpha = 2*np.pi*(sclen/r_0)  # 
g = 4*np.pi*sclen




mu_m = 1.E-06 / (r_0 * 0.529E-10)


fx = 18.5  # Hz
fy = 53
fz = 81	 # Hz
fun_unit = 7.89889045E-06 
w_ho = np.array([fx, fy, fz])
a_ho = fun_unit / np.sqrt(w_ho) / 1.E-06 * mu_m

# sqrt(hbar / ((164 atomic mass unit) * 2 * pi * Hertz)) = 7.89889045×10^-6 meters
# taken from: https://www.wolframalpha.com/input/?i=sqrt%28hbar+%2F+%28%28162+atomic+mass+unit%29+*+2+*+pi+*+Hertz%29%29


nparticles = 35000 # 3.3E+04 #this will only be used when you do imaginary time propagation

#simulation and physical parameters

nxyz = np.array([512, 64, 32] )   #if you want 1D or 2D, just change this array
nxyz = np.array(nxyz, dtype=np.int32)
#L_ok = np.array([9000., 4500., 4500.]) #you add on top of L an interface which is an absorbing area. L_ok is area without the interface
L = np.array([0.7*a_ho[0], 0.5*a_ho[1], 0.7*a_ho[2]]) * 50 # np.array([70, 10, 10]) * mu_m #if you want 1D or 2D, just change this array. This means between -L/2 and L/2
#L = np.array([0.8*a_ho[0], 0.25*a_ho[1], 1.25*a_ho[2]]) * 50 # np.array([70, 10, 10]) * mu_m #if you want 1D or 2D, just change this array. This means between -L/2 and L/2


L_ok = L * 0.8 # np.array([10000., 6000., 6000.])
EN_TOL = 1.E-10 # if the energy in the new iteration is within the EN_TOL tolerance, exit the simulation
t_equil = 100E+04
t_prod  = 100000E+04 #np.inf for infinite      #total propagation time. Count from the real (imaginary) timestep if you propagate in real (imaginary) time
deltat_prod    = 10 # real timestep

deltat_equil = 10 # this is absolute value of imaginary part


printToStdoutEvery = 100 # (int)(t_equil/deltat_equil / 100)
printDenEvery      = 100000 # (int)(t_equil/deltat_equil / 100)




kill_fac = 0.95



def init_kill_matrix(x, y, z):
	matrix = 1.
	if(N_DIM == 1):
		matrix = np.logical_or(np.abs(x) > L_ok[0]/2.)
	if(N_DIM == 2):
		matrix = np.logical_or(np.abs(x) > L_ok[0]/2., np.abs(y) > L_ok[1]/2.)
	if(N_DIM == 3):
		matrix = np.logical_or(np.abs(x) > L_ok[0]/2., np.abs(y) > L_ok[1]/2.)
		matrix = np.logical_or(matrix,   np.abs(z) > L_ok[2]/2.)
	return np.invert(matrix) + matrix*kill_fac # this matrix is 1 inside box, and kill_fac outside


ft_dip = 1.
def get_phi_dd(den):
	return np.fft.ifftn(np.fft.fftn(den) * ft_dip)

def potential_external(x, y, z):
	return 0.5 * (x**2/a_ho[0]**4 + y**2/a_ho[1]**4 + z**2/a_ho[2]**4)
	
def energy_density_interaction(den, phi_dd):
	return  alpha*den**2 + beta*den**(gamma+1.) + 0.5 * phi_dd * den
	
def dEps_dPsi(den, phi_dd):
	return 2.*alpha*den + beta*(gamma + 1.)*den**gamma + phi_dd # - 1.j * kappa2 * den**2

def init_psi(x, y, z):
	xr = x/mu_m
	return np.random.uniform(0.95, 1, np.shape(x)) *  np.exp(-0.5 * ((xr/a_ho[0])**2 + (y/a_ho[1])**2 + (z/a_ho[2])**2) ) + 0.j # * (0.10 + xr**2*np.cos(3*xr)**4) + 0.j

	psi = np.random.uniform(0, 1, np.shape(x)) * np.exp(-0.5 * ((x/a_ho[0])**2 + (y/a_ho[1])**2 + (z/a_ho[2])**2) ) + 0.j
	return psi
	
	
	n_droplets = 9
	psi =  np.exp(-0.5*(x**2/a_ho[0]**2 + y**2/a_ho[1]**2))  + 0.j #Gaussian
	for i in range(n_droplets - 1):
		drop_x = np.random.uniform(-8, 8)
		drop_y = np.random.uniform(-2, 2)
		psi += np.exp(-0.5*((x-drop_x*a_ho[0])**2 / a_ho[0]**2 + (y-drop_y*a_ho[1])**2 / a_ho[1]**2)) 
	return psi * np.exp(-0.5 * z**2/a_ho[2]**2)
		



################################################################################
################################################################################
################################################################################
################################################################################


#exit()

imProp = False
#setting up auxiliary variables
N_DIM = len(nxyz)
LHalf = L / 2
x = np.linspace(-LHalf[0], LHalf[0], num = nxyz[0], endpoint=False)
y = np.linspace(-LHalf[1], LHalf[1], num = nxyz[1], endpoint=False)
z = np.linspace(-LHalf[2], LHalf[2], num = nxyz[2], endpoint=False)


dx = L/nxyz
d3r = np.prod(dx)

kx = np.fft.fftfreq(nxyz[0], dx[0]/( 2. * np.pi))
ky = np.fft.fftfreq(nxyz[1], dx[1]/( 2. * np.pi))
kz = np.fft.fftfreq(nxyz[2], dx[2]/( 2. * np.pi))

d3k = (kx[1] - kx[0]) * (ky[1] - ky[0]) * (kz[1] - kz[0])
x, y, z = np.meshgrid(x, y, z, indexing='ij')
kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

kill_matrix = init_kill_matrix(x, y, z)

kvec = np.sqrt(kx**2 + ky**2 + kz**2)
ft_dip = 4*np.pi/3  * (2*kz**2 - kx**2 - ky**2) / kvec**2
ft_dip  = np.nan_to_num(ft_dip, posinf=0)



lmbda_array = 0.001 # 10**np.linspace(-6, -1, num = 50)
lmbda = 0.00002 / a_ho[0]**4 # lmbda_array[scan_indx] / a_ho[0]**4
pot_ext = potential_external(x, y, z) + lmbda * x**2

dt_equil = -1.j*deltat_equil
dt_prod = deltat_prod



print(" *** Parameters of the simulation ***")
print("nxyz ", nxyz)
print("L ", L)
print("d3r " , d3r)
print("d3k", d3k )






#Trotter operator, O(dt^2) global error

sumk2 = (kx**2 + ky**2 + kz**2)/2
kinprop = 1.
def T2_operator(psi, den, phi_dd):
	global dt, pot_ext, imProp, kinprop
	#exp(-1/2 * i dt * V)
	pot = pot_ext + dEps_dPsi(den, phi_dd)
	psi *= np.exp(-0.5j * pot * dt)
	
	psi = np.fft.fftn(psi)		
	psi *= kinprop
	psi = np.fft.ifftn(psi)

	den = np.absolute(psi) ** 2
	phi_dd = get_phi_dd(den)
	pot = pot_ext + dEps_dPsi(den, phi_dd)
	
	psi *= np.exp(-0.5j * pot * dt)
	if(imProp): #normalize		
		psi *= np.sqrt(nparticles)/np.sqrt(d3r * np.sum(np.absolute(psi) ** 2))
	return psi
	


def energy(psi, den, phi_dd):
	potext =  den * pot_ext
	pot_tot = energy_density_interaction(den, phi_dd)
	kin_tot = (np.conj(psi) * np.fft.ifftn(np.fft.fftn(psi) * sumk2)).real
	return np.array([d3r*np.sum(kin_tot), d3r * np.sum(potext) , d3r*np.sum(pot_tot).real])

output_dir = 'snapshots_time_evolution_0'; num=0
while os.path.exists(output_dir): num+=1; output_dir="snapshots_time_evolution_"+str(num)
if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(output_dir+'/npy_files'): os.makedirs(output_dir+'/npy_files')
file_en_equil = open(output_dir + '/en_equil.dat', 'w', buffering=1)
file_en_prod = open(output_dir + '/en_prod.dat', 'w', buffering=1)
delta_t_crit=0.


x_lin = np.linspace(-LHalf[0], LHalf[0], num = nxyz[0], endpoint=False)
y_lin = np.linspace(-LHalf[1], LHalf[1], num = nxyz[1], endpoint=False)
z_lin = np.linspace(-LHalf[2], LHalf[2], num = nxyz[2], endpoint=False)
np.save(f"{output_dir}/npy_files/x_lin", x_lin)
np.save(f"{output_dir}/npy_files/y_lin", y_lin)
np.save(f"{output_dir}/npy_files/z_lin", z_lin)

def dft_simulation(t_max,delta_t):	
	global psi, delta_t_crit, kinprop
	timestep = 0
	time = 0.
	en_old = np.inf; en = -np.inf
	while time <= t_max:
		den = np.absolute(psi) ** 2
		phi_dd = get_phi_dd(den)
		if(timestep % printToStdoutEvery == 0 or timestep==0):
			if(imProp == False):
				ncalc = d3r * np.sum(den)
			else:
				ncalc = nparticles
			kin, potext, potint = energy(psi, den, phi_dd)
			en = kin + potext + potint
			mx2 = np.sqrt(np.sum(x**2 * den) / np.sum(den) / L[0]**2)
			my2 = np.sqrt(np.sum(y**2 * den) / np.sum(den) / L[1]**2)
			mz2 = np.sqrt(np.sum(z**2 * den) / np.sum(den) / L[2]**2)
			string_out= f"{time:.3e} {ncalc:.5e} {kin:.5e} {potext:.5e} {potint:.5e} {en:.5e} {mx2:.5e} {my2:.5e} {mz2:.5e}\n"
			if(imProp):
				file_en_equil.write(string_out)
			else:
				file_en_prod.write(string_out)
			print(string_out)

		if(timestep % printDenEvery == 0 or timestep==0):
			timestep /= printDenEvery
			if(imProp==True):
				file=output_dir + '/npy_files/psi_equil_%i' % timestep
			else:
				file=output_dir + '/npy_files/psi_prod_%i' % timestep
			#if(imProp==False):
			#np.save(file, psi)   # use exponential notation
			#np.save(file + "_xy", dx[2] * np.sum(den, axis=2) )  # use exponential notation
			#np.save(file + "_xz", dx[1] * np.sum(den, axis=1) )  # use exponential notation
			#np.save(file + "_yz", dx[0] * np.sum(den, axis=0) )  # use exponential notation
			#np.save(file + "_x", np.swapaxes(den,0,2)[int(nxyz[2]/2)][int(nxyz[1]/2)])   # use exponential notation
			#np.save(file + "_y", np.swapaxes(den,1,2)[int(nxyz[0]/2)][int(nxyz[2]/2)])   # use exponential notation
			#np.save(file + "_z", den[int(nxyz[0]/2)][int(nxyz[1]/2)])   # use exponential notation
			#np.save(file + "_yz", np.sum(den, axis=0))   # use exponential notation			
			timestep *= printDenEvery
		psi = T2_operator(psi, den, phi_dd)
		#if(imProp == False):
		#	psi *= kill_matrix 
		
		time += delta_t
		timestep += 1
		if np.abs((en - en_old) / en) < EN_TOL:
			break
		
	return 0

psi = init_psi(x, y, z)	
psi *= np.sqrt(nparticles)/np.sqrt(d3r * np.sum(np.absolute(psi) ** 2))  #normalize

#den = np.absolute(psi) ** 2
#phi_dd = get_phi_dd(den)
#print(energy(psi, den, phi_dd) / nparticles, np.sum(1/a_ho**2)*0.5 / 2)
#exit()

imProp = True
dt = dt_equil

kinprop = np.exp(-1j * dt * sumk2)
print(" *** EQUILIBRATION ... **** ")

print("EQUILIBRATION ENDED at ", time.ctime())


dft_simulation(t_equil, deltat_equil)




print( f"Equilibration took {time.time() - start_time} seconds")

pot_ext = potential_external(x, y, z) 


imProp = False
dt = dt_prod
kinprop = np.exp(-1j * dt * sumk2)

print(" *** PRODUCTION ... **** ")
print(" *** delta_t_crit = %.5e" % delta_t_crit)
dft_simulation(t_prod, deltat_prod)

#np.save("psi", cp.asnumpy(psi))

print("SIM. ENDED at ", time.ctime())
print( time.time() - start_time)

	
