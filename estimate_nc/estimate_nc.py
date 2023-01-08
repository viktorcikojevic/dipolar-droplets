
import numpy as np

def fdip(x):
    x  = x + 0.*1j
    EPS = 1.E-10
    res = np.piecewise(x, [np.abs(x-1) < EPS, np.abs(x-1) >= EPS], [0.+0.j, lambda x: (1 + 2.*x**2) / (1 - x**2) - 3*x**2*np.arctanh(np.sqrt(1-x**2)) / (1 - x**2)**1.5])
    return res.real 


def enFit(x, b, c):
    return b*x**c


def en_harmonic_oscillator(sr, sz):
     # calculate energy of the quantum harmonic oscillator with N particles and harmonic length harmonic_length
    return 0.5 * (sr**2 / 10000**2 + sz**2 / 100000**2)


def en_per_particle(sr, sz, nparticles, alpha, beta, gamma, include_ho=True):
    kappa = sr / sz
    kin = 0.25 * (2./sr**2 + 1./sz**2) 
    N = nparticles
    en_mflhy = np.sqrt(2)*np.pi**(-1.5)*N*alpha/(4*sr**2*sz) + np.pi**(-1.5*gamma)*beta*np.exp(gamma*np.log(N) - 2*gamma*np.log(sr) - gamma*np.log(sz))/(gamma + 1)**(3/2)
    en_int  = en_mflhy +  N/(2*(2*np.pi)**1.5 * sr**2 * sz) * (- 4*np.pi/3 * fdip(kappa) ) 
    return kin   + en_int  + (en_harmonic_oscillator(sr, sz) if include_ho else 0)

def min_en_per_particle(sr, sz, nparticles, alpha, beta, gamma, include_ho=True):
      return np.min(en_per_particle(sr, sz, nparticles, alpha, beta, gamma, include_ho))

def local_minima(array2d):
    # find local minima in 2d array.
    # Returns array of indices of local minima
    # True if the point is a local minimum
    return ((array2d <= np.roll(array2d,  1, 0)) &
            (array2d <= np.roll(array2d, -1, 0)) &
            (array2d <= np.roll(array2d,  1, 1)) &
            (array2d <= np.roll(array2d, -1, 1)))

def estimate_nc(alpha, beta, gamma, include_ho=True, based_on='energy', verbose=False):
    # Critical atom number is the number of atoms for which the energy of a system crosses zero.
    # The energy of a system is given by the second element of the tuple returned by en_particles.
    # The first element of the tuple is the optimal x_0.
    # Implement the root-finding algorithm to find the critical atom number
    # The function should return the critical atom number and the optimal x_0 for that atom number
    # The function should also return the energy of the system for the critical atom number.
    nparticles = 3*10**5

    

    

    iter_num = 0
    
    # Run the code below if the based_on is 'energy'
    if based_on == 'energy':
        ng = 500
        sr_range = np.linspace(4, 200, ng)
        sz_range = np.linspace(4, 2000, ng)
        # make meshgrid
        sr, sz = np.meshgrid(sr_range, sz_range)
        while True:
            nparticles_new = nparticles * 0.95
            en_0 = min_en_per_particle(sr, sz, nparticles, alpha, beta, gamma, include_ho=False)
            en_1 = min_en_per_particle(sr, sz, nparticles_new, alpha, beta, gamma, include_ho=False)
            if en_0 * en_1 < 0 and en_0 < 0 and en_1 > 0:
                break
            nparticles = nparticles_new
            iter_num += 1
            if iter_num > 4000:
                # Raise value error
                raise ValueError("Too many iterations")
                # break the will loop
                break
        return nparticles, en_0
        
    if based_on == 'size':
        ng = 300
        sr_range = np.linspace(0.1, 200, ng)
        sz_range = np.linspace(0.1, 2000, ng)
        # make meshgrid
        sr, sz = np.meshgrid(sr_range, sz_range)
        
        # Calculate the critical atom number in the following way:
        # 1. Find if there is a local minima of the energy for a given number of atoms
        # 2. if there exists local minimum, decrease the number of atoms. If not, break the loop
        while True:
            en_0 = en_per_particle(sr, sz, nparticles, alpha, beta, gamma, include_ho=False)
            local_min = local_minima(en_0)
            
            # if there exists local minimum, decrease the number of atoms. If not, break the loop
            if np.sum(local_min) > 0:
                nparticles_new = nparticles * 0.9
                nparticles = nparticles_new
                
                # print the sr and sz values for the local minima
                sr_local_min = sr.flatten()[local_min.flatten()]
                sz_local_min = sz.flatten()[local_min.flatten()]
                if verbose:
                    print(f"nparticles: {nparticles}, sr_local_min: {sr_local_min}, sz_local_min: {sz_local_min}")
                
                if np.min(en_0) > 0 and (np.min(sr_local_min) > sr_range[-2] or np.min(sz_local_min) > sz_range[-2]):
                    break
                
                iter_num += 1
                if iter_num > 4000 or nparticles < 10:
                    # Raise value error
                    print("Too many iterations")
                    return 0, 0
                    # raise ValueError("Too many iterations")
                    # break the will loop
                    break
            else:
                break

        nc = nparticles
        return nc, np.min(en_0)
    
    
def get_optimal_sr_sz(alpha, beta, gamma, nparticles):
    ng = 500
    sr_range = np.linspace(4, 200, ng)
    sz_range = np.linspace(4, 2000, ng)
    # make meshgrid
    sr, sz = np.meshgrid(sr_range, sz_range)

    # Calculate energy for each sr and sz
    en_0 = en_per_particle(sr, sz, nparticles, alpha, beta, gamma, include_ho=False)
    
    # Find the local minima
    local_min = local_minima(en_0)

    # Get the sr and sz values for the local minima
    sr_local_min = sr.flatten()[local_min.flatten()][0]
    sz_local_min = sz.flatten()[local_min.flatten()][0]
    en_0 = en_per_particle(sr_local_min, sz_local_min, nparticles, alpha, beta, gamma, include_ho=False)
    
    # return the optimal sr and sz
    return {"sr": sr_local_min, "sz": sz_local_min, "en_0": en_0}