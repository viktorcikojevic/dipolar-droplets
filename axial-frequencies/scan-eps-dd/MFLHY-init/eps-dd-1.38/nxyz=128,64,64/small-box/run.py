# Import dependencies
import numpy as np
import sys
sys.path.append('../../../egpe')
from egpe import eGPE





# Define the parameters
nparticles = 35_000
fx, fy, fz = 18.5, 53, 81
# Harmonic oscillator length is given by a_ho = sqrt(hbar/(m*omega)), where m is the mass of the atom, and omega is the 2*pi*f0, where f0 is the frequency of the trap.
# We want to calculate a_h0 / r_0 = sqrt(hbar/(m*omega)/r_0, where r_0 = 390 a_0.
# a_h0/r_0 = sqrt(1.05457182e-34 m^2 kg / s / ( 162 * (1.66053907e-27 kilograms) * (2*pi*20 Hertz)) / (390 * 5.291772109E-11 m)= 1.7662456 microns / ((390 * 5.291772109E-11 m)) = 85.5825757
# a_h0_y = a_h0 * sqrt(18.5/53) = = 85.5825757 * sqrt(18.5/53) = 50.563052245
# a_h0_z = a_h0 * sqrt(18.5/81) = = 85.5825757 * sqrt(18.5/81) = 40.9005085201



# Get the optimal sr and sz
gp = eGPE(eps_dd=1.38,
          nparticles=nparticles,
          fx=fx, fy=fy, fz=fz,
          nxyz=np.array([128, 64, 64])  ,
          box_size=np.array([800, 500, 500]) * 2 * 1.5,
          lambda_pot_external=0.001
          # rho_cutoff=0.8,
          # z_cutoff=0.8,
          )

r0 = gp.box_size / 20
gp.psi  = np.exp(-0.5 * (gp.x / r0[0]-1)**2 - 0.5 * (gp.y / r0[1])**2 - 0.5 * (gp.z / r0[2])**2) + 0j
gp.psi += np.exp(-0.5 * (gp.x / r0[0]+1)**2 - 0.5 * (gp.y / r0[1])**2 - 0.5 * (gp.z / r0[2])**2) + 0j
gp.normalize_psi()





# first evolve slowly, with no output
gp.evolve(dt=1, 
          t_max=1E+03,
          verbose=False)


# Evolve faster, with output
gp.evolve(dt=10, 
          t_max=5E+05,
          verbose=False, 
          print_each_percent=1, 
          output_root_dir="1-equilibrate-output")



# set lambda to 0 and update the external potential
gp.lambda_pot_external = 0.
gp.set_external_potential()


# Set dt, t_max and evolve the system
dt = 10
t_max = 4E+06
gp.evolve(dt=dt, t_max=t_max,  time_prop="real", verbose=False, output_root_dir="2-real-time", save_x2=True, print_each_percent=1.E-04)
