# Import dependencies

import numpy as np
# from numpy import linalg
import math
import sys
import os
import time
from tqdm import tqdm
import argparse


class eGPE:

    def __init__(self,
                 nparticles,
                 nxyz=[32, 32, 32],
                 box_size=[200, 200, 2000],
                 eps_dd=None,
                 a_s=None,
                 psi_0=None,
                 fx=None,
                 fy=None,
                 fz=None,
                 lambda_pot_external=0,
                 beta=None,
                 gamma=None,
                 contact_interaction=True,
                 dipolar_interaction=True,
                 verbose=False,):
        """
        Initializes a DFT simulation of the Gross-Pitaevskii equation for a 3D BEC.
        Args:
            nparticles (int): number of particles
            nxyz (list): number of grid points in each direction
            box_size (list): size of the simulation box in each direction, in units of r_0 = 387.7 a_0 (obtained from https://www.wolframalpha.com/input?i=%28162+atomic+mass+unit%29+*+%28mu_0%29+*+%289.93+bohr+magneton%29%5E2+%2F+%284+pi+hbar%5E2%29+%2F+%28bohr+radius%29)
            eps_dd (float): interaction strength, dimensionless, equal to 1. / 3 / (self.a_s / self.r_0)
            a_s (float): interaction strength, in units of Bohr radius.
            psi_0 (np.array): initial wavefunction
            fx (float): frequency of the oscillator in the x direction, units of Hz
            fy (float): frequency of the oscillator in the y direction, units of Hz
            fz (float): frequency of the oscillator in the z direction, units of Hz
            lambda_pot_external (float): strength of the perturbation in the x direction, dimensionless
            beta (float): beyond mean-field parameter,
            gamma (float): beyond mean-field parameter, dimensionless,
            contact_interaction (bool): include contact interaction in the simulation
            dipolar_interaction (bool): include dipolar interaction in the simulation
            output_dir (str): path to the output directory
            verbose (bool): print additional information
        Returns:
            psi (np.array): wavefunction at the end of the simulation
        """
        
        # Set constants
        self.r_0 = 387.654009
        # (162 atomic mass unit) * (mu_0) * (9.93 bohr magneton)^2 / (4 pi hbar^2) / (bohr radius)
        # = (162 * 1.66053907e-27 kilograms) * (1.25663706e-6 m kg s^-2 A^-2) * (9.93 * 9.2740100783E-24 J/T)^2 / (4 pi (1.05457182e-34 m^2 kg / s)^2) / (5.291772109E-11 m)
        
        # User can set either eps_dd or a_s, but not both.
        # If eps_dd is set, a_s is calculated from it.
        # if a_s is set, eps_dd is calculated from it.
        
        if eps_dd is not None and a_s is not None:
            raise ValueError("Cannot set both eps_dd and a_s. Please set only one.")
        elif eps_dd is not None:
            self.eps_dd = eps_dd
            self.set_as()

        elif a_s is not None:
            self.a_s = a_s / self.r_0
            self.set_eps_dd()

        # Set parameters
        self.nparticles = nparticles
        self.nxyz = np.array(nxyz, dtype=np.int32)
        self.box_size = np.array(box_size, dtype=np.float64)
        self.verbose = verbose
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.lambda_pot_external = lambda_pot_external

        
        # sqrt((1.05457182e-34 m^2 kg / s) / ((162 *(1.66053907e-27 kilograms)) * 2 * pi * Hertz)) = 7.89889045 microns
        self.lho_unit = 7.89889045E-06 # taken from: https://www.wolframalpha.com/input/?i=sqrt%28hbar+%2F+%28%28162+atomic+mass+unit%29+*+2+*+pi+*+Hertz%29%29

        self.mu_m = 1.E-06 / (self.r_0 * 5.291772109E-11)
        self.EPS = 1.E-10

        

        self.alpha_mflhy = 2*np.pi*self.a_s
        self.alpha = self.alpha_mflhy

        beta_mflhy = 256*np.sqrt(np.pi)*self.a_s**(5/2) / \
            15 + 128*np.sqrt(np.pi)*np.sqrt(self.a_s)/45
        gamma_mflhy = 1.5
        self.beta_mflhy = beta_mflhy
        self.gamma_mflhy = gamma_mflhy

        self.beta = beta
        self.gamma = gamma

        if beta is None:
            self.beta = beta_mflhy
        if gamma is None:
            self.gamma = gamma_mflhy

        if contact_interaction == False:
            self.alpha = 0.
            self.beta = 0.
        
        self.contact_interaction = contact_interaction
        self.dipolar_interaction = dipolar_interaction

       

        # Set harmonic oscillator parameters
        if fx is None or fx < self.EPS:
            self.fx = self.EPS
        if fy is None or fy < self.EPS:
            self.fy = self.EPS
        if fz is None or fz < self.EPS:
            self.fz = self.EPS

        self.w_ho = np.array([self.fx, self.fy, self.fz])
        self.a_ho = self.lho_unit / np.sqrt(self.w_ho) / 1.E-06 * self.mu_m

        # replace inf values in a_ho with 1E+10
        self.a_ho = np.nan_to_num(self.a_ho, posinf=1E+10)

        # Set grid parameters
        self.update_grid_params()

        # Set initial wavefunction
        if psi_0 is None:
            print("[INFO] Initializing psi")
            self.set_random_psi()
        else:
            self.psi = psi_0

        # Normalize wavefunction as a double check
        self.normalize_psi()

        # Set initial densities (and potential)
        self.den = np.abs(self.psi)**2
        self.phi_dd = self.get_phi_dd()

    def update_grid_params(self):
         # Set grid parameters
        self.nx, self.ny, self.nz = self.nxyz
        self.Lx, self.Ly, self.Lz = self.box_size
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz

        self.d3r = self.dx * self.dy * self.dz

        self.x = np.linspace(-self.Lx/2, self.Lx/2, self.nx, endpoint=False)
        self.y = np.linspace(-self.Ly/2, self.Ly/2, self.ny, endpoint=False)
        self.z = np.linspace(-self.Lz/2, self.Lz/2, self.nz, endpoint=False)

        self.x, self.y, self.z = np.meshgrid(
            self.x, self.y, self.z, indexing='ij')

        self.kx = np.fft.fftfreq(self.nx, self.dx / (2 * np.pi))
        self.ky = np.fft.fftfreq(self.ny, self.dy / (2 * np.pi))
        self.kz = np.fft.fftfreq(self.nz, self.dz / (2 * np.pi))
        self.kx, self.ky, self.kz = np.meshgrid(
            self.kx, self.ky, self.kz, indexing='ij')
        self.KSquared = self.kx**2 + self.ky**2 + self.kz**2
        self.KSquared[self.KSquared == 0] = 1e-10
        self.KVec = np.sqrt(self.KSquared)

        # Set dipolar interaction parameters
        if self.dipolar_interaction == True:
            self.ft_dip = 4*np.pi/3 * \
                (2*self.kz**2 - self.kx**2 - self.ky**2) / self.KVec**2
            self.ft_dip = np.nan_to_num(self.ft_dip, posinf=0)
        else:
            self.ft_dip = 0.
            
            
        # set the external potential
        self.set_external_potential()

    def load_egpe(self, filename):
        """
        Loads the egpe object from a file.
        """
        with open(filename, 'rb') as f:
            egpe = pickle.load(f)
        self.__dict__.update(egpe.__dict__)

    def set_nxyz(self, nxyz):
        """
        Sets the grid size.
        """
        self.nxyz = nxyz

    def set_box_size(self, box_size):
        """
        Sets the box size.
        """
        self.box_size = box_size
        
    
    def set_as(self):
        self.a_s = 1. / 3 / self.eps_dd
    
    def set_eps_dd(self):
        self.eps_dd = 1. / 3 / (self.a_s / self.r_0)

    def potential_external(self):
        """
        Returns the external potential.
        """
        return 0.5 * ((self.x/self.a_ho[0])**2/self.a_ho[0]**2
                      + (self.y/self.a_ho[1])**2/self.a_ho[1]**2
                      + (self.z/self.a_ho[2])**2/self.a_ho[2]**2)

    def set_external_potential(self):
        """
        Sets the external potential.
        """
        self.VExt = self.potential_external()
        + self.lambda_pot_external * (self.x/self.a_ho[0])**2/self.a_ho[0]**2

    def get_phi_dd(self):
        """
        Returns the dipolar interaction potential.
        """
        return np.fft.ifftn(np.fft.fftn(self.den) * self.ft_dip)

    def energy_density_interaction(self):
        """
        Returns the energy density of the interaction potential.
        Returns:
            energy_density (np.array): energy density of the interaction potential
        """
        return self.alpha*self.den**2 + self.beta*self.den**(self.gamma+1.) + 0.5 * self.phi_dd * self.den

    def dEps_dPsi(self):
        """
        Returns the derivative of the interaction potential with respect to the wavefunction.
        Returns:
            dEps_dPsi (np.array): derivative of the interaction potential with respect to the wavefunction
        """
        return 2.*self.alpha*self.den + self.beta*(self.gamma + 1.)*self.den**self.gamma + self.phi_dd

    def normalize_psi(self):
        self.psi *= np.sqrt(self.nparticles)/np.sqrt(self.d3r *
                                                     np.sum(np.absolute(self.psi) ** 2))
        self.den = np.abs(self.psi)**2
        self.phi_dd = self.get_phi_dd()

    def set_gaussian_psi(self, sr, sz):
        """
        Initializes the wavefunction as a gaussian.
        Returns:
            psi (np.array): wavefunction
        """
        r = np.sqrt(self.x**2 + self.y**2)
        self.psi = np.exp(-0.5 * (r/sr)**2) * np.exp(-0.5 * (self.z/sz)**2) + 0.j
        self.normalize_psi()
        
    def set_random_psi(self):
        """
        Initializes the wavefunction as a random wavefunction.
        Returns:
            psi (np.array): wavefunction
        """
        # self.psi = np.exp(-0.5 * self.x**2) * np.exp(-0.5 * self.y**2) * np.exp(-0.5 * self.z**2) + 0.j
        self.psi = np.random.uniform(0.95, 1, np.shape(self.x)) + 0.j
        self.normalize_psi()

    def energy_contributions(self):
        """
        Returns the energy of the system.
        """
        potext = self.den * self.VExt
        pot_int = self.energy_density_interaction()
        kin_tot = (np.conj(self.psi) *
                   np.fft.ifftn(np.fft.fftn(self.psi) * self.KSquared/2)).real
        return {"kinetic": self.d3r*np.sum(kin_tot),
                "pot_ext": self.d3r * np.sum(potext),
                "pot_int": self.d3r*np.sum(pot_int).real
                }

    def T2_operator(self):
        """
        Performs the T2 operator on the wavefunction.
        """

        # First part of the T2 operator, propagator exp(-i*V*dt/2)
        pot = self.VExt + self.dEps_dPsi()
        self.psi *= np.exp(-0.5j * pot * self.dt)

        # Second part of the T2 operator, propagator exp(-i*K^2*dt/2)
        self.psi = np.fft.fftn(self.psi)
        self.psi *= self.kinprop
        self.psi = np.fft.ifftn(self.psi)

        # Third part of the T2 operator, propagator exp(-i*V*dt/2)
        self.den = np.absolute(self.psi) ** 2
        self.phi_dd = self.get_phi_dd()
        pot = self.VExt + self.dEps_dPsi()
        self.psi *= np.exp(-0.5j * pot * self.dt)

        # normalize the wavefunction if imaginary time propagation
        if self.time_prop == "imag":
            self.normalize_psi()

        # update the density and the dipolar interaction potential
        self.den = np.absolute(self.psi) ** 2
        self.phi_dd = self.get_phi_dd()

    def evolve(self,
               dt,
               t_max,
               time_prop="imag",
               verbose=False,
               print_each_percent=5,
               output_dir="./"):
        """
        Evolution of the wavefunction.
        Args:
            dt (float): time step, units of m_162u * r_0^2 / hbar
            t_max (float): maximum simulation time, units of m_162u * r_0^2 / hbar
            time_prop (str): propagation of time. "real" or "imag"
            verbose (bool): print additional information
            output_dir (str): path to the output directory
        Returns:
            psi (np.array): wavefunction at the end of the simulation
        """

        # set number of time steps
        n_sim_steps = int(t_max // dt)
        print_each = int(n_sim_steps * print_each_percent/ 100)
        

        # determine the timestep
        if time_prop == "real":
            dt = dt
        elif time_prop == "imag":
            dt = -1.j * dt

        self.time_prop = time_prop
        self.dt = dt
        # initialize a kinetic propagator
        self.kinprop = np.exp(-1.j * self.dt * self.KSquared / 2.)

        en_0 = np.inf
        # Loop over time steps and apply the T2 operator
        for i in tqdm(range(n_sim_steps)):
            self.T2_operator()

            if verbose and i % print_each == 0:
                en = self.energy_contributions()
                total_en = en["kinetic"] + en["pot_ext"] + en["pot_int"]
                # print kinetic, potential, total energy
                print("Kinetic energy: ", en["kinetic"])
                print("Potential energy: ", en["pot_ext"] + en["pot_int"])
                print("Total energy: ", total_en)
            
            
            if i % 500 == 0:
                en = self.energy_contributions()
                total_en = en["kinetic"] + en["pot_ext"] + en["pot_int"]
                
                # if total_en and en_0 are very close, stop the simulation
                if np.abs(total_en - en_0) < 1e-4:
                    print("Total energy converged")
                    return
                
                en_0 = total_en
                
                


    def get_avg_abs_x(self):
        """
        Returns the average absolute value of the wavefunction in the x direction.
        """
        return np.sum(self.den * np.abs(self.x)) * self.d3r / np.sum(self.den * self.d3r)
    
    def get_avg_abs_y(self):
        """
        Returns the average absolute value of the wavefunction in the y direction.
        """
        return np.sum(self.den * np.abs(self.y)) * self.d3r / np.sum(self.den * self.d3r)
    
    def get_avg_abs_z(self):
        """
        Returns the average absolute value of the wavefunction in the z direction.
        """
        return np.sum(self.den * np.abs(self.z)) * self.d3r / np.sum(self.den * self.d3r)