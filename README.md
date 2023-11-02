# Dipolar Droplets

Contains code for the study of Dy dipolar droplets with the combination of QMC and DFT techniques.


# Description of notebooks

These notebooks are conceptually ordered in the following way:

- Notebook `derivation_of_variational_energy.ipynb` contains the code for the mathematical derivation necessary for the project, using sympy.
- Notebook `critical_atom_number.ipynb` contains the code for the calculation of the critical atom number for the formation of a dipolar droplet. 
- Notebook `analyze-density-range.ipynb` contains code to analyze the dependence of $N_c$, $\beta$ and $\gamma$ on the lower and upper density range of the QMC fits. It calculates the mean $N_c$ for a given density range and the standard deviation of $N_c$ for a given density range. The outputs are files that end with `*functionals_E_N_average_std.dat`. 
- Notebook `beta-gamma-fit-vs-a_s.ipynb` contains the code for the fit of beta and gamma vs $a_s$. This is the final notebook where the plot of critical atom number is made. In this notebook, I compare the $N_c$ obtained from the fit of beta and gamma with the actual evaluations of $N_c$ from the `critical_atom_number.ipynb` notebook. 


# Folders

- Folder [energies-qmc](./energies-qmc) contains the energies of the QMC calculations for the different atom numbers.
- Folder [experimental-data](./experimental-data) contains the experimental data for the critical atom number.
