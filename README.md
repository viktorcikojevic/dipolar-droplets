# Dipolar Droplets

Contains code for the study of Dy dipolar droplets with the combination of QMC and DFT techniques.


# Description of notebooks

- Notebook [derivation_of_variational_energy.ipynb](derivation_of_variational_energy.ipynb) contains the code for the mathematical derivation necessary for the project, using sympy.
- Notebook [critical_atom_number.ipynb](critical_atom_number.ipynb) contains the code for the calculation of the critical atom number for the formation of a dipolar droplet. 
- Notebook [analyze-density-range.ipynb](analyze-density-range.ipynb) contains code to analyze the dependence of $N_c$, $\beta$ and $\gamma$ on the lower and upper density range of the QMC fits.
- Notebook [beta-gamma-fit-vs-a_s.ipynb](beta-gamma-fit-vs-a_s.ipynb) contains the code for the fit of beta and gamma of recipe 8 vs $a_s$.


# Description of outputs

- Folder [energies-qmc](energies-qmc) contains the energies of the QMC calculations for the different atom numbers.
- Folder [experimental-data](experimental-data) contains the experimental data for the critical atom number.
- Folder [plots](plots) contains the plots of the project made by the notebooks