# Study of the effect of the box size on the equilibrium of the system

The purpose of this study is to find the appropriate box size and number of particles for the system. The box size is varied, together with the number of grid points. 

## Results

The results are summarized in the following table:


| Name | Box size ($r_0$) | Number of grid points (nxyz) | 
|---------------|---------------|-----------------------|
| large-box-large-dx | np.array([90, 50, 41]) * np.array([0.9, 0.75, 1]) * 50            | np.array([256, 128, 64]) / 2     
| small-box-large-dx | np.array([90, 50, 41]) * np.array([0.9, 0.75, 1]) * 50 /2           | np.array([256, 128, 64]) / 2 / 2       
| small-box-small-dx | np.array([90, 50, 41]) * np.array([0.9, 0.75, 1]) * 50 /2           | np.array([256, 128, 64]) / 2          
| large-box-small-dx-2 | np.array([90, 50, 41]) * np.array([0.9, 0.75, 1]) * 50            | np.array([256, 64, 32])        
| small-box-small-dx-2 | np.array([90, 50, 41]) * np.array([0.9, 0.75, 1]) * 50 /2           | np.array([256, 64, 32]) / 2          
