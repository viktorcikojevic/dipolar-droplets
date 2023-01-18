# Estimate the frequencies

By performing the tests in [../2-find-appropriate-lambda](../2-find-appropriate-lambda), we found that the it is ok to use the value for $\lambda$ of 0.01. We now use this value to estimate the frequencies. These are the steps:

1. Find the equilibrium of the system with the given $\lambda$.
1. Start the real-time evolution with the initial conditions of the equilibrium, but having the $\lambda$ value of 0. Write down the <$x^2$> during the evolution.
1. Find the frequencies from the <$x^2$> data.