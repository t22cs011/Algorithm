import numpy as np

import matplotlib.pyplot as plt

# Define the function T(t)
def T(t, T0):
    return T0 / np.log(1 + t)

# Parameters
T0 = 10  # Example value for T0
t = np.linspace(1, 100, 500)  # Avoid t=0 to prevent division by zero

# Compute T(t)
T_values = T(t, T0)

# Plot the function
plt.figure(figsize=(8, 5))
plt.plot(t, T_values, label=r'$T(t) = \frac{T_0}{\log(1+t)}$', color='blue')
plt.title('Simulated Annealing$', fontsize=14)
plt.xlabel('$t$', fontsize=12)
plt.ylabel('$T(t)$', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('log.png', dpi=300)