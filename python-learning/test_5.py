import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.tanh(x)

fig = plt.plot(x, y)

plt.show()
