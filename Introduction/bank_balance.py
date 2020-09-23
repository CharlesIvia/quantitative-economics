import numpy as np
import matplotlib.pyplot as plt


r = 0.025
T = 50

b = np.empty(T+1)
b[0] = 10

for t in range(T):
    b[t+1] = (1+r)*b[t]

plt.plot(b, label="bank balance")
plt.legend()
plt.show()