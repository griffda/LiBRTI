import matplotlib.pyplot as plt
import numpy as np

results_folder = "/home/tmg25bcx/FESITM/task01"

data = np.genfromtxt(
    results_folder + "/mobile_concentration.txt", skip_header=1, delimiter=","
)

plt.plot(data[:, 0], data[:, 4], label="1.0 s")
plt.plot(data[:, 0], data[:, 3], label="0.5 s")
plt.plot(data[:, 0], data[:, 2], label="0.2 s")
plt.plot(data[:, 0], data[:, 1], label="0.1 s")

plt.xlabel("x (m)")
plt.ylabel("Mobile concentration (H/m3)")
plt.legend()
plt.show()

data = np.genfromtxt(
    results_folder + "/mobile_concentration_steady.txt", skip_header=1, delimiter=","
)

plt.plot(data[:, 0], data[:, 1])
plt.xlabel("x (m)")
plt.ylabel("Mobile concentration (H/m3)")
plt.show()
