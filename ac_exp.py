import numpy as np
import matplotlib.pyplot as plt
import tsfeatures

x = np.linspace(0, 10*np.pi, 1000)
y = np.sin(x)

acs = tsfeatures.autocorrelation.time_series_all_autocorrelation(y)
plt.subplot(211)
plt.plot(x, y)
plt.subplot(212)
plt.plot(acs)
plt.show()

index = np.arange(len(y))
max_lag = len(y) - 3

plt.ion()
for lag in range(max_lag):
    plt.plot(index, y, color="blue")
    plt.plot(index+lag, y, color="red")
    plt.pause(0.05)

plt.ioff()
plt.show()