import numpy as np

# test auto bord

a = np.arange(5)
m = 3
a = np.array([a[i:i+m] for i in range(a.size-m+1)])
a1 = a[:, np.newaxis]
a2 = a[np.newaxis, :]

print(a, end="\n\n")

print(a1, end="\n\n")

print(a2, end="\n\n")

print(a1 - a2, end="\n\n")

print(a2 - a1, end="\n\n")