import numpy as np

def dtw(s1, s2, w):
    w = max(abs(s1.size-s2.size), w)
    s = np.eye(s1.size, s2.size)
    for i in range(1, s1.size):
        for j in range(max(1, w-i), min(s2.size, w+i)):
            d = abs(s1[i]-s2[j])
            s[i,j] = d + min(s[i-1,j-1], s[i, j-1], s[i-1,j])
    return s[-1,-1], np.argmin(s, axis=0)


