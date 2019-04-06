import numpy as np
import matplotlib.pyplot as plt

s = np.random.uniform(0, 1, 1200)
count, bins, ignored = plt.hist(s, 12, normed = True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.show()