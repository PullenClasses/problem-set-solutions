import matplotlib.pyplot as plt
from numpy import genfromtxt, arange
from numpy.random import choice
from scipy.stats import gaussian_kde

ints    = genfromtxt('intersections.csv', delimiter=',')
# idxs    = choice(arange(len(ints[0])), size=1000, replace=False)
# color   = gaussian_kde(ints.T[idxs].T)(ints)

fig = plt.figure(figsize=(7,7), constrained_layout=True)
ax  = fig.add_subplot(111)

# idx = color.argsort()
# *ints, color = *ints.T[idx].T, color[idx]

ax.scatter(ints[0],ints[1], s=0.05, marker='.', c='k', alpha=0.5) # type: ignore
ax.set_xlabel('c')
ax.set_ylabel('intersection-height')
plt.show()
