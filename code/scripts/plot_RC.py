
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# own imports
from TS2VEC import random_cropping
from data import load_some_signals


# plot settings
import scienceplots
plt.style.use('science')
plt.rcParams.update({'figure.dpi': '200'})
plt.rcParams.update({"legend.frameon": True})

some_signals = load_some_signals(1)



test = random_cropping(verbose=False)


fig, axs = plt.subplots(2, 2, figsize=(6,6))

axs = axs.flatten()

for i in range(4):

    crop1, crop2, cuts = test.forward(some_signals[0])


    #ax = fig.add_subplot(111)


    max_, min_ = np.max(some_signals[0]), np.min(some_signals[0])

    height = max(abs(max_), abs(min_))


    rect1 = Rectangle((cuts['a1'], -height), cuts['b1'] - cuts['a1'], 2*height, alpha=0.3, color='b')

    rect2 = Rectangle((cuts['a2'], -height), cuts['b2'] - cuts['a2'], 2*height, alpha=0.3, color='g')


    axs[i].add_patch(rect1)
    axs[i].add_patch(rect2)

    axs[i].plot(some_signals[0])

plt.savefig('/Users/nikolajhertz/Desktop/GIT/BACHELOR_THESIS/code/results/random_cropping.png')

plt.show()


