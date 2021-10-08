import numpy as np
import matplotlib.pyplot as plt
from figure_tools.figure_tools import remove_axis

N = 1000
L = 0.5
M = 5
t = np.linspace(0, 2*np.pi, N)

xt = lambda t: np.cos(t)*(1 + L*np.sin(M*t))
yt = lambda t: np.sin(t)*(1 + L*np.sin(M*t))

x = xt(t)
y = yt(t)

plt.plot(x,y)
remove_axis(plt.gca())

