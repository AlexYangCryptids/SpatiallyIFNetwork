import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def wrapped_gaussian(x_mat, sigma, k_list):

    g = 0
    for k in k_list:
        #print('kekeke')
        temp = - np.power((x_mat + k), 2) / (2 * sigma ** 2)
        g += np.exp(temp)
        #g += np.exp(-(x+k) **2 / (2* sigma**2))

    return g / (np.sqrt(2*np.pi) * sigma)


sigma = 0.15
k_range = 50
k_list = []
for i in range(2*k_range + 1):
    k_list.append(-k_range+i)
print(k_list)
#x_list = np.linspace(-np.pi/2, np.pi/2, 50)
x_list = np.linspace(-1, 1, 100)
result_list1 = []
for x in x_list:
    result_list1.append(wrapped_gaussian(x, 0.5, k_list))

fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.meshgrid(x_list, x_list)
Z = wrapped_gaussian(X, sigma, k_list)* wrapped_gaussian(Y, sigma, k_list)
plt.xlabel('x')
plt.ylabel('y')

ax.plot_surface(X, Y, Z, cmap='rainbow')
plt.show()