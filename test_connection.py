import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def wrapped_gaussian(x_mat, sigma, k_list):

    g = 0
    for k in k_list:
        #print('kekeke')
        #temp = - np.power((x_mat+2*np.pi*k) ,2) / (2 * sigma ** 2)
        temp = - np.power((x_mat + k), 2) / (2 * sigma ** 2)
        g += np.exp(temp)
        #g += np.exp(-(x+k) **2 / (2* sigma**2))

    return g / (np.sqrt(2*np.pi) * sigma)


sigma = 0.5
k_range = 50
k_list = []
for i in range(2*k_range + 1):
    k_list.append(-k_range+i)
print(k_list)
#x_list = np.linspace(-np.pi/2, np.pi/2, 50)
x_list = np.linspace(-1, 1, 200)
result_list1 = []
result_list2 = []
result_list3 = []
result_list4 = []

for x in x_list:
    result_list1.append(wrapped_gaussian(x, 0.5, k_list))
    result_list2.append(wrapped_gaussian(x, 0.4, k_list))
    result_list3.append(wrapped_gaussian(x, 0.3, k_list))
    #result_list4.append(wrapped_gaussian(x, 0.2, k_list))


plt.plot(x_list, result_list1, color='b', label='sigma=0.5')
plt.plot(x_list, result_list2, color='r', label='sigma=0.4')
plt.plot(x_list, result_list3, color='g', label='sigma=0.2')
#plt.plot(x_list, result_list4, color='y', label='sigma=0.2')

plt.legend()
plt.show()