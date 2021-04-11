import numpy as np
import matplotlib.pyplot as plt
def wrapped_gaussian(x_mat, sigma, k_list):

    g = 0
    for k in k_list:
        temp = - np.power((x_mat+k) ,2) / (2 * sigma ** 2)
        g += np.exp(temp)
        #g += np.exp(-(x+k) **2 / (2* sigma**2))

    return g / (np.sqrt(2*np.pi) * sigma)

sigma = 0.1
k_range = 10
k_list = []
for i in range(2*k_range + 1):
    k_list.append(-k_range+i)
print(k_list)

discrete_num = 100
g = np.zeros((discrete_num, discrete_num))
#g_temp = np.zeros((2, discrete_num, discrete_num))
W = np.zeros((discrete_num, discrete_num), dtype=int)
string_list = np.linspace(0.5/discrete_num, 1-0.5/discrete_num, discrete_num)

string_map = np.zeros((discrete_num))
p_mean = 0.5

# here, make connection
for i in range(discrete_num):
    for j in range(discrete_num):
        distance = abs(i -j)/discrete_num
        g_temp = wrapped_gaussian(distance, sigma, k_list)
        g[i,j] = g_temp
        if p_mean * g_temp > np.random.uniform(0, 1):
            W[i, j] = 1

# then check connection
for x in range(discrete_num):
    string_map += g[x]
    #print(g[x])

plt.plot(string_list, string_map, 'o')
print(string_map)

#plt.plot(string_list, g[1], 'o')

plt.xlim([0,1])
plt.show()


