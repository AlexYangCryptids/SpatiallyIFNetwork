import numpy as np
from matplotlib import pyplot as plt

def eta(tau_r, tau_d, t):
    if t <= 0:
        return 0
    else:
        up = np.exp(-t/tau_d) - np.exp(-t/tau_r)
        down = tau_d - tau_r
        return up/down

t_list = np.linspace(0, 50, 100)
result_list1 = []
result_list2 = []
result_list3 = []

for t in t_list:
    result_list1.append(eta(1, 2, t))
    result_list2.append(eta(1, 4, t))
    result_list3.append(eta(1, 8, t))

plt.plot(t_list, result_list1, color = 'b', label = 'tau_d=2')
plt.plot(t_list, result_list2, color = 'r', label = 'tau_d=4')
plt.plot(t_list, result_list3, color = 'g', label = 'tau_d=8')
plt.legend()

plt.show()

print(sum(result_list1))
print(sum(result_list2))
print(sum(result_list3))