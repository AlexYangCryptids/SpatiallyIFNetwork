import numpy as np
import matplotlib.pyplot as plt


def eta(tau_r, tau_d, t):
    if t <= 0:
        return 0
    else:
        up = np.exp(-t/tau_d) - np.exp(-t/tau_r)
        down = tau_d - tau_r
        return up/down


def gen_poisson_spike(Nx, T, rx, step_size):
    poisson_spike_train = np.zeros((Nx, T),dtype=int)
    prob_delta_t = rx * 0.001 * step_size
    for t in range(T):
        for n in range(Nx):
            if prob_delta_t > np.random.uniform(0, 1):
                poisson_spike_train[n, t] = 1
    return poisson_spike_train


VL = -60 # leaky voltage
VT = -50 # exponential voltage eq3 see the paper
delta_T_e = 2 # voltage , also ses the paper, to E
delta_T_i = 0.5 # voltage , also ses the paper, to I
V_threshold = -10 # threshold voltage
Vre = -65 # reset voltage
t_ref = 1.5
tau_m = 15
tau_r = 1
tau_d = 2
tau_d_2 = 8

step_size = 0.05
total_step = 3000


V_list = []
V_list_2 = []
V_now = -50
ref_state = 1
ref_time = 0

rx = 25
J_ff = 20
poisson_train = gen_poisson_spike(1, total_step, rx, step_size)
print(np.where(poisson_train == 1))
print('poisson spike successfully generated!')

for step in range(total_step):
    if step % 100 == 0:
        print(step)

    synapse_input_ff = 0
    
    if step > 1:
        for past_step in range(step):
            synapse_input_ff += poisson_train[:, past_step] * eta(tau_r, tau_d,
                                                                  step_size * (step - past_step))

    if ref_state == 1:
        #synapse_input_ff = poisson_train[0, step]
        #print(synapse_input_ff)
        delta_V = -(V_now-VL)/tau_m + delta_T_e*np.exp((V_now-VT)/delta_T_e)/tau_m  + J_ff * synapse_input_ff
        #delta_V = -(V_now - VL) / tau_m + J_ff * synapse_input_ff
        V_next = V_now + delta_V*step_size
        if V_next > V_threshold:
            V_next = Vre
            ref_state = 0
            ref_time = t_ref

    elif ref_state == 0:
        V_next = V_now
        ref_time -= step_size

        if abs(ref_time) < 1e-5:
            ref_time = 0
            ref_state = 1

    V_list.append(V_next)
    V_now = V_next


#reset V_now for another time scale
V_now = -50

for step in range(total_step):
    if step % 100 == 0:
        print(step)

    synapse_input_ff = 0

    if step > 1:
        for past_step in range(step):
            synapse_input_ff += poisson_train[:, past_step] * eta(tau_r, tau_d_2,
                                                                  step_size * (step - past_step))

    if ref_state == 1:
        # synapse_input_ff = poisson_train[0, step]
        # print(synapse_input_ff)
        delta_V = -(V_now - VL) / tau_m + delta_T_e * np.exp((V_now - VT) / delta_T_e) / tau_m + J_ff * synapse_input_ff
        # delta_V = -(V_now - VL) / tau_m + J_ff * synapse_input_ff
        V_next = V_now + delta_V * step_size
        if V_next > V_threshold:
            V_next = Vre
            ref_state = 0
            ref_time = t_ref

    elif ref_state == 0:
        V_next = V_now
        ref_time -= step_size

        if abs(ref_time) < 1e-5:
            ref_time = 0
            ref_state = 1

    V_list_2.append(V_next)
    V_now = V_next


plt.plot([step_size*i for i in range(total_step)], V_list, 'b', label = 'taud = 2')
plt.plot([step_size*i for i in range(total_step)], V_list_2, 'r',label = 'tau_d = 8')
plt.legend()
plt.show()
