import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# this is an all-to-all network

def gen_poisson_spike(Nx, total_step, rx, step_size):
    poisson_spike_train = np.zeros((Nx, total_step))
    prob_delta_t = rx * 0.001 * step_size
    for step in range(total_step):
        for n in range(Nx):
            if prob_delta_t > np.random.uniform(0, 1):
                poisson_spike_train[n, step] = 1
    return poisson_spike_train

def eta(tau_r, tau_d, t):
    if t <= 0:
        return 0
    else:
        up = np.exp(-t/tau_d) - np.exp(-t/tau_r)
        down = tau_d - tau_r
        return up/down



VL = -60 # leaky voltage
VT = -50 # exponential voltage eq3 see the paper
delta_T_e = 2 # voltage , also ses the paper, to E
delta_T_i = 1.5 # voltage , also ses the paper, to I
V_threshold = -10 # threshold voltage
Vre = -65 # reset voltage
t_ref_e = 1.5
t_ref_i = 1.5

tau_m_e = 15
tau_m_i = 15
tau_r_e = 1
tau_r_i = 1
tau_d_e = 4
tau_d_i = 8

step_size = 0.05
total_step = 2000
#I_Cm = 1

N = 100
rx = 2000

J_ef = 0.2
J_if = 0.2
J_ee = 0.043
J_ei = -0.05
J_ie = 0.05
J_ii = -0.043

V_initial = VL
V_mat = np.zeros((N, total_step))
spike_record = np.zeros((N, total_step))
V_now = np.array([V_initial for i in range(N)])
#print(V_now)
ref_state = np.ones((N))
ref_time = np.zeros((N))


poisson_train = gen_poisson_spike(N, total_step, rx, step_size) # poisson firing rate are same to both E & I neuron
print('poisson_train generation finished!')
# 0 - N/2 are E neurons
# N/2 - N are I neurons
boundary = int(N/2)
print(f'the E/I boundary is {boundary}')

for step in range(total_step):
    if step % 20 == 0:
        print(step)

    synapse_input_e = 0
    synapse_input_i = 0
    synapse_input_ff = np.zeros(N)

    if step > 1:
        for past_step in range(step):
            synapse_input_e += sum(spike_record[:boundary, past_step]) * eta(tau_r_e, tau_d_e,
                                                                               step_size * (step - past_step))
            synapse_input_i += sum(spike_record[boundary:, past_step]) * eta(tau_r_i, tau_d_i,
                                                                               step_size * (step - past_step))
            synapse_input_ff += poisson_train[:, past_step] * eta(tau_r_e, tau_d_e,
                                                                      step_size * (step - past_step))

    delta_V_e = -(V_now[:boundary] - VL) / tau_m_e + delta_T_e * np.exp(
        (V_now[:boundary] - VT) / delta_T_e) / tau_m_e + J_ef * synapse_input_ff[
                                                                :boundary] + J_ee * synapse_input_e + J_ei * synapse_input_i

    delta_V_i = -(V_now[boundary:] - VL) / tau_m_i + delta_T_i * np.exp(
        (V_now[boundary:] - VT) / delta_T_i) / tau_m_i + J_if * synapse_input_ff[
                                                                boundary:] + J_ie * synapse_input_e + J_ii * synapse_input_i

    for i in range(N):
        # if not in ref period
        if abs(ref_state[i]-1) < 1e-3:

            if i < boundary: # if it's E neuron
                V_next = V_now[i] + delta_V_e[i] * step_size
                if V_next > V_threshold:
                    spike_record[i, step] = 1
                    V_next = Vre
                    ref_state[i] = 0
                    ref_time[i] = t_ref_e

            else:  # if it's I neuron

                V_next = V_now[i] + delta_V_i[i-boundary] * step_size
                if V_next > V_threshold:
                    spike_record[i, step] = 1
                    V_next = Vre
                    ref_state[i] = 0
                    ref_time[i] = t_ref_i


            V_mat[i, step] = V_next
            V_now[i] = V_next

        # if in ref peroid
        if abs(ref_state[i]) < 1e-3:
            ref_time[i] -= step_size
            # if ref period is over
            if abs(ref_time[i]) < 1e-5:
                ref_state[i] = 1
                ref_time[i] = 0
            V_mat[i, step] = V_now[i]

#print(spike_record)
print('challenge the boss of the gym')


plt.matshow(spike_record, cmap=plt.cm.Blues, aspect='auto')
plt.show()

#plt.matshow(V_mat, cmap='rainbow', aspect='auto')
#plt.colorbar()
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(V_mat, cmap='rainbow', aspect='auto')
temp_scale = [step_size* i for i in np.linspace(0, total_step, 5)]
temp_scale.insert(0, 0)
ax.set_xticklabels(temp_scale)

plt.show()



