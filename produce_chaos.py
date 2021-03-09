import numpy as np
import cv2
import matplotlib.pyplot as plt
from class_IF import EIF_NN_fast
import time
import matplotlib.animation as animation

def compute_xy(i, length):
    x = i % length
    y = int(i / length)
    return x, y

def xy2len(x, y, length):
    unit = 1/length
    return (x+0.5)*unit, (y+0.5)*unit

def wrapped_gaussian(x_mat, sigma, k_list):

    g = 0
    for k in k_list:
        #print('kekeke')
        temp = - np.power((x_mat+k) ,2) / (2 * sigma ** 2)
        g += np.exp(temp)
        #g += np.exp(-(x+k) **2 / (2* sigma**2))

    return g / (np.sqrt(2*np.pi) * sigma)


def compute_projection_EI(N1_1d_e, N1_1d_i, N2_1d_e, N2_1d_i, alpha_e, alpha_i, k_list, p_mean_set):

    # first set different p_means
    p_mean_ee = p_mean_set[0]
    p_mean_ie = p_mean_set[1]
    p_mean_ei = p_mean_set[2]
    p_mean_ii = p_mean_set[3]

    N1_e = N1_1d_e ** 2
    N1_i = N1_1d_i ** 2
    N2_e = N2_1d_e ** 2
    N2_i = N2_1d_i ** 2
    # N is first with E pop, then I pop

    W_ee = np.zeros((N1_e, N2_e))
    W_ie = np.zeros((N1_e, N2_i))
    W_ei = np.zeros((N1_i, N2_e))
    W_ii = np.zeros((N1_i, N2_i))

    # g just contain distance
    g_ee = np.zeros((2, N1_e, N2_e))
    g_ie = np.zeros((2, N1_e, N2_i))
    g_ei = np.zeros((2, N1_i, N2_e))
    g_ii = np.zeros((2, N1_i, N2_i))



    # first compute g_ee
    for i in range(N1_e):
        #print(i)
        x0, y0 = compute_xy(i, N1_1d_e)
        a0, b0 = xy2len(x0, y0, N1_1d_e)
        for j in range(N2_e):
            if i != j:
                x1, y1 = compute_xy(j, N2_1d_e)
                a1, b1 = xy2len(x1, y1, N2_1d_e)
                g_ee[0, i, j] = abs(a0 - a1)
                g_ee[1, i, j] = abs(b0 - b1)

    g_ee = wrapped_gaussian(g_ee, alpha_e, k_list)
    g_ee = g_ee[0] * g_ee[1]
    #print('g finish')

    # then g_ie
    for i in range(N1_e):
        x0, y0 = compute_xy(i, N1_1d_e)
        a0, b0 = xy2len(x0, y0, N1_1d_e)
        for j in range(N2_i):
            if i != j:
                x1, y1 = compute_xy(j, N2_1d_i)
                a1, b1 = xy2len(x1, y1, N2_1d_i)
                g_ie[0, i, j] = abs(a0 - a1)
                g_ie[1, i, j] = abs(b0 - b1)
    g_ie = wrapped_gaussian(g_ie, alpha_e, k_list)
    g_ie = g_ie[0] * g_ie[1]
    #print('g finish')

    # then g_ei
    for i in range(N1_i):
        x0, y0 = compute_xy(i, N1_1d_i)
        a0, b0 = xy2len(x0, y0, N1_1d_i)
        for j in range(N2_e):
            if i != j:
                x1, y1 = compute_xy(j, N2_1d_e)
                a1, b1 = xy2len(x1, y1, N2_1d_e)
                g_ei[0, i, j] = abs(a0 - a1)
                g_ei[1, i, j] = abs(b0 - b1)
    g_ei = wrapped_gaussian(g_ei, alpha_i, k_list)
    g_ei = g_ei[0] * g_ei[1]
    #print('g finish')

    # then g_ii
    for i in range(N1_i):
        x0, y0 = compute_xy(i, N1_1d_i)
        a0, b0 = xy2len(x0, y0, N1_1d_i)
        for j in range(N2_i):
            if i != j:
                x1, y1 = compute_xy(j, N2_1d_i)
                a1, b1 = xy2len(x1, y1, N2_1d_i)
                g_ii[0, i, j] = abs(a0 - a1)
                g_ii[1, i, j] = abs(b0 - b1)
    g_ii = wrapped_gaussian(g_ii, alpha_i, k_list)
    g_ii = g_ii[0] * g_ii[1]
    #print('g finish')

    # W_ee
    prob_mat = np.random.uniform(0, 1, (N1_e, N2_e))
    for i in range(N1_e):
        for j in range(N2_e):
            prob = p_mean_ee * g_ee[i,j]
            if prob > prob_mat[i, j]:
                W_ee[i][j] = 1

    # W_ie
    prob_mat = np.random.uniform(0, 1, (N1_e, N2_i))
    for i in range(N1_e):
        for j in range(N2_i):
            prob = p_mean_ie * g_ie[i, j]
            if prob > prob_mat[i, j]:
                W_ie[i][j] = 1

    # W_ei
    prob_mat = np.random.uniform(0, 1, (N1_i, N2_e))
    for i in range(N1_i):
        for j in range(N2_e):
            prob = p_mean_ei * g_ei[i, j]
            if prob > prob_mat[i, j]:
                W_ei[i][j] = 1

    # W_ii
    prob_mat = np.random.uniform(0, 1, (N1_i, N2_i))
    for i in range(N1_i):
        for j in range(N2_i):
            prob = p_mean_ii * g_ii[i, j]
            if prob > prob_mat[i, j]:
                W_ii[i][j] = 1

    temp_W_1 = np.concatenate((W_ee, W_ie), axis=1)
    temp_W_2 = np.concatenate((W_ei, W_ii), axis=1)
    W = np.concatenate((temp_W_1, temp_W_2), axis=0)

    return W



def gen_poisson_spike(Nx, T, rx, step_size):
    poisson_spike_train = np.zeros((Nx, T))
    prob_delta_t = rx * 0.001 * step_size
    for t in range(T):
        for n in range(Nx):
            if prob_delta_t > np.random.uniform(0, 1):
                poisson_spike_train[n, t] = 1
    return poisson_spike_train





# the input parameters are Ne_1d, initial_V, Wf_in, Wf_out, tau_set, J_set, alpha_rec, p_mean_rec, step_set, V_set, other_NN

Ne_1d = 40
Ni_1d = 40
Nx_1d = 20
Nx = Nx_1d ** 2
Ne_2 = Ne_1d ** 2
Ni_2 = Ni_1d ** 2
Ne_3 = Ne_1d ** 2
Ni_3 = Ni_1d ** 2
#N = Ne_1d + Ni_1d

other_NN_2 = [Nx, 0, 0, Ne_3 + Ni_3]
other_NN_3 =  [0, Ne_2, Ni_2, 0]
NN2_size_set = [Ne_1d, Ni_1d]
NN3_size_set = [Ne_1d, Ni_1d]
initial_V = -60

# set time constant, ms
tau_m_e = 15
tau_m_i = 10
tau_ref_e = 1.5
tau_ref_i = 0.5
tau_er = 1
tau_ed = 5
tau_ir = 1
tau_id = 8
tau_sr = 2
tau_sd = 100
tau_set = [tau_m_e, tau_m_i, tau_ref_e, tau_ref_i, tau_er, tau_ed, tau_ir, tau_id, tau_sr, tau_sd]

# set J
Jee = 80
Jei = -60
Jie = 80
Jii = -60
Jef_12 = 100
Jif_12 = 70
Jef_23 = 25
Jif_23 = 15
J_set_2 = [Jee, Jei, Jie, Jii, Jef_12, Jif_12]
J_set_3 = [Jee, Jei, Jie, Jii, Jef_23, Jif_23]


alpha_rec_e_2 = 0.1
alpha_rec_i_2 = 0.1
alpha_rec_e_3 = 0.2
alpha_rec_i_3 = 0.2
alpha_rec_2= [alpha_rec_e_2, alpha_rec_i_2]
alpha_rec_3 = [alpha_rec_e_3, alpha_rec_i_3]

alpha_forward_12 = 0.05
alpha_forward_23 = 0.1

# recurrent
p_mean_rec_ee = 0.01
p_mean_rec_ie = 0.03
p_mean_rec_ei = 0.04
p_mean_rec_ii = 0.04

# feedforward
p_mean_ef_12 = 0.1
p_mean_ef_23 = 0.05
p_mean_if_12 = 0.05
p_mean_if_23 = 0.05

# set p set
# p_mean_ee, p_mean_ie, p_mean_ei, p_mean_ii
p_mean_ff_set_12 = [p_mean_ef_12, p_mean_if_12, 0, 0]
p_mean_ff_set_23 = [p_mean_ef_23, p_mean_if_23, 0, 0]
p_mean_rec_set_2 = [p_mean_rec_ee, p_mean_rec_ie, p_mean_rec_ei, p_mean_rec_ii]
p_mean_rec_set_3 = [p_mean_rec_ee, p_mean_rec_ie, p_mean_rec_ei, p_mean_rec_ii]

# here is step set
step_size = 0.025
step_window = int(50 /step_size)
total_step = 500
step_set = [step_size, step_window, total_step]

# here is voltage set
VL = -60 # leaky voltage
VT = -50 # exponential voltage eq3 see the paper
delta_T_e = 2 # voltage , also ses the paper, to E
delta_T_i = 0.5 # voltage , also ses the paper, to I
V_threshold = -10 # threshold voltage
Vre = -65 # reset voltage
V_set = [VL, VT, delta_T_e, delta_T_i, V_threshold, Vre]

pf12 = 1.
ps12 = 0.
pf23 = 0.2 # fast
ps23= 0.8 # slow


k_range = 5
k_list = np.linspace(-k_range, k_range, 2*k_range)
rx = 10 # Hz, number of firing event per 1000 ms
mu_set_2 = [0,0]
mu_set_3 = [0, 0.2]



# first define 1 to 2 projection
Wf_12 = compute_projection_EI(Nx_1d, 0, Ne_1d, Ni_1d,alpha_forward_12, alpha_forward_12,  k_list, p_mean_ff_set_12)
print(f'the shape of Wf12 is {np.shape(Wf_12)}')
print('layer 1 to 2 projection finish')
print(time.ctime())

#then generate poisson spikes
poisson_spike_train  = gen_poisson_spike(Nx, total_step, rx, step_size)
print(f'the origin poisson spike train is {np.shape(poisson_spike_train)}')

'''
# let check poisson spike

fig = plt.figure()

ims = []
for t in range(total_step):
    temp_mat = np.mat(np.zeros((Nx_1d, Nx_1d)))
    for j in range(Nx):
        true_x, true_y = compute_xy(j, Nx_1d)
        if poisson_spike_train[j,t] == 1:
            temp_mat[true_x, true_y] = 1
    im = plt.imshow(temp_mat, cmap=plt.cm.Blues)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=200)
ani.save('C:\\Users\\lufen\\Desktop\\check_connection_2.gif')
#ani.save('dynamic_images.gif')
plt.show()'''


poisson_spike_train= np.mat(Wf_12.transpose()) * np.mat(poisson_spike_train)
print(f'the modified poisson spike train is {np.shape(poisson_spike_train)}')
#poisson_spike_train = np.array(poisson_spike_train)


'''
fig = plt.figure()

ims = []
for t in range(total_step):
    temp_mat = np.mat(np.zeros((Ne_1d, Ne_1d)))
    for j in range(Ne_2, Ne_2+Ni_2):
        true_x, true_y = compute_xy(j-Ne_2, Ne_1d)
        temp_mat[true_x, true_y] = poisson_spike_train[j, t]
    im = plt.imshow(temp_mat, cmap=plt.cm.Blues)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=200)
ani.save('C:\\Users\\lufen\\Desktop\\check_connection_2.gif')
#ani.save('dynamic_images.gif')
plt.show()'''

# poisson input checked!


# parameters of EIF_NN
# NN_size_set,  initial_V, Wf_in, Wf_out, tau_set, J_set, alpha_rec, p_mean_rec_set,step_set, V_set, other_NN, mu_set, pf, ps
E_layer_2 = EIF_NN_fast(NN2_size_set, initial_V, Wf_12, 0, tau_set, J_set_2, alpha_rec_2, p_mean_rec_set_2, step_set, V_set, other_NN_2, mu_set_2, pf12, ps12)
E_layer_2.make_recurrent()
#E_layer_2.check_recurrent_connection()
print('recurrent success finish')
print(time.ctime())

a = E_layer_2.show_recurrent_details()
a = np.array(a)

# recurrent checked!

print(' layer set !')
print( time.ctime())


test_steps = total_step-3
for i in range(test_steps):
    if i % 10 == 0:
        print(f'{i} and {time.ctime()}')
    E_layer_2.evolve(poisson_spike_train, [])
    #print(poisson_spike_train)
    #a = poisson_spike_train
    #layer2_spike = E_layer_2.return_inner_spikes()
    #print(np.shape(layer2_spike))

'''
s_mat = E_layer_2.return_inner_spikes()[:, :test_steps]
#plt.matshow(s, cmap=plt.cm.Blues)
plt.matshow(s_mat)
plt.show()
'''

s2 = E_layer_2.return_inner_spikes()


snapshot_set = np.zeros((test_steps, Ne_1d, Ne_1d))
for j in range(test_steps):
    s = s2[:Ne_2, j ]

    for i in range(Ne_2):
        x0, y0 = compute_xy(i, Ne_1d)
        #a0, b0 = xy2len(x0, y0, Ne_1d)
        snapshot_set[j, x0, y0] = s[i]

fig = plt.figure()

ims = []
for j in range(5, test_steps):
    im = plt.imshow(snapshot_set[j], cmap=plt.cm.Blues)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50)
ani.save('C:\\Users\\lufen\\Desktop\\dynamic_images.gif')
#ani.save('dynamic_images.gif')
plt.show()
'''
for j in range(test_steps):
    plt.matshow(snapshot_set[j])
    plt.show()'''