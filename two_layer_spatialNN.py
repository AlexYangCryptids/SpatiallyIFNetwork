import numpy as np
import matplotlib.pyplot as plt
from class_IF import EIF_NN_fast
import time
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter


def compute_xy(i, length):
    x = i % length
    y = int(i / length)
    return x, y

def xy2len(x, y, length):
    unit = 1/length
    return (x+0.5)*unit, (y+0.5)*unit

def list_to_mat(pre_neuron_number, pre_neuron_type, projection_list_mat, Ne_post_1d, Ni_post_1d):
    E_mat = np.mat(np.zeros((Ne_post_1d, Ne_post_1d)))
    I_mat = np.mat(np.zeros((Ni_post_1d, Ni_post_1d)))
    Ne_post = Ne_post_1d ** 2
    Ni_post = Ni_post_1d ** 2

    projection = projection_list_mat[pre_neuron_number]

    for i in range(Ne_post):
        x_temp, y_temp = compute_xy(i, Ne_post_1d)
        E_mat[x_temp, y_temp] = projection[i]

    for j in range(Ne_post, Ne_post + Ni_post):
        x_temp_1, y_temp_1 = compute_xy(j - Ne_post, Ni_post_1d)
        I_mat[x_temp_1, y_temp_1] = projection[j]

    return E_mat, I_mat

def wrapped_gaussian(x_mat, sigma, k_list):

    g = 0
    for k in k_list:
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

            x1, y1 = compute_xy(j, N2_1d_e)
            a1, b1 = xy2len(x1, y1, N2_1d_e)
            distance_a = abs(a0 - a1)
            distance_b = abs(b0 - b1)
            g_ee[0, i, j] = distance_a
            g_ee[1, i, j] = distance_b


    g_ee = wrapped_gaussian(g_ee, alpha_e, k_list)
    g_ee = g_ee[0] * g_ee[1]
    #plt.matshow(list_to_mat(1, 'E', g_ee, N2_1d_e, 0)[0])
    #plt.show()
    #print('g finish')

    # then g_ie
    for i in range(N1_e):
        x0, y0 = compute_xy(i, N1_1d_e)
        a0, b0 = xy2len(x0, y0, N1_1d_e)
        for j in range(N2_i):
            x1, y1 = compute_xy(j, N2_1d_i)
            a1, b1 = xy2len(x1, y1, N2_1d_i)
            distance_a = abs(a0 - a1)
            distance_b = abs(b0 - b1)
            g_ie[0, i, j] = distance_a
            g_ie[1, i, j] = distance_b

    g_ie = wrapped_gaussian(g_ie, alpha_e, k_list)
    g_ie = g_ie[0] * g_ie[1]

    #print('g finish')

    # then g_ei
    for i in range(N1_i):
        x0, y0 = compute_xy(i, N1_1d_i)
        a0, b0 = xy2len(x0, y0, N1_1d_i)
        for j in range(N2_e):

            x1, y1 = compute_xy(j, N2_1d_e)
            a1, b1 = xy2len(x1, y1, N2_1d_e)
            distance_a = abs(a0 - a1)
            distance_b = abs(b0 - b1)
            g_ei[0, i, j] = distance_a
            g_ei[1, i, j] = distance_b

    g_ei = wrapped_gaussian(g_ei, alpha_i, k_list)
    g_ei = g_ei[0] * g_ei[1]
    #print('g finish')

    # then g_ii
    for i in range(N1_i):
        x0, y0 = compute_xy(i, N1_1d_i)
        a0, b0 = xy2len(x0, y0, N1_1d_i)
        for j in range(N2_i):

            x1, y1 = compute_xy(j, N2_1d_i)
            a1, b1 = xy2len(x1, y1, N2_1d_i)
            distance_a = abs(a0 - a1)
            distance_b = abs(b0 - b1)
            g_ii[0, i, j] = distance_a
            g_ii[1, i, j] = distance_b

    g_ii = wrapped_gaussian(g_ii, alpha_i, k_list)
    g_ii = g_ii[0] * g_ii[1]
    #print('g finish')

    # W_ee
    prob_mat = np.random.uniform(0, 1, (N1_e, N2_e))
    for i in range(N1_e):
        for j in range(N2_e):
            prob = p_mean_ee * g_ee[i,j]
            if prob > prob_mat[i, j]:
                W_ee[i][j] = jee

    # W_ie
    prob_mat = np.random.uniform(0, 1, (N1_e, N2_i))
    for i in range(N1_e):
        for j in range(N2_i):
            prob = p_mean_ie * g_ie[i, j]
            if prob > prob_mat[i, j]:
                W_ie[i][j] = jie

    # W_ei
    prob_mat = np.random.uniform(0, 1, (N1_i, N2_e))
    for i in range(N1_i):
        for j in range(N2_e):
            prob = p_mean_ei * g_ei[i, j]
            if prob > prob_mat[i, j]:
                W_ei[i][j] = jei

    # W_ii
    prob_mat = np.random.uniform(0, 1, (N1_i, N2_i))
    for i in range(N1_i):
        for j in range(N2_i):
            prob = p_mean_ii * g_ii[i, j]
            if prob > prob_mat[i, j]:
                W_ii[i][j] = jii

    temp_W_1 = np.concatenate((W_ee, W_ie), axis=1)
    temp_W_2 = np.concatenate((W_ei, W_ii), axis=1)
    #print(np.shape(temp_W_1))
    #print(np.shape(temp_W_2))
    W = np.concatenate((temp_W_1, temp_W_2), axis=0)

    return W


def check_projection_connection(pre, post, W_in, Ne_pre_1d, Ni_pre_1d, Ne_post_1d, Ni_post_1d):

    Ne_pre = Ne_pre_1d ** 2
    Ni_pre = Ni_pre_1d ** 2
    Ne_post = Ne_post_1d ** 2
    Ni_post = Ni_post_1d ** 2
    a = W_in

    temp_map_EE= 0
    temp_map_IE = 0
    temp_map_EI = 0
    temp_map_II = 0

    fig = plt.figure()
    ims = []
    if pre == 'E':
        for j in range(Ne_pre):
            E_mat_0, I_mat_0 = list_to_mat(j, 'E', a, Ne_post_1d, Ni_post_1d)
            if post == 'E':
                im = plt.imshow(E_mat_0, cmap=plt.cm.Blues)
                temp_map_EE += E_mat_0
            elif post == 'I':
                im = plt.imshow(I_mat_0, cmap=plt.cm.Blues)
                temp_map_IE += I_mat_0
            ims.append([im])


    if pre == 'I':
        for j in range(Ne_pre, Ne_pre + Ni_pre):
            E_mat_0, I_mat_0 = list_to_mat(j, 'I', a, Ne_post_1d, Ni_post_1d)
            if post == 'E':
                im = plt.imshow(E_mat_0, cmap=plt.cm.Blues)
                temp_map_EI += E_mat_0
            elif post == 'I':
                im = plt.imshow(I_mat_0, cmap=plt.cm.Blues)
                temp_map_II += I_mat_0
            ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100)

    if pre == 'E' and post == 'E':
        plt.matshow(temp_map_EE)
    if pre == 'E' and post == 'I':
        plt.matshow(temp_map_IE)
    if pre == 'I' and post == 'E':
        plt.matshow(temp_map_EI)
    if pre == 'I' and post == 'I':
        plt.matshow(temp_map_II)
    plt.colorbar()
    plt.show()


def gen_poisson_spike(Nx, total_step, rx, step_size):
    poisson_spike_train = np.zeros((Nx, total_step))
    prob_delta_t = rx * 0.001 * step_size
    for step in range(total_step):
        for n in range(Nx):
            if prob_delta_t > np.random.uniform(0, 1):
                poisson_spike_train[n, step] = 1
    return poisson_spike_train

def show_mat_dynamics(time_mat, mode):
    time_bin_size = int(1 / step_size)  # thus i frames could cover the behavior in 1 ms
    mseconds = int(total_step / time_bin_size)
    snapshot_set = np.zeros((mseconds, Ne_1d, Ne_1d))

    snapshot_temp = np.zeros((Ne_1d, Ne_1d))

    def update_matrix(j, mode_here):
        snapshot_temp = np.zeros((Ne_1d, Ne_1d))
        if mode_here == 'ms':
            step_here = j * time_bin_size
            s = time_mat[:Ne_2, step_here]
            for t in range(1, time_bin_size):
                s += time_mat[:Ne_2, step_here + t]
        else:
            step_here = j
            s = time_mat[:Ne_2, step_here]

        for i in range(Ne_2):
            x0, y0 = compute_xy(i, Ne_1d)
            # a0, b0 = xy2len(x0, y0, Ne_1d)
            snapshot_temp[x0, y0] = s[i]
        matrice.set_array(snapshot_temp)

    fig1, ax1 = plt.subplots()
    if mode == 'ms':
        matrice = ax1.matshow(snapshot_temp)
        plt.colorbar(matrice)
        ani = animation.FuncAnimation(fig1, update_matrix, frames=mseconds, interval=200, fargs=(mode,))
        plt.show()
    else:
        matrice = ax1.matshow(snapshot_temp, vmin=-65, vmax=-35, cmap='rainbow')
        plt.colorbar(matrice)
        ani = animation.FuncAnimation(fig1, update_matrix, frames=total_step, interval=25, fargs=(mode,))
        plt.show()

    #ani.save('C:\\Users\\lufen\\Desktop\\firing_images.gif', writer=PillowWriter(fps=100))



# the input parameters are Ne_1d, initial_V, Wf_in, Wf_out, tau_set, J_set, alpha_rec, p_mean_rec, step_set, V_set, other_NN

Ne_1d = 40
Ni_1d = 40
Nx_1d = 15
Nx = Nx_1d ** 2
Ne_2 = Ne_1d ** 2
Ni_2 = Ni_1d ** 2
total_N_2 = Ne_2 + Ni_2


other_NN_2 = [Nx, 0, 0, 0]
NN2_size_set = [Ne_1d, Ni_1d]

initial_V = -60

# set time constant, ms
tau_m_e = 15
tau_m_i = 10
tau_ref_e = 1.5
tau_ref_i = 0.5
tau_er = 1
tau_ed = 4
tau_ir = 1
tau_id = 8
tau_sr = 2
tau_sd = 100
tau_set = [tau_m_e, tau_m_i, tau_ref_e, tau_ref_i, tau_er, tau_ed, tau_ir, tau_id, tau_sr, tau_sd]

# set J
Jee = 50
Jei = -50
Jie = 50
Jii = -50
Jef_12 = 50
Jif_12 = 50

# the true connection strength should be scaled by sqrt(N)
#j_set_2 = [Jee/(np.sqrt(total_N_2)), Jei/(np.sqrt(total_N_2)), Jie/(np.sqrt(total_N_2)),Jii/(np.sqrt(total_N_2)), Jef_12/(np.sqrt(total_N_2)), Jif_12/(np.sqrt(total_N_2))]
jee = 0.2
jei = -0.2
jie = 0.2
jii = -0.2
jef_12 = 4
jif_12 = 4
j_set_2 = [jee, jei, jie, jii, jef_12, jif_12]


alpha_rec_e_2 = 0.4
alpha_rec_i_2 = 0.4
alpha_rec_2= [alpha_rec_e_2, alpha_rec_i_2]


alpha_forward_12 = 0.05

# recurrent
p_mean_rec_ee = 0.2
p_mean_rec_ie = 0.2
p_mean_rec_ei = 0.2
p_mean_rec_ii = 0.2

# feedforward
p_mean_ef_12 = 0.03
p_mean_if_12 = 0.03

# set p set
# p_mean_ee, p_mean_ie, p_mean_ei, p_mean_ii
p_mean_ff_set_12 = [p_mean_ef_12, p_mean_if_12, 0, 0]
p_mean_rec_set_2 = [p_mean_rec_ee, p_mean_rec_ie, p_mean_rec_ei, p_mean_rec_ii]

# here is step set
step_size = 0.05
step_window = int(50 /step_size) # after 50ms, the eta function goes to zero
total_step = 1500
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


k_range = 5
k_list = []
for i in range(2*k_range + 1):
    k_list.append(-k_range+i)
rx = 50# Hz, number of firing event per 1000 ms
mu_set_2 = [0,0]

print(f'jee = {jee},jei = {jei},jie = {jie},jii = {jii},'
      f'jef = {jef_12}, jif = {jif_12}')

# first define 1 to 2 projection
Wf_12 = compute_projection_EI(Nx_1d, 0, Ne_1d, Ni_1d, alpha_forward_12, alpha_forward_12,  k_list, p_mean_ff_set_12)
print(f'the shape of Wf12 is {np.shape(Wf_12)}')
print('layer 1 to 2 projection finish')
print(time.ctime())
# check poisson connection
#check_projection_connection('E', 'I', Wf_12, Nx_1d, 0, Ne_1d, Ni_1d)


#then generate poisson spikes
poisson_spike_train  = gen_poisson_spike(Nx, total_step, rx, step_size)
print(f'the origin poisson spike train is {np.shape(poisson_spike_train)}')


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
E_layer_2 = EIF_NN_fast(NN2_size_set, initial_V, Wf_12, 0, tau_set, j_set_2, alpha_rec_2, p_mean_rec_set_2, step_set, V_set, other_NN_2, mu_set_2, pf12, ps12)
E_layer_2.make_recurrent()
#E_layer_2.check_recurrent_connection('E', 'E')
print('recurrent success finish')
print(time.ctime())

d = E_layer_2.show_recurrent_details()
d = np.array(d)

# recurrent checked!

print(' layer set !')
print( time.ctime())



for i in range(total_step):
    if i % 10 == 0:
        print(f'{i} and {time.ctime()}')
    E_layer_2.evolve(poisson_spike_train, [])
    #print(poisson_spike_train)
    #a = poisson_spike_train
    #layer2_spike = E_layer_2.return_inner_spikes()
    #print(np.shape(layer2_spike))


# show all spike trains in the time axe
s_mat = E_layer_2.return_inner_spikes()
plt.matshow(s_mat, cmap=plt.cm.Blues, aspect='auto')
plt.show()

# show spike map and voltage map with spatial order

show_mat_dynamics(s_mat, 'ms')
volatge_mat = E_layer_2.return_all_time_voltage()
print('hey!')
show_mat_dynamics(volatge_mat, 'step')

time_bin_size = int(1 / step_size)  # thus i frame could cover the behavior in 1 ms
mseconds = int(total_step / time_bin_size)

# show pdf
V_bin_number = 40
def animate(j):
    x, y = np.linspace(Vre-1, V_threshold+1, V_bin_number), E_layer_2.return_pdf(j*time_bin_size, time_bin_size, V_bin_number, 'EI')
    line.set_data((x,y))
    ax.set_xlim(Vre-1, V_threshold+1)
    ax.set_ylim(0, 1)
    return line

fig, ax = plt.subplots()
line,  = ax.plot([], [], lw=2)
anim = FuncAnimation(fig, animate, mseconds)
plt.show()

#anim.save('C:\\Users\\lufen\\Desktop\\PDF_images.gif', writer=PillowWriter(fps=100))

