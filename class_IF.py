import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# here are some fucking noisy functions
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


class EIF_NN_fast:

    def __init__(self, NN_size_set, initial_V, Wf_in, Wf_out, tau_set, J_set, alpha_rec, p_mean_rec_set,step_set, V_set, other_NN, mu_set, pf, ps):
        # parameter set contains recurrent alpha, and tau
        # since poisson_spike_train contains Wf_in, we do not need this any more
        self.Wf_in = Wf_in
        self.Wf_out = Wf_out
        # consider E and I neuron separatly
        self.Ne_1d = NN_size_set[0]
        self.Ni_1d = NN_size_set[1]
        self.Ne = self.Ne_1d * self.Ne_1d
        self.Ni = self.Ni_1d * self.Ni_1d
        self.N = self.Ne + self.Ni


        # first E pop, then I pop

        self.step_size = step_set[0]
        self.step_window = step_set[1]
        self.total_step = step_set[2]
        self.step_now = 0

        self.V = [initial_V for _ in range(self.Ne)] + [initial_V for _ in range(self.Ni)]
        self.V_record = np.zeros((self.N, self.total_step))
        self.state_mat = [1 for _ in range(self.Ne)] + [1 for _ in range(self.Ni)] # 0 means in ref peroid
        self.ref_time_mat = [0 for _ in range(self.Ne)] + [0 for _ in range(self.Ni)]


        self.tau_m_e = tau_set[0]
        self.tau_m_i = tau_set[1]
        self.tau_ref_e = tau_set[2]
        self.tau_ref_i = tau_set[3]
        self.tau_er = tau_set[4]
        self.tau_ed = tau_set[5]
        self.tau_ir = tau_set[6]
        self.tau_id = tau_set[7]
        self.tau_sr = tau_set[8]
        self.tau_sd = tau_set[9]

        #they're recurrent J
        self.Jee = J_set[0]
        self.Jei = J_set[1]
        self.Jie = J_set[2]
        self.Jii = J_set[3]
        # they are feedforward J
        self.Jef = J_set[4]
        self.Jif = J_set[5]

        # then load voltage parameters
        self.VL = V_set[0]  # leaky voltage
        self.VT = V_set[1] # exponential voltage eq3 see the paper
        self.delta_T_e = V_set[2]  # voltage , also ses the paper, to E
        self.delta_T_i = V_set[3]  # voltage , also ses the paper, to I
        self.V_threshold = V_set[4]  # threshold voltage
        self.Vre = V_set[5] # reset voltage


        # these are other NN size
        self.Nx = other_NN[0]
        self.N_in_e = other_NN[1]
        self.N_in_i = other_NN[2]
        self.N_out = other_NN[3]

        # these are static current
        self.mu_e= mu_set[0]
        self.mu_i= mu_set[1]

        # these are spatial parameters
        self.alpha_rec_e = alpha_rec[0]
        self.alpha_rec_i = alpha_rec[1]
        self.p_mean_rec_set = p_mean_rec_set

        #these are fast/slow synapse (used in feedforward input)
        self.pf = pf
        self.ps = ps

        # at first, the inner_spike_train is nothing but zero
        self.inner_spike_train = np.zeros((self.N, self.total_step))

    def show_recurrent_details(self):
        return self.Wrr

    def show_absolute_recurrent(self):
        return self.grr

    def return_inner_spikes(self):
        return self.inner_spike_train

    def return_inner_voltage(self):
        return self.V

    def return_PDF(self, step_number):
        return self.V_record[:, step_number]

    def make_recurrent(self):
        # then make recurrent connection
        k_range = 7
        k_list = np.linspace(-k_range, k_range, 2 * k_range)
        # first E pop then I pop
        p_mean_ee = self.p_mean_rec_set[0]
        p_mean_ie = self.p_mean_rec_set[1]
        p_mean_ei = self.p_mean_rec_set[2]
        p_mean_ii = self.p_mean_rec_set[3]

        N_e = self.Ne_1d ** 2
        N_i = self.Ni_1d ** 2

        # N is first with E pop, then I pop

        W_ee = np.zeros((N_e, N_e))
        W_ie = np.zeros((N_e, N_i))
        W_ei = np.zeros((N_i, N_e))
        W_ii = np.zeros((N_i, N_i))

        # g just contain distance
        g_ee = np.zeros((2, N_e, N_e))
        g_ie = np.zeros((2, N_e, N_i))
        g_ei = np.zeros((2, N_i, N_e))
        g_ii = np.zeros((2, N_i, N_i))

        # first compute g_ee
        for i in range(N_e):
            # print(i)
            x0, y0 = compute_xy(i, self.Ne_1d)
            a0, b0 = xy2len(x0, y0, self.Ne_1d)
            for j in range(N_e):
                if i != j:
                    x1, y1 = compute_xy(j, self.Ne_1d)
                    a1, b1 = xy2len(x1, y1, self.Ne_1d)
                    g_ee[0, i, j] = abs(a0 - a1)
                    g_ee[1, i, j] = abs(b0 - b1)

        g_ee = wrapped_gaussian(g_ee, self.alpha_rec_e, k_list)
        g_ee = g_ee[0] * g_ee[1]
        #print('g finish')

        # then g_ie
        for i in range(N_e):
            x0, y0 = compute_xy(i, self.Ne_1d)
            a0, b0 = xy2len(x0, y0, self.Ne_1d)
            for j in range(N_i):
                if i != j:
                    x1, y1 = compute_xy(j, self.Ni_1d)
                    a1, b1 = xy2len(x1, y1, self.Ni_1d)
                    g_ie[0, i, j] = abs(a0 - a1)
                    g_ie[1, i, j] = abs(b0 - b1)
        g_ie = wrapped_gaussian(g_ie, self.alpha_rec_e, k_list)
        g_ie = g_ie[0] * g_ie[1]
        #print('g finish')

        # then g_ei
        for i in range(N_i):
            x0, y0 = compute_xy(i, self.Ni_1d)
            a0, b0 = xy2len(x0, y0, self.Ni_1d)
            for j in range(N_e):
                if i != j:
                    x1, y1 = compute_xy(j, self.Ne_1d)
                    a1, b1 = xy2len(x1, y1, self.Ne_1d)
                    g_ei[0, i, j] = abs(a0 - a1)
                    g_ei[1, i, j] = abs(b0 - b1)
        g_ei = wrapped_gaussian(g_ei, self.alpha_rec_i, k_list)
        g_ei = g_ei[0] * g_ei[1]
        #print('g finish')

        # then g_ii
        for i in range(N_i):
            x0, y0 = compute_xy(i, self.Ni_1d)
            a0, b0 = xy2len(x0, y0, self.Ni_1d)
            for j in range(N_i):
                if i != j:
                    x1, y1 = compute_xy(j, self.Ni_1d)
                    a1, b1 = xy2len(x1, y1, self.Ni_1d)
                    g_ii[0, i, j] = abs(a0 - a1)
                    g_ii[1, i, j] = abs(b0 - b1)
        g_ii = wrapped_gaussian(g_ii, self.alpha_rec_i, k_list)
        g_ii = g_ii[0] * g_ii[1]
        #print('g finish')

        # W_ee
        prob_mat = np.random.uniform(0, 1, (N_e, N_e))
        for i in range(N_e):
            for j in range(N_e):
                prob = p_mean_ee * g_ee[i, j]
                if prob > prob_mat[i, j]:
                    W_ee[i][j] = 1

        # W_ie
        prob_mat = np.random.uniform(0, 1, (N_e, N_i))
        for i in range(N_e):
            for j in range(N_i):
                prob = p_mean_ie * g_ie[i, j]
                if prob > prob_mat[i, j]:
                    W_ie[i][j] = 1

        # W_ei
        prob_mat = np.random.uniform(0, 1, (N_i, N_e))
        for i in range(N_i):
            for j in range(N_e):
                prob = p_mean_ei * g_ei[i, j]
                if prob > prob_mat[i, j]:
                    W_ei[i][j] = 1

        # W_ii
        prob_mat = np.random.uniform(0, 1, (N_i, N_i))
        for i in range(N_i):
            for j in range(N_i):
                prob = p_mean_ii * g_ii[i, j]
                if prob > prob_mat[i, j]:
                    W_ii[i][j] = 1

        temp_W_1 = np.concatenate((W_ee, W_ie), axis=1)
        temp_W_2 = np.concatenate((W_ei, W_ii), axis=1)

        temp_g_1 = np.concatenate((g_ee, g_ie), axis=1)
        temp_g_2 = np.concatenate((g_ei, g_ii), axis=1)

        self.Wrr = np.concatenate((temp_W_1, temp_W_2), axis=0)
        self.grr = np.concatenate((temp_g_1, temp_g_2), axis=0)

        Wrr_trans = self.Wrr.transpose()

        Wrr_trans[:self.Ne, :self.Ne] = self.Jee * Wrr_trans[:self.Ne, :self.Ne] / np.sqrt(self.Ne)
        Wrr_trans[:self.Ne, self.Ne:] = self.Jei * Wrr_trans[:self.Ne, self.Ne:] / np.sqrt(self.Ni)
        Wrr_trans[self.Ne:, :self.Ne] = self.Jie * Wrr_trans[self.Ne:, :self.Ne] / np.sqrt(self.Ne)
        Wrr_trans[self.Ne:, self.Ne:] = self.Jii * Wrr_trans[self.Ne:, self.Ne:] / np.sqrt(self.Ni)

        self.Wrr_trans = Wrr_trans


    def check_recurrent_connection(self):

        def position2what(list_number, neuron_type, grr):
            E_mat = np.mat(np.zeros((self.Ne_1d, self.Ne_1d)))
            I_mat = np.mat(np.zeros((self.Ni_1d, self.Ni_1d)))

            if neuron_type == 'E':
                projection = grr[list_number]
                true_x, true_y = compute_xy(list_number, self.Ne_1d)
                print(f'the position is {true_x, true_y}')

                for i in range(self.Ne):
                    x_temp, y_temp = compute_xy(i, self.Ne_1d)
                    E_mat[x_temp, y_temp] = projection[i]
                    E_mat[true_x, true_y] = 30

                for i in range(self.Ne, self.Ne + self.Ni):
                    x_temp_1, y_temp_1 = compute_xy(i - self.Ne, self.Ni_1d)
                    I_mat[x_temp_1, y_temp_1] = projection[i]
                    # I_mat[true_x, true_y] = 30

            if neuron_type == 'I':
                projection = grr[list_number]
                list_number -= self.Ne
                true_x, true_y = compute_xy(list_number, self.Ni_1d)
                print(f'the position is {true_x, true_y}')

                for i in range(self.Ne):
                    x_temp, y_temp = compute_xy(i, self.Ne_1d)
                    E_mat[x_temp, y_temp] = projection[i]
                    # E_mat[true_x, true_y] = 30

                for i in range(self.Ne, self.Ne + self.Ni):
                    x_temp, y_temp = compute_xy(i - self.Ne, self.Ni_1d)
                    I_mat[x_temp, y_temp] = projection[i]
                    I_mat[true_x, true_y] = 30

            return E_mat, I_mat

        a = self.grr

        fig = plt.figure()
        ims = []
        for j in range(self.Ne, self.Ne+self.Ni):
        #for j in range(self.Ne):
            E_mat_0, I_mat_0 = position2what(j, 'I', a)
            im = plt.imshow(I_mat_0, cmap=plt.cm.Blues)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save('C:\\Users\\lufen\\Desktop\\checkNOW.gif')
        # ani.save('dynamic_images.gif')
        plt.show()



    def evolve(self, poisson_spike_train_origin, ff_spike_origin):
        #print(f'evovle {self.step_now}')

        # first compute recurrent term
        # let's rock!

        if self.step_now > self.step_window:

            receive_spike_mat_e = np.mat(self.Wrr_trans[:, :self.Ne]) * np.mat(self.inner_spike_train[:self.Ne, self.step_now-self.step_window:self.step_now])
            receive_spike_mat_i = np.mat(self.Wrr_trans[:, self.Ne:]) * np.mat(self.inner_spike_train[self.Ne:,self.step_now-self.step_window:self.step_now])

            nabla_mat_e = [(np.exp(-self.step_size*(self.step_window - t)/self.tau_ed) - np.exp(-self.step_size*(self.step_window - t)/self.tau_er))/(self.tau_ed - self.tau_er) for t in range(self.step_window)]
            nabla_mat_e = np.mat(nabla_mat_e)

            nabla_mat_i = [(np.exp(-self.step_size * (self.step_window - t) / self.tau_id) - np.exp(
                -self.step_size * (self.step_window - t) / self.tau_ir)) / (self.tau_id - self.tau_ir) for t in
                           range(self.step_window)]
            nabla_mat_i = np.mat(nabla_mat_i)

            reccurent_term_in_e = receive_spike_mat_e[:] * nabla_mat_e.T
            reccurent_term_in_i = receive_spike_mat_i[:] * nabla_mat_i.T

        else:# if not fit in spike window

            receive_spike_mat_e = np.mat(self.Wrr_trans[:, :self.Ne]) * np.mat(self.inner_spike_train[:self.Ne, :self.step_now+1])
            receive_spike_mat_i = np.mat(self.Wrr_trans[:, self.Ne:]) * np.mat(self.inner_spike_train[self.Ne:, :self.step_now+1])


            nabla_mat_e = [(np.exp(-self.step_size * (self.step_now+1 - t) / self.tau_ed) - np.exp(
                -self.step_size * (self.step_now+1 - t) / self.tau_er)) / (self.tau_ed - self.tau_er) for t in
                           range(self.step_now+1)]
            nabla_mat_e = np.mat(nabla_mat_e)

            nabla_mat_i = [(np.exp(-self.step_size * (self.step_now+1 - t) / self.tau_id) - np.exp(
                -self.step_size * (self.step_now+1 - t) / self.tau_ir)) / (self.tau_id - self.tau_ir) for t in
                           range(self.step_now+1)]
            nabla_mat_i = np.mat(nabla_mat_i)

            #print(np.shape(receive_spike_mat_e[:, -(self.step_now+1):]))
            #print(np.shape(nabla_mat_e.T))
            #print(self.step_now)
            reccurent_term_in_e = receive_spike_mat_e[:] * nabla_mat_e.T
            reccurent_term_in_i = receive_spike_mat_i[:] * nabla_mat_i.T

        reccurent_term  = reccurent_term_in_e + reccurent_term_in_i



        # then  poisson term and feedforward term
        poisson_spike_train = poisson_spike_train_origin.copy()
        ff_spike = ff_spike_origin.copy()
        # then  poisson term and feedforward term
        flag_p = 0
        flag_f = 0
        if len(poisson_spike_train) > 1:
            flag_p = 1
        if len(ff_spike) > 1:
            flag_f = 1
        ff_term = 0
        poisson_term = 0



        if self.step_now > self.step_window:
            # fast and slow components
            nabla_mat = [self.pf * ((np.exp(-self.step_size * (self.step_window - t) / self.tau_ed) - np.exp(
                -self.step_size * (self.step_window - t) / self.tau_er)) / (self.tau_ed - self.tau_er))
                         + self.ps * ((np.exp(-self.step_size * (self.step_window - t) / self.tau_sd) - np.exp(
                -self.step_size * (self.step_window - t) / self.tau_sr)) / (self.tau_sd - self.tau_sr)) for t in
                         range(self.step_window)]
            nabla_mat = np.mat(nabla_mat)

            # if there is poisson input
            if flag_p == 1:
                poisson_spike_train[:self.Ne] = self.Jef * poisson_spike_train[:self.Ne] / np.sqrt(self.Nx)
                poisson_spike_train[self.Ne:] = self.Jif * poisson_spike_train[self.Ne:] / np.sqrt(self.Nx)

                poisson_term = poisson_spike_train[:, self.step_now-self.step_window:self.step_now] * nabla_mat.T
            #if there is feedforward input from another layer
            if flag_f == 1:
                ff_spike[:self.Ne] = self.Jef * ff_spike[:self.Ne] / np.sqrt(self.N_in_e)
                ff_spike[self.Ne:] = self.Jif * ff_spike[self.Ne:] / np.sqrt(self.N_in_e)
                ff_term = ff_spike[:, self.step_now-self.step_window:self.step_now] * nabla_mat.T



        else:
            nabla_mat = [self.pf * ((np.exp(-self.step_size * (self.step_now+1 - t) / self.tau_ed) - np.exp(
                -self.step_size * (self.step_now+1 - t) / self.tau_er)) / (self.tau_ed - self.tau_er))
                         + self.ps * ((np.exp(-self.step_size * (self.step_now+1 - t) / self.tau_sd) - np.exp(
                -self.step_size * (self.step_now+1 - t) / self.tau_sr)) / (self.tau_sd - self.tau_sr)) for t in
                         range(self.step_now+1)]
            nabla_mat = np.mat(nabla_mat)

            # if there is poisson input
            if flag_p == 1:
                #p_spike_train = poisson_spike_train.copy()
                poisson_spike_train[:self.Ne] = self.Jef * poisson_spike_train[:self.Ne] / np.sqrt(self.Nx)
                poisson_spike_train[self.Ne:] = self.Jif * poisson_spike_train[self.Ne:] / np.sqrt(self.Nx)
                poisson_term = poisson_spike_train[:, :self.step_now+1] * nabla_mat.T


            # if there is feedforward input from another layer
            if flag_f == 1:
                ff_spike[:self.Ne] = self.Jef * ff_spike[:self.Ne] / np.sqrt(self.N_in_e)
                ff_spike[self.Ne:] = self.Jif * ff_spike[self.Ne:] / np.sqrt(self.N_in_e)
                ff_term = ff_spike[:, :self.step_now+1] * nabla_mat.T
        #print(poisson_term[0])
        total_ff_term = ff_term + poisson_term
        #print(total_ff_term[0])
        I_Cm = total_ff_term + reccurent_term
        #I_Cm = total_ff_term
        #print(f' I_Cm is {I_Cm[0]}')
        I_Cm[:self.Ne] = I_Cm[:self.Ne] + self.mu_e
        I_Cm[self.Ne:] = I_Cm[self.Ne:] + self.mu_i

        V_now = np.mat(self.V).T
        #print(np.shape(V_now))
        #print(np.shape(self.VT))
        m,n = np.shape(I_Cm)
        delta_V = np.mat(np.zeros((m, n)))
        delta_V[:self.Ne] =  - (V_now[:self.Ne] - self.VL) / self.tau_m_e + self.delta_T_e * np.exp(
                        (V_now[:self.Ne] - self.VT) / self.delta_T_e) / self.tau_m_e + I_Cm[:self.Ne]

        delta_V[self.Ne:] =  - (V_now[self.Ne:] - self.VL) / self.tau_m_i + self.delta_T_i * np.exp(
            (V_now[self.Ne:] - self.VT) / self.delta_T_i) / self.tau_m_i + I_Cm[self.Ne:]

        #print(f'V now is {V_now[0]}')
        #print(f'delta V is {delta_V[0]}')
        V_next = V_now + np.mat(delta_V.T * self.step_size).T
        #print(f'V next is {V_next[0]}')
        #print(delta_V[0])

        # see if there is any neuron thaw
        for i in range(self.N):
            if self.state_mat[i] == 0:
                self.ref_time_mat[i] -= self.step_size
                if abs(self.ref_time_mat[i]) < 1e-2:
                    self.state_mat[i] = 1


        for i in range(self.N):
            if self.state_mat[i] == 1: # it's not in ref peroid
                if V_next[i] > self.V_threshold:
                    self.inner_spike_train[i, self.step_now] = 1 # then there is a spike
                    V_next[i] = self.Vre
                    self.state_mat[i] = 0
                    if i <= self.Ne - 1:
                        self.ref_time_mat[i] = self.tau_ref_e
                    else:
                        self.ref_time_mat[i] = self.tau_ref_i
            else:
                V_next[i] = self.V.T[i]


        #print(V_next[0])
        self.V  = np.mat(V_next).T
        self.V_record[:, self.step_now] = self.V
        self.step_now += 1
























