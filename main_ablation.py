# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:57:02 2019

@author: 12440855
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
from past.utils import old_div
import pickle
import freqopttest.util as util
import freqopttest.data as data
import freqopttest.kernel as kernel
import freqopttest.tst as tst
import freqopttest.glo as glo
from scipy.stats import norm
from sklearn.utils import check_random_state
from TST_utils import get_item, MatConvert, MMDu_linear_kernel, TST_MMD_u_linear_kernel,Pdist2, MMDu, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_SCF,C2ST_NN_fit,TST_LCE

class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""

    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),  # +1 for high test power
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            # torch.nn.Linear(H, H, bias=True),
            # torch.nn.ReLU(),
            torch.nn.Linear(H, x_out, bias=True),
            torch.nn.Linear(x_out, 1, bias=True),
            # torch.nn.Softmax(),
        )

    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)

def sample_blobs(n, rows=3, cols=3, sep=1, rs=None):
    rs = check_random_state(rs)
    # ratio is eigenvalue ratio
    # correlation = (ratio - 1) / (ratio + 1)
    correlation = 0

    # generate within-blob variation
    mu = np.zeros(2)
    sigma = np.eye(2)
    X = rs.multivariate_normal(mu, sigma, size=n)

    corr_sigma = np.array([[1, correlation], [correlation, 1]])
    Y = rs.multivariate_normal(mu, corr_sigma, size=n)

    # assign to blobs
    X[:, 0] += rs.randint(rows, size=n) * sep
    X[:, 1] += rs.randint(cols, size=n) * sep
    Y[:, 0] += rs.randint(rows, size=n) * sep
    Y[:, 1] += rs.randint(cols, size=n) * sep

    return X, Y

def sample_blobs_Q(N1, sigma_mx_2, rows=3, cols=3, rs=None):
    rs = check_random_state(rs)
    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    X = rs.multivariate_normal(mu, sigma, size=N1)

    Y = rs.multivariate_normal(mu, np.eye(2), size=N1)

    # assign to blobs
    X[:, 0] += rs.randint(rows, size=N1)
    X[:, 1] += rs.randint(cols, size=N1)

    Y_row = rs.randint(rows, size=N1)
    Y_col = rs.randint(cols, size=N1)
    locs = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        Y = np.where(ind2, np.matmul(Y,L) + locs[i], Y)
    return X, Y

# Data generation - blob
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True
is_cuda = True

dtype = torch.float
device = torch.device("cuda:0")
N_per = 100 # permutation times
alpha = 0.05 # test threshold
# n_list = [10,20,40,60,70,80,90,100]
n_list = [40]
# n = 35
x_in = 2
H = 50  # 3 for lower type-I error and test power
x_out = 50
# learning_rate = 0.1 #SGD
learning_rate = 0.005#0.0005 #Adam
learning_ratea = 0.00005
learning_rate_C2ST = 0.0005  # 0.0005 for n < 90
K = 10
Results = np.zeros([6,K])
reg = 0 * 0.2**2
N = 100
N_f = 100.0
sigma = 0.3 # 1.5
sigma0_u = 0.002  # 0.1 0.01



mu_mx = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
sigma_mx_1 = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2 = np.zeros([9,2,2])
J_star_u = np.zeros([len(n_list),1000])
J_star_adp = np.zeros([len(n_list),1000])
count = 0
ep = 0.000000001
N_epoch1 = 1000
for i in range(9):
    sigma_mx_2[i] = sigma_mx_2_standard
    if i < 4:
        sigma_mx_2[i][0 ,1] = -0.02 - 0.002 * i
        sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
    if i==4:
        sigma_mx_2[i][0, 1] = 0.00
        sigma_mx_2[i][1, 0] = 0.00
    if i>4:
        sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i-5)
        sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i-5)
for n in n_list:
    s1 = np.zeros([9 * n, 2])
    s2 = np.zeros([9 * n, 2])
    N1 = 9 * n
    N2 = 9 * n
    batch_size = min(n*2,128)
    N_epoch = int(500*18*n/batch_size)
    for kk in range(K):
        if is_cuda:
            model_u = ModelLatentF(x_in, H, x_out).cuda()
            model_u1 = ModelLatentF(x_in, H, x_out).cuda()
        else:
            model_u = ModelLatentF(x_in, H, x_out)
            model_u1 = ModelLatentF(x_in, H, x_out)
        optimizer_u = torch.optim.Adam(list(model_u.parameters()), lr=learning_rate)
        optimizer_u1 = torch.optim.Adam(list(model_u1.parameters()), lr=learning_rate)

        np.random.seed(seed=112 * kk + 1 + n)
        s1, s2 = sample_blobs_Q(N1, sigma_mx_2)
        if kk==0:
            s1_o = s1
            s2_o = s2
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        # C2ST
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
        pred, STAT_C2ST, model_C2ST, w_C2ST, b_C2ST = C2ST_NN_fit(S,y,N1,x_in,H,x_out,learning_rate_C2ST,N_epoch,batch_size,device,dtype)

        # LM = MMD_L(N1, N2, device, dtype)
        # v = torch.div(torch.ones([N1+N2, N1+N2], dtype=torch.float, device=device), (N1+N2)*1.0)

        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        for t in range(N_epoch1):
            modelu1_output = model_u1(S)
            TEMP1 = MMDu(modelu1_output, N1, S, sigma, sigma0_u, is_smooth=False)
            mmd_value_temp = -1 * (TEMP1[0] + 10 ** (-8))  # 10**(-8)
            mmd_std_temp = torch.sqrt(TEMP1[1] + 10 ** (-8))  # 0.1
            if mmd_std_temp.item() == 0:
                print('error!!')
            if np.isnan(mmd_std_temp.item()):
                print('error!!')
            STAT_u1 = torch.div(mmd_value_temp, mmd_std_temp)  # - r_full / (N1+N2)
            # J_star_u[t] = STAT_u1.item()
            optimizer_u1.zero_grad()
            STAT_u1.backward(retain_graph=True)
            # Update weights using gradient descent
            optimizer_u1.step()
            if t % 100 == 0:
                print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                      -1 * STAT_u1.item())  # ,"Reg: ", loss1.item()

        h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, N1, S, sigma, sigma0_u, alpha, device, dtype, ep, is_smooth=False)
        print("h:", h_u1, "Threshold:", threshold_u1, "MMD_value:", mmd_value_u1)  # G+J

        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        # MoS = model_u(S)
        # Dxy = Pdist2(MoS[:N1,:],MoS[N1:,:])
        # sigma0_u = get_item(Dxy.view(-1,1).squeeze().kthvalue(int(Dxy.size(0)*Dxy.size(1)*0.005))[0],is_cuda).tolist()
        # print(sigma0_u)
        for t in range(N_epoch1):
            modelu_output = model_u(S)
            TEMP = MMDu_linear_kernel(modelu_output, N1)
            mmd_value_temp = -1 * (TEMP[0] + 10 ** (-8))  # 10**(-8)
            mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))  # 0.1
            if mmd_std_temp.item() == 0:
                print('error!!')
            if np.isnan(mmd_std_temp.item()):
                print('error!!')
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)  # - r_full / (N1+N2)
            # J_star_u[t] = STAT_u.item()
            optimizer_u.zero_grad()
            STAT_u.backward(retain_graph=True)
            # Update weights using gradient descent
            optimizer_u.step()
            if t % 100 == 0:
                print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                      -1 * STAT_u.item())  # ,"Reg: ", loss1.item()

        h_u, threshold_u, mmd_value_u = TST_MMD_u_linear_kernel(model_u(S), N_per, N1, alpha, device, dtype)
        print("h:", h_u, "Threshold:", threshold_u, "MMD_value:", mmd_value_u)  # L+J
        #
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        S_m = model_C2ST(S)
        Dxy_m = Pdist2(S_m[:N1, :], S_m[N1:, :])
        # Dxy = Pdist2(S[:N1, :], S[N1:, :])
        # sigma0 = Dxy.median() * (2 ** (-4))
        sigma0 = get_item(Dxy_m.median() * (2 ** (-4)), is_cuda)
        sigma0 = torch.from_numpy(sigma0).to(device, dtype)
        sigma0.requires_grad = True
        # sigma0 = 2*d * torch.rand([1]).to(device, dtype)# d * torch.rand([1]).to(device, dtype)
        # sigma0.requires_grad = True
        optimizer_sigma0 = torch.optim.Adam([sigma0], lr=learning_ratea)
        for t in range(N_epoch1):
            TEMPa = MMDu(S_m, N1, S_m, sigma, sigma0, ep, is_smooth=False)
            mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
            mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
            if mmd_std_tempa.item() == 0:
                print('std error!!')
            if np.isnan(mmd_std_tempa.item()):
                print('std error!!')
            STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
            # J_star_adp[t] = STAT_adaptive.item()
            optimizer_sigma0.zero_grad()
            STAT_adaptive.backward(retain_graph=True)
            # Update sigma0 using gradient descent
            optimizer_sigma0.step()
            if t % 100 == 0:
                print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                      -1 * STAT_adaptive.item())
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S_m, N_per, N1, S_m, sigma,
                                                                                        sigma0, alpha, device, dtype)
        print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)  # G+C

        # np.random.seed(seed=1102)
        # torch.manual_seed(1102)
        # torch.cuda.manual_seed(1102)
        #
        # f = torch.nn.Softmax()

        # for t in range(1000):
        #     modelu_output = model_u(S)
        #     TEMP = MMDu_linear_kernel(modelu_output, N1)
        #     mmd_value_temp = -1 * TEMP[0]
        #     mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
        #     STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        #     J_star_u[count, t] = STAT_u.item()
        #     optimizer_u.zero_grad()
        #     STAT_u.backward(retain_graph=True)
        #     # Update weights using gradient descent
        #     optimizer_u.step()
        #     if t % 100 == 0:
        #         print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
        #               -1 * STAT_u.item())  # ,"Reg: ", loss1.item()
        # h_u, threshold_u, mmd_value_u = TST_MMD_u_linear_kernel(model_u(S), N_per, N1, alpha, device,
        #                                           dtype)
        # print("h:", h_u, "Threshold:", threshold_u, "MMD_value:", mmd_value_u)

        # np.random.seed(seed=1102)
        # torch.manual_seed(1102)
        # torch.cuda.manual_seed(1102)
        # Dxy = Pdist2(S[:N1,:],S[N1:,:])
        # sigma0 = Dxy.median() * (2**(-3))
        # print(sigma0)
        # sigma0.requires_grad = True
        # optimizer_sigma0 = torch.optim.Adam([sigma0], lr=0.0005)
        # for t in range(1000):
        #     TEMPa = MMDu(S, N1, S, sigma, sigma0, is_smooth=False)
        #     mmd_value_tempa = -1 * TEMPa[0]
        #     mmd_std_tempa = torch.sqrt(TEMPa[1])
        #     STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        #     J_star_adp[count, t] = STAT_adaptive.item()
        #     # print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ", -1 * STAT_adaptive.item())
        #     optimizer_sigma0.zero_grad()
        #     STAT_adaptive.backward(retain_graph=True)
        #     # Update sigma0 using gradient descent
        #     optimizer_sigma0.step()
        #     if t % 100 == 0:
        #         print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
        #               -1 * STAT_adaptive.item())  # ,"Reg: ", loss1.item()
        # h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, N1, S, sigma, sigma0, alpha, device, dtype)
        # print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)

        # np.random.seed(seed=1102)
        # test_locs_ME, gwidth_ME = TST_ME(S, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=5, seed=15)
        # h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=5, seed=15)
        # print("h:", h_ME, "test_locs_ME:", test_locs_ME, "gwidth_ME:", gwidth_ME)
        #
        # np.random.seed(seed=1102)
        # test_freqs_SCF, gwidth_SCF = TST_SCF(S, N1, alpha, is_train=True, test_freqs=1, gwidth=1, J=5, seed=15)
        # h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=5, seed=15)
        # print("h:", h_SCF, "test_freqs_SCF:", test_freqs_SCF, "gwidth_SCF:", gwidth_SCF)
        # S_m = get_item(model_u(S),is_cuda)
        # s1_m = S_m[0:9*n, :]
        # s2_m = S_m[9*n:, :]

        N = 100
        N_f = 100.0
        H_u = np.zeros(N)
        T_u = np.zeros(N)
        M_u = np.zeros(N)
        H_u1 = np.zeros(N)
        T_u1 = np.zeros(N)
        M_u1 = np.zeros(N)
        H_adaptive = np.zeros(N)
        T_adaptive = np.zeros(N)
        M_adaptive = np.zeros(N)
        H_C2ST = np.zeros(N)
        Tu_C2ST = np.zeros(N)
        Tl_C2ST = np.zeros(N)
        S_C2ST = np.zeros(N)
        np.random.seed(1102)
        count_u = 0
        count_adp = 0
        count_u1 = 0
        count_C2ST = 0
        for k in range(N):
            np.random.seed(seed=11 * k + 10 + n)
            s1, s2 = sample_blobs_Q(N1, sigma_mx_2)
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, device, dtype)

            h_u, threshold_u, mmd_value_u = TST_MMD_u_linear_kernel(model_u(S), N_per, N1, alpha, device, dtype)  # L+J
            h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, N1, S, sigma, sigma0_u, alpha, device,dtype, ep, is_smooth=False)  # G+J
            h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S_m, N_per, N1, S_m, sigma,
                                                                                            sigma0, alpha, device,
                                                                                            dtype)  # G+C
            H_C2ST[k], Tu_C2ST[k], S_C2ST[k] = TST_LCE(S, N1, N_per, alpha, model_C2ST, w_C2ST, b_C2ST, device,
                                                       dtype)  # L+C
            count_u = count_u + h_u
            count_adp = count_adp + h_adaptive
            count_u1 = count_u1 + h_u1
            count_C2ST = count_C2ST + int(H_C2ST[k])
            print("L+J:", count_u, "G+J:", count_u1, "G+C:", count_adp, "L+C:", count_C2ST)
            H_u[k] = h_u
            T_u[k] = threshold_u
            M_u[k] = mmd_value_u
            H_u1[k] = h_u1
            T_u1[k] = threshold_u1
            M_u1[k] = mmd_value_u1
            H_adaptive[k] = h_adaptive
            T_adaptive[k] = threshold_adaptive
            M_adaptive[k] = mmd_value_adaptive
        print("Reject rate_LJ: ", H_u.sum() / N_f, "Reject rate_GJ: ", H_u1.sum() / N_f, "Reject rate_GC:",
              H_adaptive.sum() / N_f,
              "Reject rate_LC: ", H_C2ST.sum() / N_f)
        Results[0, kk] = H_u.sum() / N_f
        Results[1, kk] = H_u1.sum() / N_f
        Results[2, kk] = H_adaptive.sum() / N_f
        Results[3, kk] = H_C2ST.sum() / N_f
        print(Results, Results.mean(1))
    count = count + 1
    f = open('Results_'+str(n)+'_H1_abl.pckl_C2ST', 'wb')
    pickle.dump([Results,J_star_u,J_star_adp], f)
    f.close()
