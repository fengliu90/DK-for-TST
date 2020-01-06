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
from TST_utils import MatConvert, Pdist2, MMDu, MMD_L, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_SCF, TST_C2ST, C2ST_NN_fit

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
            torch.nn.Linear(H, x_out, bias=True),
        )

    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)

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
# n_list = [10,20,40,50,70,80,90,100]
n_list = [30]
x_in = 2
H = 50
x_out = 50
# learning_rate = 0.1 #SGD
learning_rate = 0.0005
learning_rate_C2ST = 0.0005
K = 10
Results = np.zeros([6,K])
Opt_ep = np.zeros([1,K])
reg = 0 * 0.2**2
N = 100
N_f = 100.0
sigma = 0.3 # 0.3 # 10 #1.5
sigma0_u = 0.002# 0.002#0.05 #0.01   # 0.002 0.1 0.01



mu_mx = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
sigma_mx_1 = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2 = np.zeros([9,2,2])
J_star_u = np.zeros([K,1000])
J_star_adp = np.zeros([K,1000])
count = 0

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
    np.random.seed(1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    s1 = np.zeros([9 * n, 2])
    s2 = np.zeros([9 * n, 2])
    N1 = 9 * n
    N2 = 9 * n
    batch_size = min(n*2,128)
    N_epoch = int(500*18*n/batch_size)
    for kk in range(K):
        if is_cuda:
            model_u = ModelLatentF(x_in, H, x_out).cuda()
        else:
            model_u = ModelLatentF(x_in, H, x_out)
        epsilonOPT = MatConvert(np.random.rand(1) * (-10), device, dtype)
        epsilonOPT.requires_grad = False
        print(epsilonOPT.item())
        optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT], lr=learning_rate) #
        # optimizer_eOPT = torch.optim.Adam([epsilonOPT], lr=learning_rate)

        # for i in range(9):
        #     np.random.seed(seed=1102*kk + i + n)
        #     s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        # for i in range(9):
        #     np.random.seed(seed=819*kk+1 + i + n)
        #     s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n) # sigma_mx_2[i]
        # np.random.seed(seed=1102 * kk + 1 + n)
        np.random.seed(seed=112 * kk + 1 + n)
        s1,s2 = sample_blobs_Q(N1, sigma_mx_2)
        if kk==0:
            s1_o = s1
            s2_o = s2
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        # C2ST
        # np.random.seed(seed=1102)
        # torch.manual_seed(1102)
        # torch.cuda.manual_seed(1102)
        y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
        pred, STAT_C2ST, model_C2ST, w_C2ST, b_C2ST = C2ST_NN_fit(S,y,N1,x_in,H,x_out,learning_rate_C2ST,1,batch_size,device,dtype)


        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        for t in range(1000):
            modelu_output = model_u(S)
            ep = epsilonOPT#torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
            mmd_value_temp = -1 * TEMP[0]
            mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
            J_star_u[kk, t] = STAT_u.item()
            optimizer_u.zero_grad()
            STAT_u.backward(retain_graph=True)
            # Update weights using gradient descent
            optimizer_u.step()
            if t % 100 == 0:
                print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                      -1 * STAT_u.item())  # ,"Reg: ", loss1.item()
        h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, alpha, device,
                                                  dtype, ep)
        print(ep, epsilonOPT)

        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        Dxy = Pdist2(S[:N1,:],S[N1:,:])
        sigma0 = Dxy.median() * 0.5
        print(sigma0)
        sigma0.requires_grad = True
        optimizer_sigma0 = torch.optim.Adam([sigma0], lr=0.0005)
        for t in range(1000):
            TEMPa = MMDu(S, N1, S, sigma, sigma0, is_smooth=False)
            mmd_value_tempa = -1 * TEMPa[0]
            mmd_std_tempa = torch.sqrt(TEMPa[1]+10**(-8))
            STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
            J_star_adp[kk, t] = STAT_adaptive.item()
            # print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ", -1 * STAT_adaptive.item())
            optimizer_sigma0.zero_grad()
            STAT_adaptive.backward(retain_graph=True)
            # Update sigma0 using gradient descent
            optimizer_sigma0.step()
            if t % 100 == 0:
                print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                      -1 * STAT_adaptive.item())  # ,"Reg: ", loss1.item()
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, N1, S, sigma, sigma0, alpha, device, dtype)
        # print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)

        np.random.seed(seed=1102)
        test_locs_ME, gwidth_ME = TST_ME(S, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=5, seed=15)
        h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=5, seed=15)
        print("h:", h_ME, "test_locs_ME:", test_locs_ME, "gwidth_ME:", gwidth_ME)

        np.random.seed(seed=1102)
        test_freqs_SCF, gwidth_SCF = TST_SCF(S, N1, alpha, is_train=True, test_freqs=1, gwidth=1, J=5, seed=15)
        h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=5, seed=15)
        print("h:", h_SCF, "test_freqs_SCF:", test_freqs_SCF, "gwidth_SCF:", gwidth_SCF)

        H_u = np.zeros(N)
        T_u = np.zeros(N)
        M_u = np.zeros(N)
        H_u1 = np.zeros(N)
        T_u1 = np.zeros(N)
        M_u1 = np.zeros(N)
        H_b = np.zeros(N)
        T_b = np.zeros(N)
        M_b = np.zeros(N)
        H_adaptive = np.zeros(N)
        T_adaptive = np.zeros(N)
        M_adaptive = np.zeros(N)
        H_m = np.zeros(N)
        T_m = np.zeros(N)
        M_m = np.zeros(N)
        H_ME = np.zeros(N)
        H_SCF = np.zeros(N)
        H_C2ST = np.zeros(N)
        Tu_C2ST = np.zeros(N)
        Tl_C2ST = np.zeros(N)
        S_C2ST = np.zeros(N)
        np.random.seed(1102)
        count_u = 0
        count_adp = 0
        count_ME = 0
        count_SCF = 0
        count_C2ST = 0
        for k in range(N):
            np.random.seed(seed=11 * k + 10 + n)
            s1, s2 = sample_blobs_Q(N1, sigma_mx_2)
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, device, dtype)
            h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, alpha, device, dtype, ep)
            h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, N1, S, sigma, sigma0, alpha, device, dtype)
            # h_m, threshold_m, mmd_value_m = TST_MMD_median(S, N_per, LM, N1, alpha, device, dtype)
            h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=1, seed=15)
            h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=1, seed=15)
            H_C2ST[k], Tu_C2ST[k], S_C2ST[k] = TST_C2ST(S,N1,N_per,alpha,model_C2ST, w_C2ST, b_C2ST,device,dtype)
            count_u = count_u + h_u
            count_adp = count_adp + h_adaptive
            count_ME = count_ME + h_ME
            count_SCF = count_SCF + h_SCF
            count_C2ST = count_C2ST + int(H_C2ST[k])
            print("MMD-DK:", count_u,"MMD-OPT:", count_adp,"MMD-ME:", count_ME,"SCF:", count_SCF, "C2ST: ", count_C2ST)
            # print("h_u:", h_u, "Threshold_u:", threshold_u, "MMD_value_u:", mmd_value_u)
            # # print("h_u1:", h_u1, "Threshold_u1:", threshold_u1, "MMD_value_u1:", mmd_value_u1)
            # # print("h_b:", h_b, "Threshold_b:", threshold_b, "MMD_value_b:", mmd_value_b)
            # print("h_adaptive:", h_adaptive, "Threshold_adaptive:", threshold_adaptive, "MMD_value_adaptive:", mmd_value_adaptive)
            # # print("h_m:", h_m, "Threshold_m:", threshold_m, "MMD_value_m:", mmd_value_m)
            # print("h_ME:", h_ME)
            # print("h_SCF:", h_SCF)
            H_u[k] = h_u
            T_u[k] = threshold_u
            M_u[k] = mmd_value_u
            # H_u1[k] = h_u1
            # T_u1[k] = threshold_u1
            # M_u1[k] = mmd_value_u1
            H_adaptive[k] = h_adaptive
            T_adaptive[k] = threshold_adaptive
            M_adaptive[k] = mmd_value_adaptive
            # H_b[k] = h_b
            # T_b[k] = threshold_b
            # M_b[k] = mmd_value_b
            # H_m[k] = h_m
            # T_m[k] = threshold_m
            # M_m[k] = mmd_value_m
            H_ME[k] = h_ME
            H_SCF[k] = h_SCF
        print("Reject rate_u: ", H_u.sum()/N_f,"Reject rate_C2ST: ", H_C2ST.sum()/N_f,"Reject rate_adaptive: ", H_adaptive.sum()/N_f,"Reject rate_ME: ", H_ME.sum()/N_f,"Reject rate_SCF: ", H_SCF.sum()/N_f,"Reject rate_m: ", H_m.sum()/N_f)
        Results[0, kk] = H_u.sum() / N_f
        Results[1, kk] = H_C2ST.sum() / N_f
        Results[2, kk] = H_adaptive.sum() / N_f
        Results[3, kk] = H_m.sum() / N_f
        Results[4, kk] = H_ME.sum() / N_f
        Results[5, kk] = H_SCF.sum() / N_f
        print(Results,Results.mean(1))
    count = count + 1
    f = open('Results_'+str(n)+'_H1_bugs_EPOPT.pckl_C2ST', 'wb')
    pickle.dump([Results,J_star_u,J_star_adp,Opt_ep], f)
    f.close()
