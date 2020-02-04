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
import argparse, sys
from scipy.stats import norm
parser = argparse.ArgumentParser()
from TST_utils_HD import MatConvert, Pdist2, MMDu, get_item, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_SCF, TST_C2ST, C2ST_NN_fit,MMDu_linear_kernel, TST_MMD_u_linear_kernel,TST_LCE

parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--d', type=int, default=10)
args = parser.parse_args()

# Data generation - blob
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True
is_cuda = True

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


dtype = torch.float
device = torch.device("cuda:0")

N_per = 100 # permutation times
alpha = 0.05 # test threshold
d = args.d
n = args.n
print('n: '+str(n)+'d: '+str(d))
Num_clusters = 2
mu_mx = np.zeros([Num_clusters,d])
mu_mx[1] = mu_mx[1] + 0.5
# mu_mx[2] = mu_mx[2] - 0.5
sigma_mx_1 = np.identity(d)
sigma_mx_2 = [np.identity(d),np.identity(d)] # ,np.identity(d)
sigma_mx_2[0][0,1] = 0.5
sigma_mx_2[0][1,0] = 0.5
sigma_mx_2[1][0,1] = -0.5
sigma_mx_2[1][1,0] = -0.5

sigma = 2*d/5 # 2*d for d=10 d for d=5
batch_size = min(n * 2, 128)
N_epoch = 1000#int(1000 * 2 * n / batch_size)
N_epoch1 = 1000
J_star_u = np.zeros([N_epoch1])
J_star_adp = np.zeros([N_epoch1])

s1 = np.zeros([n*Num_clusters, d])
s2 = np.zeros([n*Num_clusters, d])
x_in = d
H =3*d              # 30 for d=5 100 for d=30
x_out = 3*d         # 30 for d=5
# learning_rate = 0.00005 # high dimension -- low learning rate
learning_rate = 0.00001   # 0.00005 for d=10 0.0001 for d=3
learning_ratea = 0.00005
learning_rate_C2ST = 0.001
K = 10
Results = np.zeros([6,K])
reg = 0
ep = 10**(-10)
sigma0_u = 0.1/5 # 0.5 for d=5 0.05 for d = 3 0.1 for d=10
# r_full = norm.ppf(1-alpha, loc=0, scale=1)
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])

for kk in range(K):
    torch.manual_seed(kk * 19 + n)
    torch.cuda.manual_seed(kk * 19 + n)
    if is_cuda:
        model_u = ModelLatentF(x_in, H, x_out).cuda()  ## notice foolish initialization of network!!!
        model_u1 = ModelLatentF(x_in, H, x_out).cuda()
    else:
        model_u = ModelLatentF(x_in, H, x_out)
        model_u1 = ModelLatentF(x_in, H, x_out)

    optimizer_u = torch.optim.Adam(list(model_u.parameters()), lr=learning_rate)
    optimizer_u1 = torch.optim.Adam(list(model_u1.parameters()), lr=learning_rate)

    for i in range(Num_clusters):
        np.random.seed(seed=1102*kk + i + n)
        s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
    for i in range(Num_clusters):
        np.random.seed(seed=819*kk + 1 + i + n)
        s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n) # sigma_mx_2[i]
    if kk==0:
        s1_o = s1
        s2_o = s2
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    N1 = Num_clusters*n
    N2 = Num_clusters*n
    y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
    pred, STAT_C2ST, model_C2ST, w_C2ST, b_C2ST = C2ST_NN_fit(S, y, N1, x_in, H, x_out, learning_rate_C2ST, N_epoch,
                                                              batch_size, device, dtype)
    # LM = MMD_L(N1, N2, device, dtype)
    # v = torch.div(torch.ones([N1+N2, N1+N2], dtype=torch.float, device=device), (N1+N2)*1.0)

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    # MoS = model_u(S)
    # Dxy = Pdist2(MoS[:N1,:],MoS[N1:,:])
    # sigma0_u = get_item(Dxy.view(-1,1).squeeze().kthvalue(int(Dxy.size(0)*Dxy.size(1)*0.005))[0],is_cuda).tolist()
    # print(sigma0_u)
    for t in range(N_epoch1):
        modelu1_output = model_u1(S)
        TEMP1 = MMDu(modelu1_output, N1, S, sigma, sigma0_u,is_smooth=False)
        mmd_value_temp = -1 * (TEMP1[0] + 10 ** (-8))  # 10**(-8)
        mmd_std_temp = torch.sqrt(TEMP1[1] + 10 ** (-8))  # 0.1
        if mmd_std_temp.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_temp.item()):
            print('error!!')
        STAT_u1 = torch.div(mmd_value_temp, mmd_std_temp)  # - r_full / (N1+N2)
        J_star_u[t] = STAT_u1.item()
        optimizer_u1.zero_grad()
        STAT_u1.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u1.step()
        if t % 100 == 0:
            print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                  -1 * STAT_u1.item())  # ,"Reg: ", loss1.item()

    h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, N1, S, sigma, sigma0_u, ep, alpha, device, dtype, is_smooth=False)
    print("h:", h_u1, "Threshold:", threshold_u1, "MMD_value:", mmd_value_u1) # G+J

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
        mmd_value_temp = -1 * (TEMP[0]+10**(-8))  # 10**(-8)
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))  # 0.1
        if mmd_std_temp.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_temp.item()):
            print('error!!')
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp) #- r_full / (N1+N2)
        J_star_u[t] = STAT_u.item()
        optimizer_u.zero_grad()
        STAT_u.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u.step()
        if t % 100 ==0:
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
        mmd_value_tempa = -1 * (TEMPa[0]+10**(-8))
        mmd_std_tempa = torch.sqrt(TEMPa[1]+10**(-8))
        if mmd_std_tempa.item() == 0:
            print('std error!!')
        if np.isnan(mmd_std_tempa.item()):
            print('std error!!')
        STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        J_star_adp[t] = STAT_adaptive.item()
        optimizer_sigma0.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        # Update sigma0 using gradient descent
        optimizer_sigma0.step()
        if t % 100 == 0:
            print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                  -1 * STAT_adaptive.item())
    h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S_m, N_per, N1, S_m, sigma, sigma0, alpha, device, dtype)
    print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive) # G+C


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
        for i in range(Num_clusters):
            np.random.seed(seed=1102 * (k+2) + 2*kk + i + n)
            s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        for i in range(Num_clusters):
            np.random.seed(seed=819 * (k + 1) + 2*kk + i + n)
            s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n) # sigma_mx_2[i]
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        S_m = model_C2ST(S)
        h_u, threshold_u, mmd_value_u = TST_MMD_u_linear_kernel(model_u(S), N_per, N1, alpha, device, dtype) # L+J
        h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, N1, S, sigma, sigma0_u, ep, alpha, device, dtype, is_smooth=False) # G+J
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S_m, N_per, N1, S_m, sigma, sigma0, alpha, device, dtype) # G+C
        H_C2ST[k], Tu_C2ST[k], S_C2ST[k] = TST_LCE(S, N1, N_per, alpha, model_C2ST, w_C2ST, b_C2ST, device, dtype) # L+C
        count_u = count_u + h_u
        count_adp = count_adp + h_adaptive
        count_u1 = count_u1 + h_u1
        count_C2ST = count_C2ST + int(H_C2ST[k])
        print("L+J:", count_u,"G+J:", count_u1,"G+C:", count_adp,"L+C:", count_C2ST)
        H_u[k] = h_u
        T_u[k] = threshold_u
        M_u[k] = mmd_value_u
        H_u1[k] = h_u1
        T_u1[k] = threshold_u1
        M_u1[k] = mmd_value_u1
        H_adaptive[k] = h_adaptive
        T_adaptive[k] = threshold_adaptive
        M_adaptive[k] = mmd_value_adaptive
    print("Reject rate_LJ: ", H_u.sum() / N_f, "Reject rate_GJ: ", H_u1.sum() / N_f, "Reject rate_GC:", H_adaptive.sum() / N_f,
          "Reject rate_LC: ", H_C2ST.sum() / N_f)
    Results[0, kk] = H_u.sum() / N_f
    Results[1, kk] = H_u1.sum() / N_f
    Results[2, kk] = H_adaptive.sum() / N_f
    Results[3, kk] = H_C2ST.sum() / N_f
    print(Results,Results.mean(1))
f = open('./Results_HDGM_n'+str(n)+'_d'+str(d)+'_H0_ABL.pckl', 'wb')
pickle.dump([Results,J_star_u,J_star_adp], f)
f.close()