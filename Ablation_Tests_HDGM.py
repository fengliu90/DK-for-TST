# -*- coding: utf-8 -*-
"""
Created on Dec 21 14:57:02 2019
@author: Learning Deep Kernels for Two-sample Test
@Implementation of ablation studies in our paper on HDGM dataset

BEFORE USING THIS CODE:
1. This code requires PyTorch 1.1.0, which can be found in
https://pytorch.org/get-started/previous-versions/ (CUDA version is 10.1).
2. This code also requires freqopttest repo (interpretable nonparametric two-sample test)
to implement ME and SCF tests, which can be installed by
   pip install git+https://github.com/wittawatj/interpretable-test
3. Numpy and Sklearn are also required. Users can install
Python via Anaconda (Python 3.7.3) to obtain both packages. Anaconda
can be found in https://www.anaconda.com/distribution/#download-section .
"""
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
from utils_HD import MatConvert, Pdist2, MMDu, get_item, TST_MMD_adaptive_bandwidth, TST_MMD_u, C2ST_NN_fit, MMDu_linear_kernel, TST_MMD_u_linear_kernel

class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""

    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )

    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

# parameters to generate data
parser.add_argument('--n', type=int, default=1000) # number of samples per mode
parser.add_argument('--d', type=int, default=10) # dimension of samples (default value is 10)
args = parser.parse_args()

# Setup seeds
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True
is_cuda = True

# Setup for experiments
dtype = torch.float
device = torch.device("cuda:0")
N_per = 100 # permutation times
alpha = 0.05 # test threshold
d = args.d # dimension of data
n = args.n # number of samples in per mode
print('n: '+str(n)+' d: '+str(d))
x_in = d # number of neurons in the input layer, i.e., dimension of data
H =3*d # number of neurons in the hidden layer
x_out = 3*d # number of neurons in the output layer
sigma = 2*d/5 # 2*d for d=10 d for d=5
ep = 10**(-10)
sigma0_u = 0.1/5
learning_rate = 0.00005
learning_ratea = 0.00005
learning_rateLJ = 0.0001
learning_rate_C2ST = 0.001
batch_size = min(n * 2, 128) # batch size for training deep networks for G+C and D+C
N_epoch_C = 1000 # number of epochs for training deep networks for G+C and D+C
N_epoch = 1000 # number of epochs for training L+J and G+J
K = 10 # number of trails
N = 100 # # number of test sets
N_f = 100.0 # number of test sets (float)

# Generate variance and co-variance matrix of Q
Num_clusters = 2 # number of modes
mu_mx = np.zeros([Num_clusters,d])
mu_mx[1] = mu_mx[1] + 0.5
sigma_mx_1 = np.identity(d)
sigma_mx_2 = [np.identity(d),np.identity(d)]
sigma_mx_2[0][0,1] = 0.5
sigma_mx_2[0][1,0] = 0.5
sigma_mx_2[1][0,1] = -0.5
sigma_mx_2[1][1,0] = -0.5

# Naming variables
s1 = np.zeros([n*Num_clusters, d])
s2 = np.zeros([n*Num_clusters, d])
J_star_u = np.zeros([N_epoch])
J_star_adp = np.zeros([N_epoch])
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
Results = np.zeros([4,K])

# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    torch.manual_seed(kk * 19 + n)
    torch.cuda.manual_seed(kk * 19 + n)
    # Initialize parameters
    if is_cuda:
        model_u = ModelLatentF(x_in, H, x_out).cuda()
        model_u1 = ModelLatentF(x_in, H, x_out).cuda()
    else:
        model_u = ModelLatentF(x_in, H, x_out)
        model_u1 = ModelLatentF(x_in, H, x_out)
    # Setup optimizer for training deep kernels (L+J and G+J)
    optimizer_u = torch.optim.Adam(list(model_u.parameters()), lr=learning_rateLJ)
    optimizer_u1 = torch.optim.Adam(list(model_u1.parameters()), lr=learning_rate)

    # Generate HDGM-D
    for i in range(Num_clusters):
        np.random.seed(seed=1102*kk + i + n)
        s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
    for i in range(Num_clusters):
        np.random.seed(seed=819*kk + 1 + i + n)
        s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
        # REPLACE above line with
        # s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        # for validating type-I error (s1 ans s2 are from the same distribution)
    if kk==0:
        s1_o = s1
        s2_o = s2
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    N1 = Num_clusters*n
    N2 = Num_clusters*n

    # Train deep networks for G+C and D+C
    y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
    pred, STAT_C2ST, model_C2ST, w_C2ST, b_C2ST = C2ST_NN_fit(S, y, N1, x_in, H, x_out, learning_rate_C2ST, N_epoch_C,
                                                              batch_size, device, dtype)
    # Train G+J
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for t in range(N_epoch):
        modelu1_output = model_u1(S)
        TEMP1 = MMDu(modelu1_output, N1, S, sigma, sigma0_u,is_smooth=False)
        mmd_value_temp = -1 * (TEMP1[0] + 10 ** (-8))
        mmd_std_temp = torch.sqrt(TEMP1[1] + 10 ** (-8))
        if mmd_std_temp.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_temp.item()):
            print('error!!')
        STAT_u1 = torch.div(mmd_value_temp, mmd_std_temp)
        optimizer_u1.zero_grad()
        STAT_u1.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u1.step()
        if t % 100 == 0:
            print("mmd_value: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                  -1 * STAT_u1.item())
    h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, N1, S, sigma, sigma0_u, ep, alpha, device, dtype, is_smooth=False)
    print("h:", h_u1, "Threshold:", threshold_u1, "MMD_value:", mmd_value_u1)

    # Train L+J
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for t in range(N_epoch):
        modelu_output = model_u(S)
        TEMP = MMDu_linear_kernel(modelu_output, N1)
        mmd_value_temp = -1 * (TEMP[0]+10**(-8))
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
        if mmd_std_temp.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_temp.item()):
            print('error!!')
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        J_star_u[t] = STAT_u.item()
        optimizer_u.zero_grad()
        STAT_u.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u.step()
        if t % 100 ==0:
            print("mmd_value: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                  -1 * STAT_u.item())
    h_u, threshold_u, mmd_value_u = TST_MMD_u_linear_kernel(model_u(S), N_per, N1, alpha, device, dtype)
    print("h:", h_u, "Threshold:", threshold_u, "MMD_value:", mmd_value_u)

    # Train G+C
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    S_m = model_C2ST(S)
    Dxy_m = Pdist2(S_m[:N1, :], S_m[N1:, :])
    sigma0 = get_item(Dxy_m.median() * (2 ** (-4)), is_cuda)
    sigma0 = torch.from_numpy(sigma0).to(device, dtype)
    sigma0.requires_grad = True
    optimizer_sigma0 = torch.optim.Adam([sigma0], lr=learning_ratea)
    for t in range(N_epoch):
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
    print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)

    # Train D+C
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    S_m = model_C2ST(S)
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(6 * d), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
    sigma0OPT.requires_grad = True
    optimizer_sigma0_1 = torch.optim.Adam([epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=learning_ratea)
    for t in range(N_epoch):
        ep_1 = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
        sigma_1 = sigmaOPT ** 2
        sigma0_1 = sigma0OPT ** 2
        TEMPa_1 = MMDu(S_m, N1, S, sigma_1, sigma0_1, ep_1)
        mmd_value_tempa_1 = -1 * (TEMPa_1[0] + 10 ** (-8))
        mmd_std_tempa_1 = torch.sqrt(TEMPa_1[1] + 10 ** (-8))
        if mmd_std_tempa_1.item() == 0:
            print('std error!!')
        if np.isnan(mmd_std_tempa_1.item()):
            print('std error!!')
        STAT_adaptive_1 = torch.div(mmd_value_tempa_1, mmd_std_tempa_1)
        J_star_adp[t] = STAT_adaptive_1.item()
        optimizer_sigma0_1.zero_grad()
        STAT_adaptive_1.backward(retain_graph=True)
        # Update sigma0 using gradient descent
        optimizer_sigma0_1.step()
        if t % 100 == 0:
            print("mmd_value: ", -1 * mmd_value_tempa_1.item(), "mmd_std: ", mmd_std_tempa_1.item(), "Statistic: ",
                  -1 * STAT_adaptive_1.item())
    h_adaptive_1, threshold_adaptive_1, mmd_value_adaptive_1 = TST_MMD_u(S_m, N_per, N1, S, sigma_1, sigma0_1, ep_1,
                                                                                    alpha, device, dtype)
    print("h:", h_adaptive_1, "Threshold:", threshold_adaptive_1, "MMD_value:", mmd_value_adaptive_1)

    # Compute test power of deep kernel based tests
    H_u = np.zeros(N)
    T_u = np.zeros(N)
    M_u = np.zeros(N)
    H_u1 = np.zeros(N)
    T_u1 = np.zeros(N)
    M_u1 = np.zeros(N)
    H_adaptive = np.zeros(N)
    T_adaptive = np.zeros(N)
    M_adaptive = np.zeros(N)
    H_adaptive_1 = np.zeros(N)
    T_adaptive_1 = np.zeros(N)
    M_adaptive_1 = np.zeros(N)
    np.random.seed(1102)
    count_u = 0
    count_adp = 0
    count_u1 = 0
    count_adp_1 = 0
    for k in range(N):
        # Generate HDGM-D
        for i in range(Num_clusters):
            np.random.seed(seed=1102 * (k+2) + 2*kk + i + n)
            s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        for i in range(Num_clusters):
            np.random.seed(seed=819 * (k + 1) + 2*kk + i + n)
            s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
            # REPLACE above line with
            # s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
            # for validating type-I error (s1 ans s2 are from the same distribution)
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        # Run two sample tests (baselines) on generated data
        S_m = model_C2ST(S)
        # L+J
        h_u, threshold_u, mmd_value_u = TST_MMD_u_linear_kernel(model_u(S), N_per, N1, alpha, device, dtype)
        # G+J
        h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, N1, S, sigma, sigma0_u, ep, alpha, device,
                                                     dtype, is_smooth=False)
        # G+C
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S_m, N_per, N1, S_m, sigma,
                                                                                        sigma0, alpha, device,
                                                                                        dtype)
        # D+C
        h_adaptive_1, threshold_adaptive_1, mmd_value_adaptive_1 = TST_MMD_u(S_m, N_per, N1, S, sigma_1,sigma0_1,
                                                                                              ep_1, alpha, device,
                                                                                              dtype)
        # Gather results
        count_u = count_u + h_u
        count_adp = count_adp + h_adaptive
        count_u1 = count_u1 + h_u1
        count_adp_1 = count_adp_1 + h_adaptive_1
        print("L+J:", count_u, "G+J:", count_u1, "G+C:", count_adp, "D+C:", count_adp_1)
        H_u[k] = h_u
        T_u[k] = threshold_u
        M_u[k] = mmd_value_u
        H_u1[k] = h_u1
        T_u1[k] = threshold_u1
        M_u1[k] = mmd_value_u1
        H_adaptive[k] = h_adaptive
        T_adaptive[k] = threshold_adaptive
        M_adaptive[k] = mmd_value_adaptive
        H_adaptive_1[k] = h_adaptive_1
        T_adaptive_1[k] = threshold_adaptive_1
        M_adaptive_1[k] = mmd_value_adaptive_1
        # Print test power of deep kernel based tests
    print("Reject rate_LJ: ", H_u.sum() / N_f, "Reject rate_GJ: ", H_u1.sum() / N_f, "Reject rate_GC:",
          H_adaptive.sum() / N_f,
          "Reject rate_DC: ", H_adaptive_1.sum() / N_f)
    Results[0, kk] = H_u.sum() / N_f
    Results[1, kk] = H_u1.sum() / N_f
    Results[2, kk] = H_adaptive.sum() / N_f
    Results[3, kk] = H_adaptive_1.sum() / N_f
    print("Test Power of deep kernel based tests (K times): ")
    print(Results)
    print("Average Test Power of deep kernel based tests (K times): ")
    print(Results.sum(1) / (kk + 1))