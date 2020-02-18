# -*- coding: utf-8 -*-
"""
Created on Dec 21 14:57:02 2019
@author: Learning Deep Kernels for Two-sample Test
@Implementation of MMD-D in our paper on HDGM dataset

BEFORE USING THIS CODE:
1. This code requires PyTorch 1.1.0, which can be found in
https://pytorch.org/get-started/previous-versions/ (CUDA version is 10.1).
2. Numpy and Sklearn are also required. Users can install
Python via Anaconda (Python 3.7.3) to obtain both packages. Anaconda
can be found in https://www.anaconda.com/distribution/#download-section .
"""
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
from TST_utils_HD import MatConvert, MMDu, TST_MMD_u

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
N_per = 200 # permutation times
alpha = 0.05 # test threshold
d = args.d # dimension of data
n = args.n # number of samples in per mode
print('n: '+str(n)+' d: '+str(d))
x_in = d # number of neurons in the input layer, i.e., dimension of data
H =3*d # number of neurons in the hidden layer
x_out = 3*d # number of neurons in the output layer
learning_rate = 0.00005 # default learning rate for MMD-D on HDGM
N_epoch = 1000 # number of training epochs
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
s1 = np.zeros([n*Num_clusters, d])
s2 = np.zeros([n*Num_clusters, d])

# Naming variables
Results = np.zeros([1,K])
J_star_u = np.zeros([N_epoch])
J_star_adp = np.zeros([N_epoch])
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    torch.manual_seed(kk * 19 + n)
    torch.cuda.manual_seed(kk * 19 + n)
    # Initialize parameters
    if is_cuda:
        model_u = ModelLatentF(x_in, H, x_out).cuda()
    else:
        model_u = ModelLatentF(x_in, H, x_out)
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * d), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
    sigma0OPT.requires_grad = True
    print(epsilonOPT.item())

    # Setup optimizer for training deep kernel
    optimizer_u = torch.optim.Adam(list(model_u.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT],
                                   lr=learning_rate)
    # Generate HDGM-D
    for i in range(Num_clusters):
        np.random.seed(seed=1102*kk + i + n)
        s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
    for i in range(Num_clusters):
        np.random.seed(seed=819*kk + 1 + i + n)
        s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n) # sigma_mx_2[i]
    if kk==0:
        s1_o = s1
        s2_o = s2
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    N1 = Num_clusters*n
    N2 = Num_clusters*n

    # Train deep kernel to maximize test power
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for t in range(N_epoch):
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        sigma0_u = sigma0OPT ** 2
        # Compute output of the deep network
        modelu_output = model_u(S)
        # Compute J (STAT_u)
        TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
        mmd_value_temp = -1 * (TEMP[0]+10**(-8))
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
        if mmd_std_temp.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_temp.item()):
            print('error!!')
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        J_star_u[t] = STAT_u.item()
        # Initialize optimizer and Compute gradient
        optimizer_u.zero_grad()
        STAT_u.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u.step()
        # Print MMD, std of MMD and J
        if t % 100 ==0:
            print("mmd_value: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                  -1 * STAT_u.item())

    h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, ep, alpha, device, dtype)
    print("h:", h_u, "Threshold:", threshold_u, "MMD_value:", mmd_value_u)
    ep_OPT[kk] = ep.item()
    s_OPT[kk] = sigma.item()
    s0_OPT[kk] = sigma0_u.item()

    # Compute test power of deep kernel based MMD
    H_u = np.zeros(N)
    T_u = np.zeros(N)
    M_u = np.zeros(N)
    np.random.seed(1102)
    count_u = 0
    for k in range(N):
        # Generate Blob-D
        for i in range(Num_clusters):
            np.random.seed(seed=1102 * (k+2) + 2*kk + i + n)
            s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        for i in range(Num_clusters):
            np.random.seed(seed=819 * (k + 1) + 2*kk + i + n)
            s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n) # sigma_mx_2[i]
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        # Run two sample test (deep kernel) on generated data
        h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, ep, alpha, device, dtype)
        # Gather results
        count_u = count_u + h_u
        print("MMD-DK:", count_u)
        H_u[k] = h_u
        T_u[k] = threshold_u
        M_u[k] = mmd_value_u
    # Print test power of MMD-D
    print("Test Power of MMD-D: ", H_u.sum() / N_f)
    Results[0, kk] = H_u.sum() / N_f
    print("Test Power of MMD-D (K times): ", Results[0])
    print("Average Test Power of MMD-D: ", Results[0].sum() / (kk + 1))
np.save('./Results_HDGM_n'+str(n)+'_d'+str(d)+'_H1_MMD-D', Results)