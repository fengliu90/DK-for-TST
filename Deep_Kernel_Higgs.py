# -*- coding: utf-8 -*-
"""
Created on Dec 21 14:57:02 2019
@author: Learning Deep Kernels for Two-sample Test
@Implementation of MMD-D in our paper on Higgs dataset

BEFORE USING THIS CODE:
1. This code requires PyTorch 1.1.0, which can be found in
https://pytorch.org/get-started/previous-versions/ (CUDA version is 10.1).
2. Numpy and Sklearn are also required. Users can install
Python via Anaconda (Python 3.7.3) to obtain both packages. Anaconda
can be found in https://www.anaconda.com/distribution/#download-section .
"""
import numpy as np
import torch
import pickle
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
parser.add_argument('--n', type=int, default=1000)
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
d = 4 # dimension of data
n = args.n # number of samples in one set
print('n: '+str(n)+' d: '+str(d))
N_epoch = 1000 # number of training epochs
x_in = d # number of neurons in the input layer, i.e., dimension of data
H = 20 # number of neurons in the hidden layer
x_out = 20 # number of neurons in the output layer
learning_rate = 0.00005
learning_ratea = 0.001
learning_rate_C2ST = 0.001
K = 10 # number of trails
N = 100 # number of test sets
N_f = 100.0 # number of test sets (float)

# Load data
data = pickle.load(open('./HIGGS_TST.pckl', 'rb'))
dataX = data[0]
dataY = data[1]
del data

# Naming variables
J_star_u = np.zeros([N_epoch])
J_star_adp = np.zeros([N_epoch])
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
Results = np.zeros([1,K])

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
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2*d), device, dtype)  # d = 3,5 ??
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = False
    print(epsilonOPT.item())

    # Setup optimizer for training deep kernel
    optimizer_u = torch.optim.Adam(list(model_u.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=learning_rate)

    # Generate Higgs (P,Q)
    N1_T = dataX.shape[0]
    N2_T = dataY.shape[0]
    np.random.seed(seed=1102 * kk + n)
    ind1 = np.random.choice(N1_T, n, replace=False)
    np.random.seed(seed=819 * kk + n)
    ind2 = np.random.choice(N2_T, n, replace=False)
    s1 = dataX[ind1,:4]
    s2 = dataY[ind2,:4]
    N1 = n
    N2 = n
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)

    # Train deep kernel to maximize test power
    for t in range(N_epoch):
        # Compute epsilon, sigma and sigma_0
        ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))  # 10 ** (-10)#
        sigma = sigmaOPT ** 2
        sigma0_u = sigma0OPT ** 2
        # Compute output of the deep network
        modelu_output = model_u(S)
        # Compute J (STAT_u)
        TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
        mmd_value_temp = -1 * (TEMP[0]+10**(-8))  # 10**(-8)
        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))  # 0.1
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
            print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                  -1 * STAT_u.item())  # ,"Reg: ", loss1.item()

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
        # Generate Higgs (P,Q)
        np.random.seed(seed=1102 * (k+1) + n)
        ind1 = np.random.choice(N1_T, n, replace=False)
        np.random.seed(seed=819 * (k+2) + n)
        ind2 = np.random.choice(N2_T, n, replace=False)
        s1 = dataX[ind1, :4]
        s2 = dataY[ind2, :4]
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
np.save('./Results_HIGGS_n' + str(n) + '_H1_MMD-D', Results)