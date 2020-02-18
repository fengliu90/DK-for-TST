# -*- coding: utf-8 -*-
"""
Created on Dec 21 14:57:02 2019
@author: Learning Deep Kernels for Two-sample Test
@Implementation of MMD-D in our paper on Blob dataset

BEFORE USING THIS CODE:
1. This code requires PyTorch 1.1.0, which can be found in
https://pytorch.org/get-started/previous-versions/ (CUDA version is 10.1).
2. Numpy and Sklearn are also required. Users can install
Python via Anaconda (Python 3.7.3) to obtain both packages. Anaconda
can be found in https://www.anaconda.com/distribution/#download-section .
"""
import numpy as np
import torch
from sklearn.utils import check_random_state
from utils import MatConvert, MMDu, TST_MMD_u

def sample_blobs(n, rows=3, cols=3, sep=1, rs=None):
    """Generate Blob-S for testing type-I error."""
    rs = check_random_state(rs)
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
    """Generate Blob-D for testing type-II error (or test power)."""
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

# Setup seeds
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True
is_cuda = True
# Setup for all experiments
dtype = torch.float
device = torch.device("cuda:0")
N_per = 100 # permutation times
alpha = 0.05 # test threshold
n_list = [10,20,40,50,70,80,90,100] # number of samples in per mode
x_in = 2 # number of neurons in the input layer, i.e., dimension of data
H = 50 # number of neurons in the hidden layer
x_out = 50 # number of neurons in the output layer
learning_rate = 0.0005 # learning rate for MMD-D on Blob
N_epoch = 1000 # number of training epochs
K = 10 # number of trails
N = 100 # # number of test sets
N_f = 100.0 # number of test sets (float)
# Generate variance and co-variance matrix of Q
sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2 = np.zeros([9,2,2])
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
# For each n in n_list, train deep kernel and run two-sample test
for n in n_list:
    np.random.seed(1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N1 = 9 * n
    N2 = 9 * n
    Results = np.zeros([1, K])
    Opt_ep = np.zeros([1, K])
    J_star_u = np.zeros([K, N_epoch])
    ep_OPT = np.zeros([K])
    s_OPT = np.zeros([K])
    s0_OPT = np.zeros([K])
    # Repeat experiments K times (K = 10) and report average test power (rejection rate)
    for kk in range(K):
        # Initialize parameters
        if is_cuda:
            model_u = ModelLatentF(x_in, H, x_out).cuda()
        else:
            model_u = ModelLatentF(x_in, H, x_out)
        epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
        sigma0OPT.requires_grad = True
        # Setup optimizer for training deep kernel
        optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate) #
        # Generate Blob-D
        np.random.seed(seed=112 * kk + 1 + n)
        s1,s2 = sample_blobs_Q(N1, sigma_mx_2)
        if kk==0:
            s1_o = s1
            s2_o = s2
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        # Train deep kernel to maximize test power
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        for t in range(N_epoch):
            # Compute epsilon, sigma and sigma_0
            ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2
            # Compute output of the deep network
            modelu_output = model_u(S)
            # Compute J (STAT_u)
            TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
            mmd_value_temp = -1 * TEMP[0]
            mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
            J_star_u[kk, t] = STAT_u.item()
            # Initialize optimizer and Compute gradient
            optimizer_u.zero_grad()
            STAT_u.backward(retain_graph=True)
            # Update weights using gradient descent
            optimizer_u.step()
            # Print MMD, std of MMD and J
            if t % 100 == 0:
                print("mmd_value: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic J: ",
                      -1 * STAT_u.item())
        h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, alpha, device,
                                                  dtype, ep)
        ep_OPT[kk] = ep.item()
        s_OPT[kk] = sigma.item()
        s0_OPT[kk] = sigma0_u.item()
        print(ep, epsilonOPT)
        # Compute test power of deep kernel based MMD
        H_u = np.zeros(N)
        T_u = np.zeros(N)
        M_u = np.zeros(N)
        np.random.seed(1102)
        count_u = 0
        for k in range(N):
            # Generate Blob-D
            np.random.seed(seed=11 * k + 10 + n)
            s1,s2 = sample_blobs_Q(N1, sigma_mx_2)
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, device, dtype)
            # Run two sample test (deep kernel) on generated data
            h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, alpha, device, dtype, ep)
            # Gather results
            count_u = count_u + h_u
            print("MMD-D:", count_u)
            H_u[k] = h_u
            T_u[k] = threshold_u
            M_u[k] = mmd_value_u
        # Print test power of MMD-D
        print("n =",str(n),"--- Test Power of MMD-D: ", H_u.sum()/N_f)
        Results[0, kk] = H_u.sum() / N_f
        print("n =",str(n),"--- Test Power of MMD-D (K times): ",Results[0])
        print("n =",str(n),"--- Average Test Power of MMD-D: ",Results[0].sum()/(kk+1))
    np.save('./Results_Blob_'+str(n)+'_H1_MMD-D',Results)