# -*- coding: utf-8 -*-
"""
Created on Dec 21 14:57:02 2019
@author: Learning Deep Kernels for Two-sample Test
@Implementation of baselines in our paper on Higgs dataset

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
import pickle
import argparse
parser = argparse.ArgumentParser()
from TST_utils_HD import MatConvert, MMDu, TST_MMD_adaptive_bandwidth, TST_LCE, TST_ME, TST_SCF, TST_C2ST, C2ST_NN_fit

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
N_epoch_C = 1000
x_in = d # number of neurons in the input layer, i.e., dimension of data
H = 20 # number of neurons in the hidden layer
x_out = 20 # number of neurons in the output layer
learning_ratea = 0.001
batch_size = min(n * 2, 128)
sigma0_u = 0.005
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
Results = np.zeros([5,K])
# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    torch.manual_seed(kk * 19 + n)
    torch.cuda.manual_seed(kk * 19 + n)
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

    # Train C2ST-L
    y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
    pred, STAT_C2ST_L, model_C2ST_L, w_C2ST_L, b_C2ST_L = C2ST_NN_fit(S, y, N1, x_in, H, x_out, 0.001,
                                                                      N_epoch, batch_size, device, dtype)
    # Train C2ST-S
    np.random.seed(seed=819 * kk + n)
    y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
    pred, STAT_C2ST_S, model_C2ST_S, w_C2ST_S, b_C2ST_S = C2ST_NN_fit(S, y, N1, x_in, H, x_out, 0.001, N_epoch,
                                                              batch_size, device, dtype)

    # Train MMD-O
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    sigma0 = 2*d * torch.rand([1]).to(device, dtype)
    sigma0.requires_grad = True
    optimizer_sigma0 = torch.optim.Adam([sigma0], lr=learning_ratea)
    for t in range(N_epoch):
        TEMPa = MMDu(S, N1, S, 0, sigma0, is_smooth=False)
        mmd_value_tempa = -1 * (TEMPa[0]+10**(-8))
        mmd_std_tempa = torch.sqrt(TEMPa[1]+10**(-8))
        if mmd_std_tempa.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_tempa.item()):
            print('error!!')
        STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        J_star_adp[t] = STAT_adaptive.item()
        optimizer_sigma0.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        # Update sigma0 using gradient descent
        optimizer_sigma0.step()
        if t % 100 == 0:
            print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                  -1 * STAT_adaptive.item())
    h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, N1, S, 0, sigma0, alpha, device, dtype)
    print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)

    # Train ME
    np.random.seed(seed=1102)
    test_locs_ME, gwidth_ME = TST_ME(S, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=10, seed=15)
    h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=10, seed=15)

    # Train SCF
    np.random.seed(seed=1102)
    test_freqs_SCF, gwidth_SCF = TST_SCF(S, N1, alpha, is_train=True, test_freqs=1, gwidth=1, J=10, seed=15)
    h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=10, seed=15)

    # Compute test power of baselines
    H_adaptive = np.zeros(N)
    T_adaptive = np.zeros(N)
    M_adaptive = np.zeros(N)
    H_ME = np.zeros(N)
    H_SCF = np.zeros(N)
    H_C2ST_S = np.zeros(N)
    H_C2ST_L = np.zeros(N)
    Tu_C2ST_S = np.zeros(N)
    Tu_C2ST_L = np.zeros(N)
    S_C2ST_S = np.zeros(N)
    S_C2ST_L = np.zeros(N)
    np.random.seed(1102)
    count_adp = 0
    count_ME = 0
    count_SCF = 0
    count_C2ST_S = 0
    count_C2ST_L = 0
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

        # Run two sample tests (baselines) on generated data
        # MMD-O
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, N1, S, 0, sigma0,
                                                                                        alpha, device, dtype)
        # ME
        h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=10, seed=15)
        # SCF
        h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=10, seed=15)
        # C2ST-S
        H_C2ST_S[k], Tu_C2ST_S[k], S_C2ST_S[k] = TST_C2ST(S, N1, N_per, alpha, model_C2ST_S, w_C2ST_S, b_C2ST_S,
                                                          device, dtype)
        # C2ST-L
        H_C2ST_L[k], Tu_C2ST_L[k], S_C2ST_L[k] = TST_LCE(S, N1, N_per, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L,
                                                         device, dtype)
        # Gather results
        count_adp = count_adp + h_adaptive
        count_ME = count_ME + h_ME
        count_SCF = count_SCF + h_SCF
        count_C2ST_S = count_C2ST_S + int(H_C2ST_S[k])
        count_C2ST_L = count_C2ST_L + int(H_C2ST_L[k])
        print("MMD-O:", count_adp, "C2ST-L:", count_C2ST_L, "C2ST-S:", count_C2ST_S, "ME:", count_ME, "SCF:", count_SCF)
        H_adaptive[k] = h_adaptive
        M_adaptive[k] = mmd_value_adaptive
        H_ME[k] = h_ME
        H_SCF[k] = h_SCF
    print("Test Power of MMD-O: ", H_adaptive.sum() / N_f, "Test Power of C2ST-L: ", H_C2ST_L.sum() / N_f,
          "Test Power of C2ST-S: ", H_C2ST_S.sum() / N_f, "Test Power of ME:", H_ME.sum() / N_f,
          "Test Power of SCF: ", H_SCF.sum() / N_f)
    Results[0, kk] = H_adaptive.sum() / N_f
    Results[1, kk] = H_C2ST_L.sum() / N_f
    Results[2, kk] = H_C2ST_S.sum() / N_f
    Results[3, kk] = H_ME.sum() / N_f
    Results[4, kk] = H_SCF.sum() / N_f
    print("Test Power of Baselines (K times): ")
    print(Results)
    print("Average Test Power of Baselines (K times): ")
    print("MMD-O: ", (Results.sum(1) / (kk + 1))[0], "C2ST-L: ", (Results.sum(1) / (kk + 1))[1],
          "C2ST-S: ", (Results.sum(1) / (kk + 1))[2], "ME:", (Results.sum(1) / (kk + 1))[3],
          "SCF: ", (Results.sum(1) / (kk + 1))[4])
np.save('./Results_HIGGS_' + str(n) + '_H1_Baselines', Results)