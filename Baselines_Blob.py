# -*- coding: utf-8 -*-
"""
Created on Dec 21 14:57:02 2019
@author: Learning Deep Kernels for Two-sample Test
@Implementation of baselines in our paper on Blob dataset

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
from sklearn.utils import check_random_state
from utils import MatConvert, Pdist2, MMDu, TST_LCE, TST_MMD_adaptive_bandwidth, TST_ME, TST_SCF, TST_C2ST, C2ST_NN_fit

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
learning_rate_MMD_O = 0.0005
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
    Results = np.zeros([5, K])
    J_star_adp = np.zeros([K, N_epoch])
    batch_size = min(n * 2, 128) # batch size for C2ST-S and C2ST-L
    N_epoch_C2ST = int(500 * 18 * n / batch_size)
    # Repeat experiments K times (K = 10) and report average test power (rejection rate)
    for kk in range(K):
        # Generate Blob-D
        np.random.seed(seed=112 * kk + 1 + n)
        s1, s2 = sample_blobs_Q(N1, sigma_mx_2)
        if kk == 0:
            s1_o = s1
            s2_o = s2
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        # Train C2ST-L
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
        pred, STAT_C2ST_L, model_C2ST_L, w_C2ST_L, b_C2ST_L = C2ST_NN_fit(S, y, N1, x_in, H, x_out, 0.0005,
                                                                          N_epoch_C2ST, batch_size, device, dtype)
        # Train C2ST-S
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        y = (torch.cat((torch.zeros(N1, 1), torch.ones(N2, 1)), 0)).squeeze(1).to(device, dtype).long()
        pred, STAT_C2ST_S, model_C2ST_S, w_C2ST_S, b_C2ST_S = C2ST_NN_fit(S, y, N1, x_in, H, x_out, 0.0005,
                                                                          N_epoch_C2ST, batch_size, device, dtype)

        # Train MMD-O
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        Dxy = Pdist2(S[:N1,:],S[N1:,:])
        sigma0 = Dxy.median() * (2**(-2.1))
        print(sigma0)
        sigma0.requires_grad = True
        optimizer_sigma0 = torch.optim.Adam([sigma0], lr=learning_rate_MMD_O)
        for t in range(N_epoch):
            TEMPa = MMDu(S, N1, S, 0, sigma0, is_smooth=False)
            mmd_value_tempa = -1 * (TEMPa[0]+10**(-8))
            mmd_std_tempa = torch.sqrt(TEMPa[1]+10**(-8))
            STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
            J_star_adp[kk, t] = STAT_adaptive.item()
            optimizer_sigma0.zero_grad()
            STAT_adaptive.backward(retain_graph=True)
            optimizer_sigma0.step()
            if t % 100 == 0:
                print("mmd_value: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic J: ",
                      -1 * STAT_adaptive.item())
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, N1, S, 0, sigma0, alpha, device, dtype)

        # Train ME
        np.random.seed(seed=1102)
        test_locs_ME, gwidth_ME = TST_ME(S, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=5, seed=15)
        h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=5, seed=15)
        print("h:", h_ME, "test_locs_ME:", test_locs_ME, "gwidth_ME:", gwidth_ME)

        # Train SCF
        np.random.seed(seed=1102)
        test_freqs_SCF, gwidth_SCF = TST_SCF(S, N1, alpha, is_train=True, test_freqs=1, gwidth=1, J=5, seed=15)
        h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=5, seed=15)
        print("h:", h_SCF, "test_freqs_SCF:", test_freqs_SCF, "gwidth_SCF:", gwidth_SCF)

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
            # Generate Blob-D
            np.random.seed(seed=11 * k + 10 + n)
            s1, s2 = sample_blobs_Q(N1, sigma_mx_2)
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, device, dtype)
            # Run two sample tests (baselines) on generated data
            # MMD-O
            h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, N1, S, 0, sigma0,
                                                                                            alpha, device, dtype)
            # ME
            h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=5, seed=15)
            # SCF
            h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=5, seed=15)
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
            print("MMD-O:", count_adp,"C2ST-L:", count_C2ST_L,"C2ST-S:", count_C2ST_S,"ME:", count_ME, "SCF:", count_SCF)
            H_adaptive[k] = h_adaptive
            T_adaptive[k] = threshold_adaptive
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
        print("n =",str(n),"--- Test Power of Baselines (K times): ")
        print(Results)
        print("n =", str(n), "--- Average Test Power of Baselines (K times): ")
        print("MMD-O: ", (Results.sum(1) / (kk+1))[0], "C2ST-L: ", (Results.sum(1) / (kk+1))[1],
              "C2ST-S: ", (Results.sum(1) / (kk+1))[2], "ME:", (Results.sum(1) / (kk+1))[3],
              "SCF: ", (Results.sum(1) / (kk+1))[4])
    np.save('./Results_Blob_' + str(n) + '_H1_Baselines', Results)