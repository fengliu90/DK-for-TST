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
            torch.nn.Linear(H, H+1, bias=True),
            torch.nn.Softplus(),
            # torch.nn.Linear(H, H, bias=True),
            # torch.nn.ReLU(),
            # torch.nn.Linear(H, H, bias=True),
            # torch.nn.ReLU(),
            torch.nn.Linear(H+1, x_out+2, bias=True),
        )

    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

def get_item(x, is_cuda):
    if is_cuda:
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    return x

def MatConvert(x, device, dtype):
    x = torch.from_numpy(x).to(device, dtype)
    return x


def Pdist2(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return Pdist


def guassian_kernel(Fea, len_s, is_median = False):  # , kernel_mul=2.0, kernel_num=5, fix_sigma=None
    #    FeaALL = torch.cat([FeaS,FeaT],0)
    L2_distance = Pdist2(Fea, Fea)
    if is_median:
        L2D = L2_distance[0:len_s-1,len_s:Fea.size(1)]
        bandwidth = L2D[L2D != 0].median()
        kernel_val = torch.exp(-L2_distance /bandwidth)
    else:
        kernel_val = torch.exp(-L2_distance / 0.1)
    return kernel_val


def MyMMD(Fea, LM, len_s, is_median = False):  # , kernel_mul=2.0, kernel_num=5, fix_sigma=None
    kernels = guassian_kernel(Fea, len_s, is_median)
    loss = 0
    loss = kernels.mm(LM).trace()
    return loss


def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """
    Same as h1_mean_var() but takes in Gram matrices directly.
    """

    nx = Kx.shape[0]
    ny = Ky.shape[0]
    xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
    yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))

    # one-sample U-statistic.
    if use_1sample_U:
        xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
    else:
        xy = torch.div(torch.sum(Kxy), (nx * ny))
    mmd2 = xx - 2 * xy + yy

    if not is_var_computed:
        return mmd2, None

    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    # V2 = (hh - torch.diag(torch.diag(hh))).sum() / (nx-1) / nx
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    # hh = hh - torch.diag(torch.diag(hh))
    # # V1_t = torch.einsum('ij,in->ijn',[hh,hh])
    # # V1_diags = torch.einsum('...ii->...i', V1_t)
    # V1 = (torch.dot(hh.sum(1),hh.sum(1)) - (hh**2).sum())*6.0/ny/(ny-1)/(ny-2)/2.0
    # # V2_t = torch.einsum('ij,mn->ijmn', [hh, hh])
    # # V2_diags = torch.einsum('ijij->ij', V2_t)
    # V2 = (hh.sum()*hh.sum() - (hh**2).sum()) * 24.0 / ny / (ny - 1) / (ny - 2) / (ny - 3) /2.0
    # print(V1,V2)
    # varEst = 4 * (V1 - V2)

    # compute the variance
    # Kxd = Kx - torch.diag(torch.diag(Kx))
    # Kyd = Ky - torch.diag(torch.diag(Ky))
    # m = nx
    # n = ny
    # v = torch.zeros(11).cuda()
    #
    # Kxd_sum = torch.sum(Kxd)
    # Kyd_sum = torch.sum(Kyd)
    # Kxy_sum = torch.sum(Kxy)
    # Kxy2_sum = torch.sum(Kxy ** 2)
    # Kxd0_red = torch.sum(Kxd, 1)
    # Kyd0_red = torch.sum(Kyd, 1)
    # Kxy1 = torch.sum(Kxy, 1)
    # Kyx1 = torch.sum(Kxy, 0)

    # #  varEst = 1/m/(m-1)/(m-2)    * ( sum(Kxd,1)*sum(Kxd,2) - sum(sum(Kxd.^2)))  ...
    # v[0] = 1.0 / m / (m - 1) / (m - 2) * (torch.dot(Kxd0_red, Kxd0_red) - torch.sum(Kxd ** 2))
    # #           -  (  1/m/(m-1)   *  sum(sum(Kxd))  )^2 ...
    # v[1] = -(1.0 / m / (m - 1) * Kxd_sum) ** 2
    # #           -  2/m/(m-1)/n     *  sum(Kxd,1) * sum(Kxy,2)  ...
    # v[2] = -2.0 / m / (m - 1) / n * torch.dot(Kxd0_red, Kxy1)
    # #           +  2/m^2/(m-1)/n   * sum(sum(Kxd))*sum(sum(Kxy)) ...
    # v[3] = 2.0 / (m ** 2) / (m - 1) / n * Kxd_sum * Kxy_sum
    # #           +  1/(n)/(n-1)/(n-2) * ( sum(Kyd,1)*sum(Kyd,2) - sum(sum(Kyd.^2)))  ...
    # v[4] = 1.0 / n / (n - 1) / (n - 2) * (torch.dot(Kyd0_red, Kyd0_red) - torch.sum(Kyd ** 2))
    # #           -  ( 1/n/(n-1)   * sum(sum(Kyd))  )^2	...
    # v[5] = -(1.0 / n / (n - 1) * Kyd_sum) ** 2
    # #           -  2/n/(n-1)/m     * sum(Kyd,1) * sum(Kxy',2)  ...
    # v[6] = -2.0 / n / (n - 1) / m * torch.dot(Kyd0_red, Kyx1)
    #
    # #           +  2/n^2/(n-1)/m  * sum(sum(Kyd))*sum(sum(Kxy)) ...
    # v[7] = 2.0 / (n ** 2) / (n - 1) / m * Kyd_sum * Kxy_sum
    # #           +  1/n/(n-1)/m   * ( sum(Kxy',1)*sum(Kxy,2) -sum(sum(Kxy.^2))  ) ...
    # v[8] = 1.0 / n / (n - 1) / m * (torch.dot(Kxy1, Kxy1) - Kxy2_sum)
    # #           - 2*(1/n/m        * sum(sum(Kxy))  )^2 ...
    # v[9] = -2.0 * (1.0 / n / m * Kxy_sum) ** 2
    # #           +   1/m/(m-1)/n   *  ( sum(Kxy,1)*sum(Kxy',2) - sum(sum(Kxy.^2)))  ;
    # v[10] = 1.0 / m / (m - 1) / n * (torch.dot(Kyx1, Kyx1) - Kxy2_sum)
    #
    # # %additional low order correction made to some terms compared with ICLR submission
    # # %these corrections are of the same order as the 2nd order term and will
    # # %be unimportant far from the null.
    #
    # #   %Eq. 13 p. 11 ICLR 2016. This uses ONLY first order term
    # #   varEst = 4*(m-2)/m/(m-1) *  varEst  ;
    # varEst1st = 4.0 * (m - 2) / m / (m - 1) * torch.sum(v)

    # Kxyd = Kxy - torch.diag(torch.diag(Kxy))
    # #   %Eq. 13 p. 11 ICLR 2016: correction by adding 2nd order term
    # #   varEst2nd = 2/m/(m-1) * 1/n/(n-1) * sum(sum( (Kxd + Kyd - Kxyd - Kxyd').^2 ));
    # varEst2nd = 2.0 / m / (m - 1) * 1 / n / (n - 1) * torch.sum((Kxd + Kyd - Kxyd - Kxyd.transpose(0, 1)) ** 2)
    #
    # #   varEst = varEst + varEst2nd;
    # varEst = varEst2nd#varEst1st +
    #
    # #   %use only 2nd order term if variance estimate negative
    # if varEst < 0:
    #     varEst = varEst2nd
    return mmd2, varEst


def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, is_smooth=True, is_mixture=False, beta=None, is_var_computed=True, use_1sample_U=True):
    """
    X: nxd numpy array
    Y: nxd numpy array
    k: a Kernel object
    is_var_computed: if True, compute the variance. If False, return None.
    use_1sample_U: if True, use one-sample U statistic for the cross term
      i.e., k(X, Y).

    Code based on Arthur Gretton's Matlab implementation for
    Bounliphone et. al., 2016.

    return (MMD^2, var[MMD^2]) under H1
    """
    X = Fea[0:len_s, :]
    Y = Fea[len_s:, :]
    X_org = Fea_org[0:len_s, :]
    Y_org = Fea_org[len_s:, :]

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_mixture:
        if is_smooth:
            Kx = torch.exp(-Dxx / sigma0 - Dxx_org / sigma)
            Ky = torch.exp(-Dyy / sigma0 - Dyy_org / sigma)
            Kxy = torch.exp(-Dxy / sigma0 - Dxy_org / sigma)
        else:
            Kx = torch.exp(-Dxx / sigma0)
            Ky = torch.exp(-Dyy / sigma0)
            Kxy = torch.exp(-Dxy / sigma0)
    else:
        if is_smooth:
            Kx = torch.exp(-Dxx / sigma0 - Dxx_org / sigma)
            Ky = torch.exp(-Dyy / sigma0 - Dyy_org / sigma)
            Kxy = torch.exp(-Dxy / sigma0 - Dxy_org / sigma)
        else:
            Kx = torch.exp(-Dxx / sigma0)
            Ky = torch.exp(-Dyy / sigma0)
            Kxy = torch.exp(-Dxy / sigma0)

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)


def MMD_L(N1, N2, device, dtype):  # , kernel_mul=2.0, kernel_num=5, fix_sigma=None
    Lii = torch.ones(N1, N1, device=device, dtype=dtype) / N1 / N1
    Ljj = torch.ones(N2, N2, device=device, dtype=dtype) / N2 / N2
    Lij = -1 * torch.ones(N1, N2, device=device, dtype=dtype) / N1 / N2
    Lu = torch.cat([Lii, Lij], 1)
    Ll = torch.cat([Lij.transpose(0, 1), Ljj], 1)
    LM = torch.cat([Lu, Ll], 0)
    return LM

def TST_MMD_median(Fea, N_per, LM, N1, alpha, device, dtype):
    mmd_vector = np.zeros(N_per)
    mmd_value = get_item(MyMMD(Fea, LM, N1, is_median = True),is_cuda)
    count = 0
    Fea = get_item(Fea,is_cuda)
    for i in range(N_per):
        Fea_per = np.random.permutation(Fea)
        Fea_per = MatConvert(Fea_per, device, dtype)
        mmd_vector[i] = get_item(MyMMD(Fea_per, LM, N1, is_median = True),is_cuda)
        if mmd_vector[i] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_mmd_vector[np.int(np.ceil(N_per * alpha))]
    return h, threshold, mmd_value.item()

def TST_MMD_adaptive_bandwidth(Fea, N_per, LM, N1, Fea_org, sigma, sigma0, alpha, device, dtype):
    mmd_vector = np.zeros(N_per)
    # mmd_value = MyMMD(Fea, LM, N1).detach().numpy()
    mmd_value = get_item(MMDu(Fea, N1, Fea_org, sigma, sigma0)[0],is_cuda)
    count = 0
    Fea = get_item(Fea,is_cuda)
    for i in range(N_per):
        Fea_per = np.random.permutation(Fea)
        Fea_per = MatConvert(Fea_per, device, dtype)
        # mmd_vector[i] = MyMMD(Fea_per, LM, N1).detach().numpy()
        mmd_vector[i] = get_item(MMDu(Fea_per, N1, Fea_org, sigma, sigma0)[0],is_cuda)
        if mmd_vector[i] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_mmd_vector[np.int(np.ceil(N_per * alpha))]
    return h, threshold, mmd_value.item()

def TST_MMD_u(Fea, N_per, LM, N1, Fea_org, sigma, alpha, device, dtype):
    mmd_vector = np.zeros(N_per)
    # mmd_value = MyMMD(Fea, LM, N1).detach().numpy()
    mmd_value = get_item(MMDu(Fea, N1, Fea_org, sigma)[0],is_cuda)
    count = 0
    Fea = get_item(Fea,is_cuda)
    for i in range(N_per):
        Fea_per = np.random.permutation(Fea)
        Fea_per = MatConvert(Fea_per, device, dtype)
        # mmd_vector[i] = MyMMD(Fea_per, LM, N1).detach().numpy()
        mmd_vector[i] = get_item(MMDu(Fea_per, N1, Fea_org, sigma)[0],is_cuda)
        if mmd_vector[i] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_mmd_vector[np.int(np.ceil(N_per * alpha))]
    return h, threshold, mmd_value.item()

def TST_MMD_b(Fea, N_per, LM, N1, alpha, device, dtype):
    mmd_vector = np.zeros(N_per)
    mmd_value = get_item(MyMMD(Fea, LM, N1),is_cuda)
    count = 0
    Fea = get_item(Fea,is_cuda)
    for i in range(N_per):
        Fea_per = np.random.permutation(Fea)
        Fea_per = MatConvert(Fea_per, device, dtype)
        mmd_vector[i] = get_item(MyMMD(Fea_per, LM, N1),is_cuda)
        if mmd_vector[i] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_mmd_vector[np.int(np.ceil(N_per * alpha))]
    return h, threshold, mmd_value.item()

def TST_ME(Fea, N1, alpha, is_train, test_locs, gwidth, J = 1, seed = 15):
    Fea = get_item(Fea,is_cuda)
    tst_data = data.TSTData(Fea[0:N1,:], Fea[N1:,:])
    h = 0
    if is_train:
        op = {
            'n_test_locs': J,  # number of test locations to optimize
            'max_iter': 300,  # maximum number of gradient ascent iterations
            'locs_step_size': 1.0,  # step size for the test locations (features)
            'gwidth_step_size': 0.1,  # step size for the Gaussian width
            'tol_fun': 1e-4,  # stop if the objective does not increase more than this.
            'seed': seed + 5,  # random seed
        }
        test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tst_data, alpha, **op)
        return test_locs, gwidth
    else:
        met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
        test_result = met_opt.perform_test(tst_data)
        if test_result['h0_rejected']:
            h = 1
        return h

def TST_SCF(Fea, N1, alpha, is_train, test_freqs, gwidth, J = 1, seed = 15):
    Fea = get_item(Fea,is_cuda)
    tst_data = data.TSTData(Fea[0:N1,:], Fea[N1:,:])
    h = 0
    if is_train:
        op = {'n_test_freqs': J, 'seed': seed, 'max_iter': 300,
              'batch_proportion': 1.0, 'freqs_step_size': 0.1,
              'gwidth_step_size': 0.01, 'tol_fun': 1e-4}
        test_freqs, gwidth, info = tst.SmoothCFTest.optimize_freqs_width(tst_data, alpha, **op)
        return test_freqs, gwidth
    else:
        scf_opt = tst.SmoothCFTest(test_freqs, gwidth, alpha=alpha)
        test_result = scf_opt.perform_test(tst_data)
        if test_result['h0_rejected']:
            h = 1
        return h


dtype = torch.float
device = torch.device("cuda:0")

N_per = 50 # permutation times
alpha = 0.05 # test threshold
mu_mx = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
sigma_mx_1 = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2 = np.zeros([9,2,2])
sigma = 1.5
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
n = 50
s1 = np.zeros([9 * n, 2])
s2 = np.zeros([9 * n, 2])
x_in = 2
H = 5
x_out = 5
learning_rate = 0.05
K = 1
Results = np.zeros([6,K])
# random.seed(1102)

for kk in range(K):

    if is_cuda:
        model_u = ModelLatentF(x_in, H, x_out).cuda()
        model_u1 = ModelLatentF(x_in, H, x_out).cuda()
        model_b = ModelLatentF(x_in, H, x_out).cuda()
    else:
        model_u = ModelLatentF(x_in, H, x_out)
        model_u1 = ModelLatentF(x_in, H, x_out)
        model_b = ModelLatentF(x_in, H, x_out)

    # optimizer = torch.optim.SGD(list(modellantentF.parameters()), lr=learning_rate) #
    optimizer_u = torch.optim.Adam(list(model_u.parameters()), lr=learning_rate)
    optimizer_b = torch.optim.Adam(list(model_b.parameters()), lr=learning_rate)
    optimizer_u1 = torch.optim.Adam(list(model_u1.parameters()), lr=learning_rate)

    for i in range(9):
        # np.random.seed(seed=1102)
        s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
    for i in range(9):
        # np.random.seed(seed=1103 + i)
        s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n) # sigma_mx_2[i]
    # plt.plot(s1[:, 0], s1[:, 1], 'ro')
    # plt.plot(s2[:, 0], s2[:, 1], 'bo')
    # plt.show()
    if kk==0:
        s1_o = s1
        s2_o = s2
    S = np.concatenate((s1, s2), axis=0)
    S = MatConvert(S, device, dtype)
    # S.requires_grad = True
    # optimizer_us = torch.optim.Adam([S], lr=learning_rate)
    N1 = 9*n
    N2 = 9*n
    LM = MMD_L(N1, N2, device, dtype)
    v = torch.div(torch.ones([N1+N2, N1+N2], dtype=torch.float, device=device), (N1+N2)*1.0)
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for t in range(500):
        modelu_output = model_u(S)
        TEMP = MMDu(modelu_output, N1, S, sigma)
        mmd_value_temp = -1 * TEMP[0]
        mmd_std_temp = torch.sqrt(TEMP[1])
        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
        optimizer_u.zero_grad()
        # K = torch.exp(-Pdist2(modelu_output, modelu_output) / 0.1)
        # K.backward(v, create_graph=True)
        # optimizer_u.zero_grad()
        # loss1 = ((S.grad ** 2).sum(1)).mean()
        # loss = STAT_u + 0.01*loss1
        # loss.backward()
        STAT_u.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u.step()
        print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ", -1 * STAT_u.item()) #,"Reg: ", loss1.item()
    h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, LM, N1, S, sigma, alpha, device, dtype)
    print("h:", h_u, "Threshold:", threshold_u, "MMD_value:", mmd_value_u)
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for t in range(500):
        modelu1_output = model_u1(S)
        TEMP = MMDu(modelu1_output, N1, S, sigma)
        mmd_value_temp = -1 * TEMP[0]
        mmd_std_temp = torch.sqrt(TEMP[1])
        STAT_u1 = torch.div(mmd_value_temp, 1.0)
        print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ", -1 * STAT_u1.item())
        optimizer_u1.zero_grad()
        STAT_u1.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer_u1.step()
    h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, LM, N1, S, sigma, alpha, device, dtype)
    print("h:", h_u1, "Threshold:", threshold_u1, "MMD_value:", mmd_value_u1)
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    # for t in range(500):
    #     STAT_b = -1*MyMMD(model_b(S), LM, N1)
    #     # print(mmd_value_temp,mmd_std_temp)
    #     print("mmd: ", -1 * STAT_b.item(), "Statistic: ", -1 * STAT_b.item())
    #     optimizer_b.zero_grad()
    #     STAT_b.backward(retain_graph=True)
    #     # Update weights using gradient descent
    #     optimizer_b.step()
    # h_b, threshold_b, mmd_value_b = TST_MMD_b(model_b(S), N_per, LM, N1, alpha, device, dtype)
    # print("h:", h_b, "Threshold:", threshold_b, "MMD_value:", mmd_value_b)
    sigma0 = torch.rand([1]).to(device, dtype)
    sigma0.requires_grad = True
    optimizer_sigma0 = torch.optim.Adam([sigma0], lr=learning_rate)
    for t in range(500):
        TEMP = MMDu(S, N1, S, sigma, sigma0**2, is_smooth=False)
        mmd_value_temp = -1 * TEMP[0]
        mmd_std_temp = torch.sqrt(TEMP[1])
        STAT_adaptive = torch.div(mmd_value_temp, mmd_std_temp)
        print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ", -1 * STAT_adaptive.item())
        optimizer_sigma0.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        # Update sigma0 using gradient descent
        optimizer_sigma0.step()
    h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, LM, N1, S, sigma, sigma0, alpha, device, dtype)
    print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)
    np.random.seed(seed=1102)
    test_locs_ME, gwidth_ME = TST_ME(S, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=10, seed=15)
    np.random.seed(seed=1102)
    test_freqs_SCF, gwidth_SCF = TST_SCF(S, N1, alpha, is_train=True, test_freqs=1, gwidth=1, J=10, seed=15)

    S_m = get_item(model_u(S),is_cuda)
    s1_m = S_m[0:9*n, :]
    s2_m = S_m[9*n:, :]
    # plt.plot(s1_m[:, 0], s1_m[:, 1], 'ro')
    # plt.show()
    # plt.plot(s2_m[:, 0], s2_m[:, 1], 'bo')
    # plt.show()
    N = 100
    N_f = 100.0
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
#     np.random.seed(1102)
#     for k in range(N):
#         for i in range(9):
#             np.random.seed(seed=1102 * kk * k + i)
#             s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
#         for i in range(9):
#             np.random.seed(seed=819 * kk * k + i)
#             s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n) # sigma_mx_2[i]
#         S = np.concatenate((s1, s2), axis=0)
#         S = MatConvert(S, device, dtype)
#         h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, LM, N1, S, sigma, alpha, device, dtype)
#         h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, LM, N1, S, sigma, alpha, device, dtype)
# #         h_b, threshold_b, mmd_value_b = TST_MMD_b(model_b(S), N_per, LM, N1, alpha, device, dtype)
#         h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, LM, N1, S, sigma, sigma0, alpha, device, dtype)
#         h_m, threshold_m, mmd_value_m = TST_MMD_b(S, N_per, LM, N1, alpha, device, dtype)
#         h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=1, seed=15)
#         h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_ME, J=1, seed=15)
#         print("h_u:", h_u, "Threshold_u:", threshold_u, "MMD_value_u:", mmd_value_u)
#         print("h_u1:", h_u1, "Threshold_u1:", threshold_u1, "MMD_value_u1:", mmd_value_u1)
#         # print("h_b:", h_b, "Threshold_b:", threshold_b, "MMD_value_b:", mmd_value_b)
#         print("h_adaptive:", h_adaptive, "Threshold_adaptive:", threshold_adaptive, "MMD_value_adaptive:", mmd_value_adaptive)
#         print("h_m:", h_m, "Threshold_m:", threshold_m, "MMD_value_m:", mmd_value_m)
#         print("h_ME:", h_ME)
#         print("h_SCF:", h_SCF)
#         H_u[k] = h_u
#         T_u[k] = threshold_u
#         M_u[k] = mmd_value_u
#         H_u1[k] = h_u1
#         T_u1[k] = threshold_u1
#         M_u1[k] = mmd_value_u1
#         H_adaptive[k] = h_adaptive
#         T_adaptive[k] = threshold_adaptive
#         M_adaptive[k] = mmd_value_adaptive
#         # H_b[k] = h_b
#         # T_b[k] = threshold_b
#         # M_b[k] = mmd_value_b
#         H_m[k] = h_m
#         T_m[k] = threshold_m
#         M_m[k] = mmd_value_m
#         H_ME[k] = h_ME
#         H_SCF[k] = h_SCF
#     print("Reject rate_u: ", H_u.sum()/N_f,"Reject rate_u1: ", H_u1.sum()/N_f,"Reject rate_adaptive: ", H_adaptive.sum()/N_f,"Reject rate_ME: ", H_ME.sum()/N_f,"Reject rate_SCF: ", H_SCF.sum()/N_f,"Reject rate_m: ", H_m.sum()/N_f)
#     Results[0, kk] = H_u.sum() / N_f
#     Results[1, kk] = H_u1.sum() / N_f
#     Results[2, kk] = H_adaptive.sum() / N_f
#     Results[3, kk] = H_m.sum() / N_f
#     Results[4, kk] = H_ME.sum() / N_f
#     Results[5, kk] = H_SCF.sum() / N_f
#     print(Results,Results.mean(1))
# f = open('Results_'+str(n)+'_H1.pckl', 'wb')
# pickle.dump([Results], f)
# f.close()

step = 0.05
x = np.arange(-3,3,step)
y = np.arange(-3,3,step)
X,Y = np.meshgrid(x,y)

xx = np.zeros([1,2])
ZZ = np.zeros([9,120,120])
for i in range(120):
    for j in range(120):
        xx[0][0] = X[i,j]
        xx[0][1] = Y[i,j]
        xxx = MatConvert(xx, device, dtype)
        for kk in range(9):
            yyy = MatConvert(np.array([mu_mx[kk]]), device, dtype)
            ZZ[kk,i,j] = (torch.exp(-Pdist2(model_u(xxx),model_u(yyy))/0.1 - Pdist2(xxx,yyy)/sigma)).cpu().detach().numpy()
f = open('./Contour_results'+str(n)+'_3.pckl', 'wb')
pickle.dump([X,Y,ZZ,s1_o,s2_o], f)
f.close()