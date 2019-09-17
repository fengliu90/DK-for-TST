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
from TST_utils import MatConvert, MMDu, MMD_L, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_SCF

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
        )

    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.xavier_normal_(m.bias)
        # m.bias = torch.randn(m.bias.size())

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
# n_list = [10,20,30,50,70,80,90,100]
n_list = [20]
# n = 35
x_in = 2
H = 30  # 3 for lower type-I error and test power
x_out = 30
# learning_rate = 0.1 #SGD
learning_rate = 0.0005#0.0005 #Adam
learning_rate_C2ST = 0.00005
K = 10
Results = np.zeros([6,K])
reg = 0 * 0.2**2
N = 100
N_f = 100.0
sigma = 1.5 # 1.5
# sigma0_u_np = np.array(0.1)
# sigma0_u = torch.from_numpy(sigma0_u_np).to(device, dtype)
# sigma0_u.requires_grad = True
sigma0_u = np.sqrt(0.01)  # 0.1



mu_mx = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
sigma_mx_1 = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2 = np.zeros([9,2,2])
J_star_u = np.zeros([len(n_list),500])
J_star_adp = np.zeros([len(n_list),500])
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
    s1 = np.zeros([9 * n, 2])
    s2 = np.zeros([9 * n, 2])
    N1 = 9 * n
    N2 = 9 * n
    batch_size = 64
    N_epoch = int(500*18*n/batch_size)
    threshold_C2ST = norm.ppf(0.5 + alpha / 2, loc=0.5, scale=np.sqrt(1 / 18 / n)) - 0.5
    for kk in range(K):
        # torch.manual_seed(kk*19+n # n=40
        # torch.cuda.manual_seed(kk*19+n)
        if is_cuda:
            model_u = ModelLatentF(x_in, H, x_out).cuda()
            model_C2ST = ModelLatentF(x_in, H, x_out).cuda()
            # model_u.apply(init_normal)
            # model_u1 = ModelLatentF(x_in, H, x_out).cuda()
            # model_b = ModelLatentF(x_in, H, x_out).cuda()
        else:
            model_u = ModelLatentF(x_in, H, x_out)
            model_C2ST = ModelLatentF(x_in, H, x_out)
            # model_u.apply(init_normal)
            # model_u1 = ModelLatentF(x_in, H, x_out)
            # model_b = ModelLatentF(x_in, H, x_out)

        w_C2ST = torch.randn([x_out,2]).to(device, dtype)
        b_C2ST = torch.randn([1,2]).to(device, dtype)
        w_C2ST.requires_grad = True
        b_C2ST.requires_grad = True

        optimizer_u = torch.optim.Adam(list(model_u.parameters()), lr=learning_rate)
        optimizer_C2ST = torch.optim.Adam(list(model_C2ST.parameters())+[w_C2ST]+[b_C2ST], lr=learning_rate_C2ST)
        # optimizer_b = torch.optim.Adam(list(model_b.parameters()), lr=learning_rate)
        # optimizer_u1 = torch.optim.Adam(list(model_u1.parameters()), lr=learning_rate)

        criterion = torch.nn.CrossEntropyLoss()
        f = torch.nn.Softmax()
        y = (torch.cat((torch.zeros(N1,1),torch.ones(N2,1)),0)).squeeze(1).to(device, dtype).long()

        for i in range(9):
            np.random.seed(seed=1102*kk + i + n)
            s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        for i in range(9):
            np.random.seed(seed=819*kk+1 + i + n)
            s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n) # sigma_mx_2[i]
        if kk==0:
            s1_o = s1
            s2_o = s2
        S = np.concatenate((s1, s2), axis=0)
        S = MatConvert(S, device, dtype)
        # C2ST
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        dataset = torch.utils.data.TensorDataset(S, y)
        dataloader_C2ST = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        len_dataloader = len(dataloader_C2ST)
        for epoch in range(N_epoch):
            data_iter = iter(dataloader_C2ST)
            tt = 0
            while tt < len_dataloader:
                # training model using source data
                data_source = data_iter.next()
                S_b, y_b = data_source
                output_b = model_C2ST(S_b).mm(w_C2ST) + b_C2ST
                loss_C2ST = criterion(output_b, y_b)
                optimizer_C2ST.zero_grad()
                loss_C2ST.backward(retain_graph=True)
                # Update sigma0 using gradient descent
                optimizer_C2ST.step()
                tt=tt+1
            if epoch % 100 ==0:
                print(loss_C2ST.item())

        output = f(model_C2ST(S).mm(w_C2ST)+b_C2ST)
        pred = output.max(1, keepdim=True)[1]
        acc_C2ST_train = pred.eq(y.data.view_as(pred)).cpu().sum().item()*1.0/((N1+N2)*1.0)
        STAT = abs(acc_C2ST_train - 0.5)
        if STAT<threshold_C2ST:
            h_C2ST = 0
        else:
            h_C2ST = 1

        LM = MMD_L(N1, N2, device, dtype)
        v = torch.div(torch.ones([N1+N2, N1+N2], dtype=torch.float, device=device), (N1+N2)*1.0)
        # for t in range(500):
        #     modelu1_output = model_u1(S)
        #     TEMP = MMDu(modelu1_output, N1, S, sigma)
        #     mmd_value_temp = -1 * TEMP[0]
        #     mmd_std_temp = torch.sqrt(TEMP[1])
        #     STAT_u1 = torch.div(mmd_value_temp, 1.0)
        #     print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ", -1 * STAT_u1.item())
        #     optimizer_u1.zero_grad()
        #     STAT_u1.backward(retain_graph=True)
        #     # Update weights using gradient descent
        #     optimizer_u1.step()
        # h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, LM, N1, S, sigma, alpha, device, dtype)
        # print("h:", h_u1, "Threshold:", threshold_u1, "MMD_value:", mmd_value_u1)
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        sigma0 = torch.rand([1]).to(device, dtype)
        sigma0.requires_grad = True
        optimizer_sigma0 = torch.optim.Adam([sigma0], lr=learning_rate)
        for t in range(500):
            TEMPa = MMDu(S, N1, S, sigma, sigma0**2 + reg, is_smooth=False)
            mmd_value_tempa = -1 * TEMPa[0]
            mmd_std_tempa = torch.sqrt(TEMPa[1]+0.001)
            STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
            J_star_adp[count, t] = STAT_adaptive.item()
            # print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ", -1 * STAT_adaptive.item())
            optimizer_sigma0.zero_grad()
            STAT_adaptive.backward(retain_graph=True)
            # Update sigma0 using gradient descent
            optimizer_sigma0.step()
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, LM, N1, S, sigma, sigma0**2 + reg, alpha, device, dtype)
        # print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)

        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        for t in range(500):
            modelu_output = model_u(S)
            TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u ** 2)
            mmd_value_temp = -1 * TEMP[0]
            mmd_std_temp = torch.sqrt(TEMP[1]+0.001)
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
            J_star_u[count, t] = STAT_u.item()
            optimizer_u.zero_grad()
            STAT_u.backward(retain_graph=True)
            # Update weights using gradient descent
            optimizer_u.step()
            # print("mmd: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
            #       -1 * STAT_u.item())  # ,"Reg: ", loss1.item()
        h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, LM, N1, S, sigma, sigma0_u ** 2, alpha, device,
                                                  dtype)
        # print("h:", h_u, "Threshold:", threshold_u, "MMD_value:", mmd_value_u)

        np.random.seed(seed=1102)
        test_locs_ME, gwidth_ME = TST_ME(S, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=5, seed=15)
        h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=5, seed=15)
        print("h:", h_ME, "test_locs_ME:", test_locs_ME, "gwidth_ME:", gwidth_ME)

        np.random.seed(seed=1102)
        test_freqs_SCF, gwidth_SCF = TST_SCF(S, N1, alpha, is_train=True, test_freqs=1, gwidth=1, J=5, seed=15)
        h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=5, seed=15)
        print("h:", h_SCF, "test_freqs_SCF:", test_freqs_SCF, "gwidth_SCF:", gwidth_SCF)
        # S_m = get_item(model_u(S),is_cuda)
        # s1_m = S_m[0:9*n, :]
        # s2_m = S_m[9*n:, :]

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
        np.random.seed(1102)
        count_u = 0
        count_adp = 0
        count_ME = 0
        count_SCF = 0
        count_C2ST = 0
        for k in range(N):
            for i in range(9):
                np.random.seed(seed=1102 * k + 2*kk + i + n)
                s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
            for i in range(9):
                np.random.seed(seed=819 * (k+1) + 2*kk + i + n)
                s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n) # sigma_mx_2[i]
            S = np.concatenate((s1, s2), axis=0)
            S = MatConvert(S, device, dtype)
            h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, LM, N1, S, sigma, sigma0_u**2, alpha, device, dtype)
            # h_u1, threshold_u1, mmd_value_u1 = TST_MMD_u(model_u1(S), N_per, LM, N1, S, sigma, alpha, device, dtype)
    #         h_b, threshold_b, mmd_value_b = TST_MMD_b(model_b(S), N_per, LM, N1, alpha, device, dtype)
            h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S, N_per, LM, N1, S, sigma, sigma0**2 + reg, alpha, device, dtype)
            # h_m, threshold_m, mmd_value_m = TST_MMD_median(S, N_per, LM, N1, alpha, device, dtype)
            h_ME = TST_ME(S, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=1, seed=15)
            h_SCF = TST_SCF(S, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=1, seed=15)
            output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
            pred = output.max(1, keepdim=True)[1]
            acc_C2ST_test = pred.eq(y.data.view_as(pred)).cpu().sum().item() * 1.0 / ((N1 + N2) * 1.0)
            STAT = abs(acc_C2ST_test - 0.5)
            if STAT < threshold_C2ST:
                H_C2ST[k] = 0
            else:
                H_C2ST[k] = 1
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
    f = open('Results_'+str(n)+'_H0.pckl', 'wb')
    pickle.dump([Results,J_star_u,J_star_adp], f)
    f.close()
    # np.random.seed(1102)
    # torch.manual_seed(1102)
    # torch.cuda.manual_seed(1102)
    # del sigma0,model_u
