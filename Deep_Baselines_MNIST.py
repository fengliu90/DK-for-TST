import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import pickle
from TST_utils_HD import MatConvert, Pdist2, MMDu, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_SCF, TST_C2ST_D, TST_LCE_D

os.makedirs("images", exist_ok=True)
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n", type=int, default=100, help="number of samples in one set")
opt = parser.parse_args()
print(opt)
dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 100))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
dataloader_FULL = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=60000,
    shuffle=True,
)
for i, (imgs, Labels) in enumerate(dataloader_FULL):
    data_all = imgs
    label_all = Labels

dataloader_FULL_te = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=10000,
    shuffle=True,
)
for i, (imgs, Labels) in enumerate(dataloader_FULL_te):
    data_all_te = imgs
    label_all_te = Labels

sigma = 2*32*32
sigma0_u = 0.005 #0.25
N_per = 100
alpha = 0.05
N1 = opt.n
K = 10
Results = np.zeros([6,K])

# Loss function
adversarial_loss = torch.nn.CrossEntropyLoss()
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
for kk in range(K):
    torch.manual_seed(kk * 19 + N1)
    torch.cuda.manual_seed(kk * 19 + N1)
    np.random.seed(seed=1102 * (kk + 10) + N1)
    # Initialize generator and discriminator
    featurizer = Featurizer()
    discriminator = Discriminator()

    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2*32*32), device, dtype)  # d = 3,5 ??
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = True
    print(epsilonOPT.item())

    if cuda:
        featurizer.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # generate data
    np.random.seed(seed=819 * (kk + 9) + N1)
    train_data = []
    ind_M_all = np.arange(4000)
    ind_M_tr = np.random.choice(4000, N1, replace=False)
    ind_M_te = np.delete(ind_M_all,ind_M_tr)
    for i in ind_M_tr:
       train_data.append([data_all[i], label_all[i]])

    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    Fake_MNIST = pickle.load(open('./Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
    ind_all = np.arange(4000)
    ind_tr = np.random.choice(4000, N1, replace=False)
    ind_te = np.delete(ind_all,ind_tr)
    Fake_MNIST_tr = torch.from_numpy(Fake_MNIST[0][ind_tr])
    Fake_MNIST_te = torch.from_numpy(Fake_MNIST[0][ind_te])

    # Optimizers
    optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            if True:
                ind = np.random.choice(N1, imgs.shape[0], replace=False)
                Fake_imgs = Fake_MNIST_tr[ind]
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                Fake_imgs = Variable(Fake_imgs.type(Tensor))
                X = torch.cat([real_imgs, Fake_imgs], 0)
                Y = torch.cat([valid, fake], 0).squeeze().long()
                # ------------------------------
                #  Train deep network for MMD-D
                # ------------------------------

                optimizer_F.zero_grad()

                modelu_output = featurizer(X)

                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))  # 10 ** (-10)#
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2

                TEMP = MMDu(modelu_output, imgs.shape[0], X.view(X.shape[0],-1), sigma, sigma0_u, ep)
                mmd_value_temp = -1 * (TEMP[0])  # 10**(-8)
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))  # 0.1
                if mmd_std_temp.item() == 0:
                    print('error std!!')
                if np.isnan(mmd_std_temp.item()):
                    print('error mmd!!')
                f_loss = torch.div(mmd_value_temp, mmd_std_temp)  # - r_full / (N1+N2)
                f_loss.backward()
                optimizer_F.step()
                # J_star_u[t] = f_loss.item()

                # ------------------------------------------
                #  Train deep network for C2ST-S and C2ST-L
                # ------------------------------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                d_loss = adversarial_loss(discriminator(X), Y)
                d_loss.backward()
                optimizer_D.step()
                if (epoch+1) % 100 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Stat: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), -f_loss.item())
                    )

                batches_done = epoch * len(dataloader) + i
            else:
                break

    s1 = data_all[ind_M_tr]
    s2 = Variable(Fake_MNIST_tr.type(Tensor))
    S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
    Sv = S.view(2*N1,-1)
    h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device, dtype)
    h_C2ST_S, threshold_C2ST_S, s_C2ST_S = TST_C2ST_D(S, N1, N_per, alpha, discriminator, device, dtype)
    h_C2ST_L, threshold_C2ST_L, s_C2ST_L = TST_LCE_D(S, N1, N_per, alpha, discriminator, device, dtype)

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    Dxy = Pdist2(Sv[:N1, :], Sv[N1:, :])
    sigma0 = Dxy.median()
    sigma0.requires_grad = True
    optimizer_sigma0 = torch.optim.Adam([sigma0], lr=0.0005)
    for t in range(2000):
        TEMPa = MMDu(Sv, N1, Sv, sigma, sigma0, is_smooth=False)
        mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
        mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
        if mmd_std_tempa.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_tempa.item()):
            print('error!!')
        STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        optimizer_sigma0.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        # Update sigma0 using gradient descent
        optimizer_sigma0.step()
        if t % 100 == 0:
            print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                  -1 * STAT_adaptive.item())
    h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(Sv, N_per, N1, Sv, sigma, sigma0, alpha,
                                                                                    device, dtype)
    print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)

    np.random.seed(seed=1102)
    test_locs_ME, gwidth_ME = TST_ME(Sv, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=5, seed=15)
    h_ME = TST_ME(Sv, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=5, seed=15)

    np.random.seed(seed=1102)
    test_freqs_SCF, gwidth_SCF = TST_SCF(Sv, N1, alpha, is_train=True, test_freqs=1, gwidth=1, J=5, seed=15)
    h_SCF = TST_SCF(Sv, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=5, seed=15)

    ep_OPT[kk] = ep.item()
    s_OPT[kk] = sigma.item()
    s0_OPT[kk] = sigma0_u.item()

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
    H_ME = np.zeros(N)
    H_SCF = np.zeros(N)
    H_C2ST_S = np.zeros(N)
    Tu_C2ST_S = np.zeros(N)
    S_C2ST_S = np.zeros(N)
    H_C2ST_L = np.zeros(N)
    Tu_C2ST_L = np.zeros(N)
    S_C2ST_L = np.zeros(N)
    np.random.seed(1102)
    count_u = 0
    count_adp = 0
    count_ME = 0
    count_SCF = 0
    count_C2ST_S = 0
    count_C2ST_L = 0
    for k in range(N):
        # generate data
        np.random.seed(seed=1102 * (k + 1) + N1)
        ind_M = np.random.choice(len(ind_M_te), N1, replace=False)
        s1 = data_all[ind_M_te[ind_M]]
        np.random.seed(seed=819 * (k + 3) + N1)
        ind_F = np.random.choice(len(Fake_MNIST_te), N1, replace=False)
        s2 = Variable(Fake_MNIST_te[ind_F].type(Tensor))
        S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
        Sv = S.view(2 * N1, -1)

        h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device, dtype)
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(Sv, N_per, N1, Sv, sigma, sigma0, alpha, device, dtype)
        h_ME = TST_ME(Sv, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=10, seed=15)
        h_SCF = TST_SCF(Sv, N1, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=10, seed=15)
        H_C2ST_S[k], Tu_C2ST_S[k], S_C2ST_S[k] = TST_C2ST_D(S, N1, N_per, alpha, discriminator, device, dtype)
        H_C2ST_L[k], Tu_C2ST_L[k], S_C2ST_L[k] = TST_LCE_D(S, N1, N_per, alpha, discriminator, device, dtype)

        count_u = count_u + h_u
        count_adp = count_adp + h_adaptive
        count_ME = count_ME + h_ME
        count_SCF = count_SCF + h_SCF
        count_C2ST_S = count_C2ST_S + int(H_C2ST_S[k])
        count_C2ST_L = count_C2ST_L + int(H_C2ST_L[k])
        print("MMD-DK:", count_u, "MMD-OPT:", count_adp, "MMD-ME:", count_ME, "SCF:", count_SCF, "C2ST_S: ",
              count_C2ST_S, "C2ST_L: ", count_C2ST_L)
        H_u[k] = h_u
        T_u[k] = threshold_u
        M_u[k] = mmd_value_u
        H_adaptive[k] = h_adaptive
        T_adaptive[k] = threshold_adaptive
        M_adaptive[k] = mmd_value_adaptive
        H_ME[k] = h_ME
        H_SCF[k] = h_SCF
    print("Reject rate_u: ", H_u.sum() / N_f, "Reject rate_C2ST-L: ", H_C2ST_L.sum() / N_f, "Reject rate_C2ST-S: ",
          H_C2ST_S.sum() / N_f, "Reject rate_adaptive: ",
          H_adaptive.sum() / N_f, "Reject rate_ME: ", H_ME.sum() / N_f, "Reject rate_SCF: ", H_SCF.sum() / N_f)

    Results[0, kk] = H_u.sum() / N_f
    Results[1, kk] = H_C2ST_L.sum() / N_f
    Results[2, kk] = H_C2ST_S.sum() / N_f
    Results[3, kk] = H_adaptive.sum() / N_f
    Results[4, kk] = H_ME.sum() / N_f
    Results[5, kk] = H_SCF.sum() / N_f
    print(Results, Results.mean(1))