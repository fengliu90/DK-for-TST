import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
from TST_utils_HD import MatConvert, Pdist2, MMDu, get_item, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_SCF, TST_C2ST_D, MMDu_linear_kernel, TST_MMD_u_linear_kernel,TST_LCE_D


os.makedirs("images", exist_ok=True)
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate") #0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--N1", type=int, default=100, help="number of samples")
opt = parser.parse_args()
print(opt)
dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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
            nn.ReLU())
        self.output_layer = nn.Sequential(
            nn.Linear(100, 2),
            nn.Softmax())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        fea = self.adv_layer(out)
        validity = self.output_layer(fea)

        return validity, fea

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
N1 = opt.N1
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
    featurizer_linear_kernel = Featurizer()
    discriminator = Discriminator()

    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = False
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2*32*32), device, dtype)  # d = 3,5 ??
    sigmaOPT.requires_grad = False
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = False
    print(epsilonOPT.item())

    if cuda:
        featurizer.cuda()
        discriminator.cuda()
        featurizer_linear_kernel.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    # featurizer.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)

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

    # data = np.load('./mnist_16_s10.npz')
    # Fake_MNIST = data.f.arr_0.reshape(10000,1,32,32)
    # ind_shuffle = np.random.choice(10000,10000, replace=False)
    # Fake_MNIST = Fake_MNIST[ind_shuffle]
    # ind_all = np.arange(4000)
    # ind_tr = np.random.choice(4000, N1, replace=False)
    # ind_te = np.delete(ind_all, ind_tr)
    # Fake_MNIST_tr = torch.from_numpy(Fake_MNIST[ind_tr])
    # Fake_MNIST_te = torch.from_numpy(Fake_MNIST[ind_te])

    Fake_MNIST = pickle.load(open('./Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
    ind_all = np.arange(4000)
    ind_tr = np.random.choice(4000, N1, replace=False)
    ind_te = np.delete(ind_all,ind_tr)
    Fake_MNIST_tr = torch.from_numpy(Fake_MNIST[0][ind_tr])
    Fake_MNIST_te = torch.from_numpy(Fake_MNIST[0][ind_te])

    # Optimizers
    optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=opt.lr)
    optimizer_F_linear_kernel = torch.optim.Adam(list(featurizer_linear_kernel.parameters()), lr=opt.lr)
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
                # -----------------
                #  Train Featurizer
                # -----------------

                optimizer_F.zero_grad()

                modelu_output = featurizer(X)


                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))  # 10 ** (-10)#
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2

                TEMP = MMDu(modelu_output, imgs.shape[0], X.view(X.shape[0],-1), sigma, sigma0_u, ep, is_smooth=False)
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

                optimizer_F_linear_kernel.zero_grad()
                modelu_output_linear = featurizer_linear_kernel(X)
                TEMP_l = MMDu_linear_kernel(modelu_output_linear, imgs.shape[0])
                mmd_value_temp_l = -1 * (TEMP_l[0])  # 10**(-8)
                mmd_std_temp_l = torch.sqrt(TEMP_l[1] + 10 ** (-8))  # 0.1
                if mmd_std_temp_l.item() == 0:
                    print('error std!!')
                if np.isnan(mmd_std_temp_l.item()):
                    print('error mmd!!')
                f_loss_l = torch.div(mmd_value_temp_l, mmd_std_temp_l)  # - r_full / (N1+N2)
                f_loss_l.backward()
                optimizer_F_linear_kernel.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                d_loss = adversarial_loss(discriminator(X)[0], Y)
                d_loss.backward()
                optimizer_D.step()
                if (epoch+1) % 100 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Stat: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), -f_loss.item())
                    )

                batches_done = epoch * len(dataloader) + i
                # if batches_done % opt.sample_interval == 0:
                #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            else:
                break



    s1 = data_all[ind_M_tr]
    s2 = Variable(Fake_MNIST_tr.type(Tensor))
    S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
    Sv = S.view(2*N1,-1)
    # G+C
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    S_m_v = discriminator(S)[1].view(2 * N1, -1)

    Dxy = Pdist2(S_m_v[:N1, :], S_m_v[N1:, :])
    # sigma0 = Dxy.median()
    sigma0 = torch.tensor(2*100) * torch.rand([1]).to(device, dtype) #2 * 1024 * torch.rand([1]).to(device, dtype)  # d * torch.rand([1]).to(device, dtype)
    sigma0.requires_grad = True
    optimizer_sigma0 = torch.optim.Adam([sigma0], lr=0.001)
    for t in range(2000):
        TEMPa = MMDu(S_m_v, N1, S_m_v, sigma, sigma0, is_smooth=False)
        mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
        mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))
        if mmd_std_tempa.item() == 0:
            print('error!!')
        if np.isnan(mmd_std_tempa.item()):
            print('error!!')
        STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        # J_star_adp[t] = STAT_adaptive.item()
        optimizer_sigma0.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        # Update sigma0 using gradient descent
        optimizer_sigma0.step()
        if t % 100 == 0:
            print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                  -1 * STAT_adaptive.item())
    # G+C
    h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S_m_v, N_per, N1, S_m_v, sigma, sigma0, alpha,
                                                                                    device, dtype)
    # G+J
    h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device, dtype, is_smooth=False)
    # L+J
    h_u_l, threshold_u_l, mmd_value_u_l = TST_MMD_u_linear_kernel(featurizer(S), N_per, N1, alpha, device, dtype)
    # L+C
    h_C2ST, threshold_C2ST, s_C2ST = TST_LCE_D(S, N1, N_per, alpha, discriminator, device, dtype)

    ep_OPT[kk] = ep.item()
    s_OPT[kk] = sigma.item()
    s0_OPT[kk] = sigma0_u.item()

    N = 100
    N_f = 100.0
    H_u = np.zeros(N)
    T_u = np.zeros(N)
    M_u = np.zeros(N)
    H_u_l = np.zeros(N)
    T_u_l = np.zeros(N)
    M_u_l = np.zeros(N)
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
    count_u_l = 0
    count_C2ST = 0
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
        S_m_v = discriminator(S)[1].view(2 * N1, -1)

        # h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device, dtype)
        h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(S_m_v, N_per, N1, S_m_v, sigma, sigma0, alpha, device, dtype)
        # G+J
        h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device,
                                                  dtype, is_smooth=False)
        # L+J
        h_u_l, threshold_u_l, mmd_value_u_l = TST_MMD_u_linear_kernel(featurizer(S), N_per, N1, alpha, device, dtype)
        # L+C
        H_C2ST[k], Tu_C2ST[k], S_C2ST[k] = TST_LCE_D(S, N1, N_per, alpha, discriminator, device, dtype)
        count_u = count_u + h_u
        count_adp = count_adp + h_adaptive
        count_u_l = count_u_l + h_u_l
        count_C2ST = count_C2ST + int(H_C2ST[k])
        print("L+J:", count_u_l,"G+J:", count_u,"G+C:", count_adp,"L+C:", count_C2ST)
        H_u[k] = h_u
        T_u[k] = threshold_u
        M_u[k] = mmd_value_u
        H_u_l[k] = h_u_l
        T_u_l[k] = threshold_u_l
        M_u_l[k] = mmd_value_u_l
        H_adaptive[k] = h_adaptive
        T_adaptive[k] = threshold_adaptive
        M_adaptive[k] = mmd_value_adaptive
    print("Reject rate_LJ: ", H_u_l.sum() / N_f, "Reject rate_GJ: ", H_u.sum() / N_f, "Reject rate_GC:",
          H_adaptive.sum() / N_f,
          "Reject rate_LC: ", H_C2ST.sum() / N_f)
    Results[0, kk] = H_u_l.sum() / N_f
    Results[1, kk] = H_u.sum() / N_f
    Results[2, kk] = H_adaptive.sum() / N_f
    Results[3, kk] = H_C2ST.sum() / N_f
    print(Results, Results.mean(1))
f = open('./Results_MNIST_n' + str(N1) + '_H1_abl.pckl', 'wb')
pickle.dump([Results,ep_OPT,s_OPT,s0_OPT], f)
f.close()