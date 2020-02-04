import argparse
import os
import numpy as np
import math
import pdb
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
from TST_utils_HD import compute_ME_stat, MatConvert, Pdist2, MMDu, get_item, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_ME_DK, TST_SCF, TST_C2ST_D


os.makedirs("images", exist_ok=True)
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=500, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate") #0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--N1", type=int, default=1000, help="number of samples")
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
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)]
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
            nn.Linear(128 * ds_size ** 2, 300),
            nn.ReLU(),
            nn.Linear(300, 2),
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
            nn.Linear(128 * ds_size ** 2, 300))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature

sigma = 2*32*32 # 2*64*64 for 64x64
sigma0_u = 20 #0.25 10 for 64x64
N_per = 100
alpha = 0.05
N1 = opt.N1
K = 10
Results = np.zeros([6,K])
# Configure data loader
dataset_test = datasets.CIFAR10(root='./data/cifar10', download=False,train=False,
                           transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000,
                                             shuffle=True, num_workers=1)

for i, (imgs, Labels) in enumerate(dataloader_test):
    data_all = imgs
    label_all = Labels

Ind_all = np.arange(len(data_all))


data_new = np.load('/home/fengliu/45_AISTATS/data/cifar10.1_v4_data.npy')
data_T = np.transpose(data_new, [0,3,1,2])
ind_M = np.random.choice(len(data_T), len(data_T), replace=False)
data_T = data_T[ind_M]
TT = transforms.Compose([transforms.Resize(opt.img_size),transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans = transforms.ToPILImage()
data_trans = torch.zeros([len(data_T),3,opt.img_size,opt.img_size])
data_T_tensor = torch.from_numpy(data_T)
for i in range(len(data_T)):
    d0 = trans(data_T_tensor[i])
    data_trans[i] = TT(d0)

Ind_v4_all = np.arange(len(data_T))


# Loss function
adversarial_loss = torch.nn.CrossEntropyLoss()
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
H_train = np.zeros([K])
J = 10
T_org_OPT = torch.zeros([K,J,3,64,64])

for kk in range(K):
    torch.manual_seed(kk * 19 + N1)
    torch.cuda.manual_seed(kk * 19 + N1)
    np.random.seed(seed=1102 * (kk + 10) + N1)
    # Initialize generator and discriminator
    featurizer = Featurizer()
    discriminator = Discriminator()

    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)  # d = 3,5 ??
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype) # 0.005
    sigma0OPT.requires_grad = True
    TT_org = MatConvert(np.random.randn(J,3,64,64), device, dtype)
    TT_org.requires_grad = True
    print(epsilonOPT.item())

    if cuda:
        featurizer.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    # featurizer.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)

    # generate data

    Ind_tr = np.random.choice(len(data_all), N1, replace=False)
    Ind_te = np.delete(Ind_all, Ind_tr)
    train_data = []
    for i in Ind_tr:
       train_data.append([data_all[i], label_all[i]])

    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    np.random.seed(seed=819 * (kk + 9) + N1)
    Ind_tr_v4 = np.random.choice(len(data_T), N1, replace=False)
    Ind_te_v4 = np.delete(Ind_v4_all, Ind_tr_v4)
    Fake_MNIST_tr = data_trans[Ind_tr_v4]
    Fake_MNIST_te = data_trans[Ind_te_v4]

    # Optimizers
    optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT] + [TT_org], lr=0.0002) # 0.001
    optimizer_T = torch.optim.Adam([sigmaOPT] + [TT_org], lr=0.0002) # 0.001
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ---------------------
    #  Training deep kernel
    # ---------------------
    # np.random.seed(seed=1102)
    # torch.manual_seed(1102)
    # torch.cuda.manual_seed(1102)
    # for epoch in range(opt.n_epochs):
    #     for i, (imgs, _) in enumerate(dataloader):
    #         if True:
    #             ind = np.random.choice(N1, imgs.shape[0], replace=False)
    #             Fake_imgs = Fake_MNIST_tr[ind]
    #             # Adversarial ground truths
    #             valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
    #             fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
    #
    #             # Configure input
    #             real_imgs = Variable(imgs.type(Tensor))
    #             Fake_imgs = Variable(Fake_imgs.type(Tensor))
    #             X = torch.cat([real_imgs, Fake_imgs], 0)
    #             Y = torch.cat([valid, fake], 0).squeeze().long()
    #             # -----------------
    #             #  Train Featurizer
    #             # -----------------
    #
    #             optimizer_F.zero_grad()
    #
    #             modelu_output = featurizer(X)
    #
    #             ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))  # 10 ** (-10)#
    #             sigma = sigmaOPT ** 2
    #             sigma0_u = sigma0OPT ** 2
    #
    #             TEMP = MMDu(modelu_output, imgs.shape[0], X.view(X.shape[0], -1), sigma, sigma0_u, ep)
    #             mmd_value_temp = -1 * (TEMP[0])  # 10**(-8)
    #             mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))  # 0.1
    #             if mmd_std_temp.item() == 0:
    #                 print('error std!!')
    #             if np.isnan(mmd_std_temp.item()):
    #                 print('error mmd!!')
    #             f_loss = torch.div(mmd_value_temp, mmd_std_temp)  # - r_full / (N1+N2)
    #             f_loss.backward()
    #             optimizer_F.step()
    #             # J_star_u[t] = f_loss.item()
    #
    #             # ---------------------
    #             #  Train Discriminator
    #             # ---------------------
    #
    #             optimizer_D.zero_grad()
    #
    #             # Measure discriminator's ability to classify real from generated samples
    #             d_loss = adversarial_loss(discriminator(X), Y)
    #             d_loss.backward()
    #             optimizer_D.step()
    #             if (epoch + 1) % 100 == 0:
    #                 print(
    #                     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Stat: %f]"
    #                     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), -f_loss.item())
    #                 )
    #
    #             batches_done = epoch * len(dataloader) + i
    #             # if batches_done % opt.sample_interval == 0:
    #             #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
    #         else:
    #             break

    # ---------------------------
    #  Training for test location
    # ---------------------------
    T_org = TT_org
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

                modelu_output_real = featurizer(real_imgs)
                modelu_output_fake = featurizer(Fake_imgs)


                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))  # 10 ** (-10)#
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                T_org = TT_org #/ TT_org.max()

                modelu_output_T = featurizer(T_org)

                TEMP = compute_ME_stat(modelu_output_real, modelu_output_fake, modelu_output_T, real_imgs.view(imgs.shape[0],-1), Fake_imgs.view(imgs.shape[0],-1), T_org.view(J,-1), sigma, sigma0_u, ep)

                mmd_value_temp = -1 * (TEMP[0])  # 10**(-8)

                f_loss = mmd_value_temp
                f_loss.backward()
                optimizer_F.step()
                # J_star_u[t] = f_loss.item()
    #
    #             if (epoch + 1) % 100 == 0:
    #                 print(
    #                     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Stat: %f]"
    #                     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), -f_loss.item())
    #                 )


    # ---------------------------
    #  SELECT test location
    # ---------------------------

    

    s1 = data_all[Ind_tr]
    s2 = data_trans[Ind_tr_v4]
    S = torch.cat([s1.cpu(),s2.cpu()],0).cuda()
    Sv = S.view(2*N1,-1)
    # pdb.set_trace()
    h_u, threshold_u, mmd_value_u = TST_ME_DK(featurizer(S), Sv, N1, featurizer(T_org), T_org.view(J,-1), alpha, sigma, sigma0_u, ep)
        # TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device, dtype)
    H_train[kk] = h_u
    print("h:", h_u, "Threshold:", threshold_u, "MMD_value:", mmd_value_u)

    ep_OPT[kk] = ep.item()
    s_OPT[kk] = sigma.item()
    s0_OPT[kk] = sigma0_u.item()
    T_org_OPT[kk] = T_org
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
    H_C2ST = np.zeros(N)
    Tu_C2ST = np.zeros(N)
    Tl_C2ST = np.zeros(N)
    S_C2ST = np.zeros(N)
    np.random.seed(1102)
    count_u = 0
    count_adp = 0
    count_ME = 0
    count_SCF = 0
    count_C2ST = 0
    for k in range(N):

        # generate data
        np.random.seed(seed=1102 * (k + 1) + N1)
        data_all_te = data_all[Ind_te]
        N_te = 1000#len(data_trans)-N1
        Ind_N_te = np.random.choice(len(Ind_te), N_te, replace=False)
        # s1 = data_all_te[:N_te]
        s1 = data_all_te[Ind_N_te]
        s2 = data_trans[Ind_te_v4[:1000]]
        S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
        Sv = S.view(2 * N_te, -1)
        # pdb.set_trace()
        h_u, threshold_u, mmd_value_u = TST_ME_DK(featurizer(S), Sv, N_te, featurizer(T_org), T_org.view(J,-1), alpha, sigma, sigma0_u, ep)
        # h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(Sv, N_per, N_te, Sv, sigma, sigma0, alpha, device, dtype)
        # h_ME = TST_ME(Sv, N_te, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=10, seed=15)
        # h_SCF = TST_SCF(Sv, N_te, alpha, is_train=False, test_freqs=test_freqs_SCF, gwidth=gwidth_SCF, J=10, seed=15)
        # H_C2ST[k], Tu_C2ST[k], S_C2ST[k] = TST_C2ST_D(S, N_te, N_per, alpha, discriminator, device, dtype)
        count_u = count_u + h_u
        # count_adp = count_adp + h_adaptive
        # count_ME = count_ME + h_ME
        # count_SCF = count_SCF + h_SCF
        # count_C2ST = count_C2ST + int(H_C2ST[k])
        print("MMD-DK:", count_u, "MMD-OPT:", count_adp, "MMD-ME:", count_ME, "SCF:", count_SCF, "C2ST: ", count_C2ST)
        H_u[k] = h_u
        T_u[k] = threshold_u
        M_u[k] = mmd_value_u
        # H_adaptive[k] = h_adaptive
        # T_adaptive[k] = threshold_adaptive
        # M_adaptive[k] = mmd_value_adaptive
        # H_ME[k] = h_ME
        # H_SCF[k] = h_SCF
    print("Reject rate_u: ", H_u.sum() / N_f, "Reject rate_C2ST: ", H_C2ST.sum() / N_f, "Reject rate_adaptive: ",
          H_adaptive.sum() / N_f, "Reject rate_ME: ", H_ME.sum() / N_f, "Reject rate_SCF: ", H_SCF.sum() / N_f,
          "Reject rate_m: ", H_m.sum() / N_f)
    Results[0, kk] = H_u.sum() / N_f
    Results[1, kk] = H_C2ST.sum() / N_f
    Results[2, kk] = H_adaptive.sum() / N_f
    Results[3, kk] = H_m.sum() / N_f
    Results[4, kk] = H_ME.sum() / N_f
    Results[5, kk] = H_SCF.sum() / N_f
    print(Results, Results.mean(1))
f = open('/home/fengliu/45_AISTATS/Results_Cifar10_n' + str(N1) + '_H1_Interp_ALLOPT.pckl', 'wb')
pickle.dump([Results,ep_OPT,s_OPT,s0_OPT,T_org_OPT,H_train], f)
f.close()