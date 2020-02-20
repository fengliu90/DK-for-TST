# -*- coding: utf-8 -*-
"""
Created on Dec 21 14:57:02 2019
@author: Learning Deep Kernels for Two-sample Test
@Implementation of Deep-kernel ME (training test locations) and
 ME in our paper on CIFAR dataset (Interpretability experiments).

BEFORE USING THIS CODE:
1. This code requires PyTorch 1.1.0, which can be found in
https://pytorch.org/get-started/previous-versions/ (CUDA version is 10.1).
2. This code also requires freqopttest repo (interpretable nonparametric two-sample test)
to implement ME and SCF tests, which can be installed by
   pip install git+https://github.com/wittawatj/interpretable-test
3. Numpy, Sklearn, PIL are also required. Users can install
Python via Anaconda (Python 3.7.3) to obtain both packages. Anaconda
can be found in https://www.anaconda.com/distribution/#download-section .

Note that runing time of ME test on CIFAR dataset is very long due to the huge dimension of data
"""
import argparse
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from PIL import Image
import numpy as np
from utils_HD import compute_ME_stat, MatConvert, MMDu, TST_ME, TST_ME_DK_per

# Setup seeds
os.makedirs("images", exist_ok=True)
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n", type=int, default=1000, help="number of samples")
opt = parser.parse_args()
print(opt)
dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False
N_per = 100 # permutation times
alpha = 0.05 # test threshold
N1 = opt.n # number of samples in one set
K = 10 # number of trails
J = 1 # number of test locations
N = 100 # number of test sets
N_f = 100.0 # number of test sets (float)

# Loss function
adversarial_loss = torch.nn.CrossEntropyLoss()

# Naming variables
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
T_org_OPT = torch.zeros([K,J,3,64,64]) # Record test locations obtained by MMD-D
ME_test_locs = np.zeros([K,J,3*64*64]) # Record test locations obtained by ME
Results = np.zeros([2,K])

# Define the deep network for distinguishing two sets of samples
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

# Define the deep network for MMD-D
class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()

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
            nn.Linear(128 * ds_size ** 2, 300))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature

# Configure data loader
dataset_test = datasets.CIFAR10(root='./data/cifar10', download=True,train=False,
                           transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000,
                                             shuffle=True, num_workers=1)

# Obtain CIFAR10 images
for i, (imgs, Labels) in enumerate(dataloader_test):
    data_all = imgs
    label_all = Labels
Ind_all = np.arange(len(data_all))

# Obtain CIFAR10.1 images
data_new = np.load('./cifar10.1_v4_data.npy')
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

# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    print(kk)
    torch.manual_seed(kk * 19 + N1)
    torch.cuda.manual_seed(kk * 19 + N1)
    np.random.seed(seed=1102 * (kk + 10) + N1)
    # Initialize deep networks for MMD-D
    featurizer = Featurizer()
    discriminator = Discriminator()
    # Initialize parameters
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = True
    TT_org = MatConvert(np.random.randn(J,3,64,64), device, dtype)
    TT_org.requires_grad = True
    print(epsilonOPT.item())
    if cuda:
        featurizer.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Collect CIFAR10 images
    Ind_tr = np.random.choice(len(data_all), N1, replace=False)
    Ind_te = np.delete(Ind_all, Ind_tr)
    train_data = []
    for i in Ind_tr:
       train_data.append([data_all[i], label_all[i]])
    print(len(train_data))
    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Collect CIFAR10.1 images
    np.random.seed(seed=819 * (kk + 9) + N1)
    Ind_tr_v4 = np.random.choice(len(data_T), N1, replace=False)
    Ind_te_v4 = np.delete(Ind_v4_all, Ind_tr_v4)
    New_CIFAR_tr = data_trans[Ind_tr_v4]
    New_CIFAR_te = data_trans[Ind_te_v4]

    # Initialize optimizers
    optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=opt.lr) # optimizer for training deep kernel
    optimizer_T = torch.optim.Adam([sigmaOPT] + [sigma0OPT] + [TT_org], lr=opt.lr) # optimizer for training test location
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr) # optimizer for training deep networks to distinguish two sets of samples
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ---------------------
    #  Training deep kernel
    # ---------------------
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            if True:
                ind = np.random.choice(N1, imgs.shape[0], replace=False)
                Fake_imgs = New_CIFAR_tr[ind]
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
                # Initialize optimizer
                optimizer_F.zero_grad()
                # Compute output of deep network
                modelu_output = featurizer(X)
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                # Compute Compute J (STAT_u)
                TEMP = MMDu(modelu_output, imgs.shape[0], X.view(X.shape[0], -1), sigma, sigma0_u, ep)
                mmd_value_temp = -1 * (TEMP[0])
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                STAT_u_F = torch.div(mmd_value_temp, mmd_std_temp)
                # Compute gradient
                STAT_u_F.backward()
                # Update weights using gradient descent
                optimizer_F.step()

                # ------------------------------------------------------
                #  Train deep network to distinguish two sets of samples
                # ------------------------------------------------------
                # Initialize optimizer
                optimizer_D.zero_grad()
                # Compute Cross-Entropy (loss_C) loss between two samples
                loss_C = adversarial_loss(discriminator(X), Y)
                # Compute gradient
                loss_C.backward()
                # Update weights using gradient descent
                optimizer_D.step()
                if (epoch + 1) % 100 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Stat: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), loss_C.item(), -STAT_u_F.item())
                    )
                batches_done = epoch * len(dataloader) + i
            else:
                break

    # ---------------------------
    #  Training for test location
    # ---------------------------

    # Fetch training data
    s1 = data_all[Ind_tr]
    s2 = data_trans[Ind_tr_v4]
    S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
    print(S.shape)
    Sv = S.view(2 * N1, -1)
    # Fetch test data
    data_all_te = data_all[Ind_te]
    N_te = 1000
    Ind_N_te = np.random.choice(len(Ind_te), N_te, replace=False)
    s1_te = data_all_te[Ind_N_te]
    s2_te = data_trans[Ind_te_v4[:N_te]]
    S_te = torch.cat([s1_te.cpu(), s2_te.cpu()], 0).cuda()
    print(S_te.shape)

    # Initialize test location
    T_org = TT_org

    # Train test location
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            if True:
                ind = np.random.choice(N1, imgs.shape[0], replace=False)
                Fake_imgs = New_CIFAR_tr[ind]
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
                # Configure input
                real_imgs = imgs.cuda()
                Fake_imgs = Fake_imgs.cuda()
                X = torch.cat([real_imgs, Fake_imgs], 0)
                Y = torch.cat([valid, fake], 0).squeeze().long()

                # ---------------------
                #  Train test location
                # ---------------------
                # Initialize optimizer
                optimizer_T.zero_grad()
                # Compute output of deep network
                modelu_output = featurizer(X)
                modelu_output_real = featurizer(real_imgs)
                modelu_output_fake = featurizer(Fake_imgs)
                # Compute epsilon, sigma, sigma_0 and test location
                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                T_org = torch.div(TT_org, 1 + TT_org.abs())
                T_org = T_org / T_org.max()
                # Compute output of deep network(T_org)
                modelu_output_T = featurizer(T_org)
                # Compute ME statistic based on the trained deep kernel and test location
                TEMP = compute_ME_stat(featurizer(real_imgs), featurizer(Fake_imgs), featurizer(T_org),
                                       real_imgs.view(imgs.shape[0], -1), Fake_imgs.view(imgs.shape[0], -1),
                                       T_org.view(J, -1), sigma, sigma0_u, ep)
                # Compute ME statistic based on the trained deep kernel and test location on training data
                TEMP_tr = compute_ME_stat(featurizer(S[:N1,:]), featurizer(S[N1:,:]), featurizer(T_org),
                                       S[:N1,:].view(N1, -1), S[N1:,:].view(N1, -1),
                                       T_org.view(J, -1), sigma, sigma0_u, ep)
                # Compute ME statistic based on the trained deep kernel and test location on test data
                TEMP_te = compute_ME_stat(featurizer(S_te[:N_te, :]), featurizer(S_te[N_te:, :]), featurizer(T_org),
                                            S_te[:N_te, :].view(N_te, -1), S_te[N_te:, :].view(N_te, -1),
                                            T_org.view(J, -1), sigma, sigma0_u, ep)
                # Fetch ME statistic
                ME_value_temp = -1 * (TEMP[0])
                # Compute gradient
                STAT_u_T = ME_value_temp
                STAT_u_T.backward(retain_graph=True)
                # Update test location using gradient descent
                optimizer_T.step()

                if (epoch + 1) % 50 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [Stat_test: %f] [Stat: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), TEMP_te[0], TEMP[0])
                    )
    # Compute ME statistic based on the trained deep kernel and test location on training data
    stat__me = compute_ME_stat(featurizer(S[:N1, :]), featurizer(S[N1:, :]), featurizer(T_org),
                    S[:N1, :].view(N1, -1), S[N1:, :].view(N1, -1),
                    T_org.view(J, -1), sigma, sigma0_u, ep)
    # Run two-sample test based on deep-kernel ME
    h_u, threshold_u, mmd_value_u = TST_ME_DK_per(featurizer(S[:N1, :]), featurizer(S[N1:, :]), featurizer(T_org),
                    S[:N1, :].view(N1, -1), S[N1:, :].view(N1, -1),
                    T_org.view(J, -1), alpha, sigma, sigma0_u, ep)

    print("h:", h_u, "Threshold:", threshold_u, "MMD_value:", mmd_value_u.item(), "ME_stats:", stat__me.item())

    # Train ME
    np.random.seed(seed=1102)
    test_locs_ME, gwidth_ME = TST_ME(Sv, N1, alpha, is_train=True, test_locs=1, gwidth=1, J=1, seed=15)
    h_ME = TST_ME(Sv, N1, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=1, seed=15)

    # Record the best epsilon, sigma, sigma_0 and test location (deep-kernel ME and ordinary ME)
    ep_OPT[kk] = ep.item()
    s_OPT[kk] = sigma.item()
    s0_OPT[kk] = sigma0_u.item()
    T_org_OPT[kk] = T_org
    ME_test_locs[kk] = test_locs_ME

    # Compute test power of MMD-D and baselines
    H_u = np.zeros(N)
    T_u = np.zeros(N)
    M_u = np.zeros(N)
    H_ME = np.zeros(N)
    np.random.seed(1102)
    count_u = 0
    count_ME = 0
    for k in range(N):
        # Fetch test data
        np.random.seed(seed=1102 * (k + 1) + N1)
        data_all_te = data_all[Ind_te]
        N_te = 1000
        Ind_N_te = np.random.choice(len(Ind_te), N_te, replace=False)
        s1 = data_all_te[Ind_N_te]
        s2 = data_trans[Ind_te_v4[:N_te]]
        S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
        Sv = S.view(2 * N_te, -1)
        # Deep-kernel ME
        h_u, threshold_u, mmd_value_u = TST_ME_DK_per(featurizer(S[:N_te, :]), featurizer(S[N_te:, :]), featurizer(T_org),
                    S[:N_te, :].view(N_te, -1), S[N_te:, :].view(N_te, -1),
                    T_org.view(J, -1), alpha, sigma, sigma0_u, ep)
        # ME
        h_ME = TST_ME(Sv, N_te, alpha, is_train=False, test_locs=test_locs_ME, gwidth=gwidth_ME, J=1, seed=15)

        # Gather results
        count_u = count_u + h_u
        count_ME = count_ME + h_ME
        print("DKME:", count_u, "ME:", count_ME)
        H_u[k] = h_u
        T_u[k] = threshold_u
        M_u[k] = mmd_value_u
        H_ME[k] = h_ME
    print("Reject rates DKME: ", H_u.sum() / N_f, "Reject rates ME: ",H_ME.sum() / N_f)
    Results[0, kk] = H_u.sum() / N_f
    Results[1, kk] = H_ME.sum() / N_f
    print("Test power: ",Results)
    print("Average Test power: ", Results.sum(1) / (kk+1))

# Print test locations obtain by ME
for ii in range(10):
    T0 = np.transpose(ME_test_locs[ii, 0].reshape([3, 64, 64]), (1, 2, 0))
    img = Image.fromarray(T0, 'RGB')
    img.save('T_locs_CIFAR10_ME_' + str(ii) + '.png')

# Print test locations obtain by deep-kernel ME
for ii in range(10):
    T0 = np.transpose(T_org_OPT[ii,0].detach().numpy(),(1,2,0)) # convert 3*64*64 to 64*64*3
    img = Image.fromarray(T0, 'RGB')
    img.save('T_locs_CIFAR10_DKME_'+str(ii)+'.png')

