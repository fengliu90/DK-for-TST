import numpy as np
import torch
import torchvision # use it for torch.utils.data
import freqopttest.data as data
import freqopttest.tst as tst
import scipy.stats as stats
import pdb

is_cuda = True

class ModelLatentF(torch.nn.Module):
    """define deep networks."""
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

def get_item(x, is_cuda):
    """get the numpy value from a torch tensor."""
    if is_cuda:
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    return x

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None, Kxyxy
    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        print('error_var!!'+str(V1))
    return mmd2, varEst, Kxyxy

def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)

    nx = X.shape[0]
    ny = Y.shape[0]
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    K_Ix = torch.eye(nx).cuda()
    K_Iy = torch.eye(ny).cuda()
    if is_smooth:
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def MMDu_linear_kernel(Fea, len_s, is_var_computed=True, use_1sample_U=True):
    """compute value of (deep) lineaer-kernel MMD and std of (deep) lineaer-kernel MMD using merged data."""
    try:
        X = Fea[0:len_s, :]
        Y = Fea[len_s:, :]
    except:
        X = Fea[0:len_s].unsqueeze(1)
        Y = Fea[len_s:].unsqueeze(1)

    Kx = X.mm(X.transpose(0,1))
    Ky = Y.mm(Y.transpose(0,1))
    Kxy = X.mm(Y.transpose(0,1))

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def C2ST_NN_fit(S,y,N1,x_in,H,x_out,learning_rate_C2ST,N_epoch,batch_size,device,dtype):
    """Train a deep network for C2STs."""
    N = S.shape[0]
    if is_cuda:
        model_C2ST = ModelLatentF(x_in, H, x_out).cuda()
    else:
        model_C2ST = ModelLatentF(x_in, H, x_out)
    w_C2ST = torch.randn([x_out, 2]).to(device, dtype)
    b_C2ST = torch.randn([1, 2]).to(device, dtype)
    w_C2ST.requires_grad = True
    b_C2ST.requires_grad = True
    optimizer_C2ST = torch.optim.Adam(list(model_C2ST.parameters()) + [w_C2ST] + [b_C2ST], lr=learning_rate_C2ST)
    criterion = torch.nn.CrossEntropyLoss()
    f = torch.nn.Softmax()
    ind = np.random.choice(N, N, replace=False)
    tr_ind = ind[:np.int(np.ceil(N * 1))]
    te_ind = tr_ind
    dataset = torch.utils.data.TensorDataset(S[tr_ind, :], y[tr_ind])
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
            tt = tt + 1
        if epoch % 100 == 0:
            print(criterion(model_C2ST(S).mm(w_C2ST) + b_C2ST, y).item())

    output = f(model_C2ST(S[te_ind, :]).mm(w_C2ST) + b_C2ST)
    pred = output.max(1, keepdim=True)[1]
    STAT_C2ST = abs(pred[:N1].type(torch.FloatTensor).mean() - pred[N1:].type(torch.FloatTensor).mean())
    return pred, STAT_C2ST, model_C2ST, w_C2ST, b_C2ST

def gauss_kernel(X, test_locs, X_org, test_locs_org, sigma, sigma0, epsilon):
    """compute a deep kernel matrix between a set of samples between test locations."""
    DXT = Pdist2(X, test_locs)
    DXT_org = Pdist2(X_org, test_locs_org)
    # Kx = torch.exp(-(DXT / sigma0))
    Kx = (1 - epsilon) * torch.exp(-(DXT / sigma0) - DXT_org / sigma) + epsilon * torch.exp(-DXT_org / sigma)
    return Kx

def compute_ME_stat(X, Y, T, X_org, Y_org, T_org, sigma, sigma0, epsilon):
    """compute a deep kernel based ME statistic."""
    # if gwidth is None or gwidth <= 0:
    #     raise ValueError('require gaussian_width > 0. Was %s.' % (str(gwidth)))
    reg = 0#10**(-8)
    n = X.shape[0]
    J = T.shape[0]
    g = gauss_kernel(X, T, X_org, T_org, sigma, sigma0, epsilon)
    h = gauss_kernel(Y, T, Y_org, T_org, sigma, sigma0, epsilon)
    Z = g - h
    W = Z.mean(0)
    Sig = ((Z - W).transpose(1, 0)).mm((Z - W))
    if is_cuda:
        IJ = torch.eye(J).cuda()
    else:
        IJ = torch.eye(J)
    s = n*W.unsqueeze(0).mm(torch.solve(W.unsqueeze(1),Sig + reg*IJ)[0])
    return s

def mmd2_permutations(K, n_X, permutations=200):
    """
        Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    return est.item(), p_val.item(), rest

def TST_MMD_adaptive_bandwidth(Fea, N_per, N1, Fea_org, sigma, sigma0, alpha, device, dtype):
    """run two-sample test (TST) using ordinary Gaussian kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, is_smooth=False)
    mmd_value = get_item(TEMP[0],is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]
        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
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
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

def TST_MMD_u(Fea, N_per, N1, Fea_org, sigma, sigma0, ep, alpha, device, dtype, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, ep, is_smooth)
    mmd_value = get_item(TEMP[0], is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
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
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

def TST_MMD_u_linear_kernel(Fea, N_per, N1, alpha,  device, dtype):
    """run two-sample test (TST) using (deep) lineaer kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu_linear_kernel(Fea, N1)
    mmd_value = get_item(TEMP[0], is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1

    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
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
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

def TST_C2ST(S,N1,N_per,alpha,model_C2ST, w_C2ST, b_C2ST,device,dtype):
    """run C2ST-S on non-image datasets."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax()
    output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
    pred_C2ST = output.max(1, keepdim=True)[1]
    STAT = abs(pred_C2ST[:N1].type(torch.FloatTensor).mean() - pred_C2ST[N1:].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(pred_C2ST[ind_X].type(torch.FloatTensor).mean() - pred_C2ST[ind_Y].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    threshold_lower = S_vector[np.int(np.ceil(N_per *  alpha))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    # if STAT.item() < threshold_lower:
    #     h = 1
    return h, threshold, STAT

def TST_LCE(S,N1,N_per,alpha,model_C2ST, w_C2ST, b_C2ST, device,dtype):
    """run C2ST-L on non-image datasets."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax()
    output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
    # pred_C2ST = output.max(1, keepdim=True)[1]
    STAT = abs(output[:N1,0].type(torch.FloatTensor).mean() - output[N1:,0].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(output[ind_X,0].type(torch.FloatTensor).mean() - output[ind_Y,0].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    threshold_lower = S_vector[np.int(np.ceil(N_per *  alpha))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT

def TST_ME(Fea, N1, alpha, is_train, test_locs, gwidth, J = 1, seed = 15):
    """run ME test."""
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
    """run SCF test."""
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

def TST_C2ST_D(S,N1,N_per,alpha,discriminator,device,dtype):
    """run C2ST-S on MNIST and CIFAR datasets."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax()
    output = discriminator(S)
    pred_C2ST = output.max(1, keepdim=True)[1]
    STAT = abs(pred_C2ST[:N1].type(torch.FloatTensor).mean() - pred_C2ST[N1:].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        STAT_vector[r] = abs(pred_C2ST[ind_X].type(torch.FloatTensor).mean() - pred_C2ST[ind_Y].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    threshold_lower = S_vector[np.int(np.ceil(N_per *  alpha))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT

def TST_LCE_D(S,N1,N_per,alpha,discriminator,device,dtype):
    """run C2ST-L on MNIST and CIFAR datasets."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax()
    output = discriminator(S)
    STAT = abs(output[:N1,0].type(torch.FloatTensor).mean() - output[N1:,0].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(output[ind_X,0].type(torch.FloatTensor).mean() - output[ind_Y,0].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT

def TST_ME_DK(X, Y, T, X_org, Y_org, T_org, alpha, sigma, sigma0, epsilon, flag_debug = False):
    """run deep-kernel ME test (using chi^2 to confirm the threshold) on CIFAR datasets (this code does not work)."""
    J = T.shape[0]
    s = compute_ME_stat(X, Y, T, X_org, Y_org, T_org, sigma, sigma0, epsilon)
    pvalue = stats.chi2.sf(s.item(), J)
    if pvalue<alpha:
        h = 1
    else:
        h = 0
    if flag_debug:
        pdb.set_trace()
    return h, pvalue, s

def TST_ME_DK_per(X, Y, T, X_org, Y_org, T_org, alpha, sigma, sigma0, epsilon):
    """run deep-kernel ME test (using permutations to confirm the threshold) on CIFAR datasets."""
    N_per = 100
    J = T.shape[0]
    s = compute_ME_stat(X, Y, T, X_org, Y_org, T_org, sigma, sigma0, epsilon)
    Fea = torch.cat([X.cpu(), Y.cpu()], 0).cuda()
    Fea_org = torch.cat([X_org.cpu(), Y_org.cpu()], 0).cuda()
    N1 = X.shape[0]
    N = Fea.shape[0]
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = compute_ME_stat(Fea[ind_X,:], Fea[ind_Y,:], T, Fea_org[ind_X,:], Fea_org[ind_Y,:], T_org, sigma, sigma0, epsilon)
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if s.item() > threshold:
        h = 1
    return h, threshold, s