import torch.nn.init as init
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import torch.nn.functional as F
import torch.nn as nn


def permute_dims(hs,hn):
    assert hs.dim() == 2
    assert hn.dim() == 2

    B, _ = hs.size()

    perm = torch.randperm(B).to(hs.device)
    perm_hs= hs[perm]
    perm = torch.randperm(B).to(hs.device)
    perm_hn= hn[perm]

    return perm_hs, perm_hn


def multinomial_loss_function(x_logit, x, z_mu, z_var, z, beta=1.):

    """
    Computes the cross entropy loss function while summing over batch dimension, not averaged!
    :param x_logit: shape: (batch_size, num_classes * num_channels, pixel_width, pixel_height), real valued logits
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param args: global parameter settings
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    # num_classes = 256
    batch_size = x.size(0)

    target = x
    ce = nn.MSELoss(size_average=False)(x_logit, target)

    kl = - (0.5 * torch.sum(1 + z_var.log() - z_mu.pow(2) - z_var.log().exp()))

    loss = ce + beta * kl
    loss = loss / float(batch_size)
    ce = ce / float(batch_size)
    kl = kl / float(batch_size)

    return loss, ce, kl


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

    def update(self, it, acc):
        self.acc_list += [acc]
        self.iter_list += [it]
        self.save_model = False
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            self.save_model = True

    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.iter_list += [it]
        self.save_model = False
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U_T = acc_u
            self.best_acc_S_T = acc_s
            self.save_model = True


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.normal_(m.weight.data,  mean=0, std=0.02)
        init.constant_(m.bias, 0.0)


def log_print(s, log):
    print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')

def synthesize_feature_test_ori(netG, dataset, opt):
    gen_feat = torch.FloatTensor(dataset.ntest_class * opt.nSample, opt.X_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for i in range(dataset.ntest_class):
            text_feat = np.tile(dataset.test_att[i].astype('float32'), (opt.nSample, 1))
            text_feat = torch.from_numpy(text_feat).to(opt.gpu)
            z = torch.randn(opt.nSample, opt.Z_dim).to(opt.gpu)
            G_sample = netG.decode(z, text_feat)
            gen_feat[i*opt.nSample:(i+1)*opt.nSample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([opt.nSample])*i))
    return gen_feat, torch.from_numpy(gen_label.astype(int))

def synthesize_feature_test(netG, ae, dataset, opt):
    gen_feat = torch.FloatTensor(dataset.ntest_class * opt.nSample, opt.S_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for i in range(dataset.ntest_class):
            text_feat = np.tile(dataset.test_att[i].astype('float32'), (opt.nSample, 1))
            text_feat = torch.from_numpy(text_feat).to(opt.gpu)
            z = torch.randn(opt.nSample, opt.Z_dim).to(opt.gpu)
            G_sample = ae.encoder(netG.decode(z, text_feat))[:,:opt.S_dim]
            gen_feat[i*opt.nSample:(i+1)*opt.nSample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([opt.nSample])*i))
    return gen_feat, torch.from_numpy(gen_label.astype(int))


def save_model(it, netG, netD, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'state_dict_D': netD.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)


def getloss(pred, x, z, opt):
    loss = 1/(2*opt.sigma**2) * torch.pow(x - pred, 2).sum() + 1/2 * torch.pow(z, 2).sum()
    loss /= x.size(0)
    return loss


def save_model(it, model, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict': model.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

    def update(self, it, acc):
        self.acc_list += [acc]
        self.iter_list += [it]
        self.save_model = False
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            self.save_model = True
    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.iter_list += [it]
        self.save_model = False
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U_T = acc_u
            self.best_acc_S_T = acc_s
            self.save_model = True


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.normal_(m.weight.data,  mean=0, std=0.02)
        init.constant_(m.bias, 0.0)


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in test_label.unique():
        idx = (test_label == i)
        acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class

def eval_zsl_knn(gen_feat, gen_label, dataset):
    # cosince predict K-nearest Neighbor
    n_test_sample = dataset.test_unseen_feature.shape[0]
    sim = cosine_similarity(dataset.test_unseen_feature, gen_feat)
    # only count first K nearest neighbor
    idx_mat = np.argsort(-1 * sim, axis=1)[:, 0:opt.Knn]
    label_mat = gen_label[idx_mat.flatten()].reshape((n_test_sample,-1))
    preds = np.zeros(n_test_sample)
    for i in range(n_test_sample):
        label_count = Counter(label_mat[i]).most_common(1)
        preds[i] = label_count[0][0]
    acc = eval_MCA(preds, dataset.test_unseen_label.numpy()) * 100
    return acc

def eval_MCA(preds, y):
    cls_label = np.unique(y)
    acc = list()
    for i in cls_label:
        acc.append((preds[y == i] == i).mean())
    return np.asarray(acc).mean()

def eval_zsl_knn(gen_feat, gen_label, dataset):
    # cosince predict K-nearest Neighbor
    n_test_sample = dataset.test_unseen_feature.shape[0]
    sim = cosine_similarity(dataset.test_unseen_feature, gen_feat)
    # only count first K nearest neighbor
    idx_mat = np.argsort(-1 * sim, axis=1)[:, 0:5]
    label_mat = gen_label[idx_mat.flatten()].reshape((n_test_sample,-1))
    preds = np.zeros(n_test_sample)
    for i in range(n_test_sample):
        label_count = Counter(label_mat[i]).most_common(1)
        preds[i] = label_count[0][0]
    acc = eval_MCA(preds, dataset.test_unseen_label.numpy()) * 100
    return acc


def eval_MCA(preds, y):
    cls_label = np.unique(y)
    acc = list()
    for i in cls_label:
        acc.append((preds[y == i] == i).mean())
    return np.asarray(acc).mean()
