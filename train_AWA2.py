import torch.optim as optim
import glob
import json
import argparse
import os
import random
from time import gmtime, strftime
from models import *
from dataset_GBU import FeatDataLayer, DATA_LOADER
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch.backends.cudnn as cudnn
import classifier

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='dataset: CUB, AWA1, APY, FLO')
parser.add_argument('--dataroot', default='./SDGZSL_data', help='path to dataset')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)
parser.add_argument('--finetune', type=bool, default=False, help='Use fine-tuned feature')
parser.add_argument('--gen_nepoch', type=int, default=220, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00003, help='learning rate to train generater')

parser.add_argument('--ga', type=float, default=0.5, help='relationNet weight')
parser.add_argument('--beta', type=float, default=1, help='tc weight')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--dis', type=float, default=0.3, help='discriminator loss weight')

parser.add_argument('--classifier_lr', type=float, default=0.003, help='learning rate to train softmax classifier')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=5000, help='number features to generate per class')

parser.add_argument('--disp_interval', type=int, default=200)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=300)
parser.add_argument('--manualSeed', type=int, default=6152, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--gh_dim',     type=int, default=4096, help='dimention of hidden layer in decoder')
parser.add_argument('--eh_dim',     type=int, default=4096, help='dimention of hidden layer in encoder')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
parser.add_argument('--S_dim', type=int, default=312)
parser.add_argument('--NS_dim', type=int, default=312)
parser.add_argument('--zsl', default=False)

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")

def train():
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class
    out_dir = 'out/{}/b-{}_g-{}_lr-{}_ds-{}__nS-{}_nZ-{}_bs-{}_gh-{}_eh-{}'.format(opt.dataset, opt.beta, opt.ga, opt.lr,
                            opt.S_dim, opt.nSample, opt.Z_dim, opt.batchsize, opt.gh_dim, opt.eh_dim)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)

    opt.niter = int(dataset.ntrain/opt.batchsize) * opt.gen_nepoch

    result_gzsl_soft = Result()
    result_zsl_soft = Result()
    model = VAE(opt).to(opt.gpu)
    relationNet = RelationNet(opt).to(opt.gpu)
    discriminator = Discriminator(opt).to(opt.gpu)
    ae = AE(opt).to(opt.gpu)
    print(model)

    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('Generative Model Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    relation_optimizer = optim.Adam(relationNet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ae_optimizer =  optim.Adam(ae.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ones = torch.ones(opt.batchsize, dtype=torch.long, device=opt.gpu)
    zeros = torch.zeros(opt.batchsize, dtype=torch.long, device=opt.gpu)
    mse = nn.MSELoss().to(opt.gpu)

    import math
    iters = math.ceil(dataset.ntrain/opt.batchsize)
    beta = 0.01
    coin = 0
    gamma = 0
    for it in range(start_step, opt.niter+1):

        if it % iters == 0:
            beta = min(0.01*(it/iters), 1)
            gamma = min(0.001 * (it / iters), 1)
        blobs = data_layer.forward()
        feat_data = blobs['data']  # image data
        labels_numpy = blobs['labels'].astype(int)  # class labels
        labels = torch.from_numpy(labels_numpy.astype('int')).to(opt.gpu)

        C = np.array([dataset.train_att[i,:] for i in labels])
        C = torch.from_numpy(C.astype('float32')).to(opt.gpu)
        X = torch.from_numpy(feat_data).to(opt.gpu)
        sample_C = torch.from_numpy(np.array([dataset.train_att[i, :] for i in labels.unique()])).to(opt.gpu)
        sample_C_n = labels.unique().shape[0]
        sample_label = labels.unique().cpu()

        x_mean, z_mu, z_var, z = model(X, C)

        loss, ce, kl = multinomial_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)

        sample_labels = np.array(sample_label)
        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        one_hot_labels = torch.tensor(
            torch.zeros(opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1)).to(opt.gpu)

        x1, h1, hs1, hn1 = ae(x_mean)
        relations = relationNet(hs1, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])
        p_loss = opt.ga * mse(relations, one_hot_labels)

        x2, h2, hs2, hn2 = ae(X)
        relations = relationNet(hs2, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])
        p_loss = p_loss + opt.ga * mse(relations, one_hot_labels)

        rec = (mse(x1, X) + mse(x2, X))

        if coin > 0:
            s_score = discriminator(h1)
            tc_loss = opt.beta * gamma * ((s_score[:, :1] - s_score[:, 1:]).mean())
            s_score = discriminator(h2)
            tc_loss = tc_loss + opt.beta* gamma * ((s_score[:, :1] - s_score[:, 1:]).mean())
            loss = loss + p_loss + rec + tc_loss
            coin -= 1
        else:
            s, n = permute_dims(hs1, hn1)
            b = torch.cat((s, n), 1).detach()
            s_score = discriminator(h1)
            n_score = discriminator(b)
            tc_loss = opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

            s, n = permute_dims(hs2, hn2)
            b = torch.cat((s, n), 1).detach()
            s_score = discriminator(h2)
            n_score = discriminator(b)
            tc_loss = tc_loss + opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

            dis_optimizer.zero_grad()
            tc_loss.backward(retain_graph=True)
            dis_optimizer.step()

            loss = loss + p_loss + rec
            coin += 2

        optimizer.zero_grad()
        relation_optimizer.zero_grad()
        ae_optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        relation_optimizer.step()
        ae_optimizer.step()

        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; loss: {:.3f}; kl:{:.3f}; p_loss:{:.3f}; rec:{:.3f}; tc:{:.3f}; gamma:{:.3f}'.format(it,
                                             opt.niter, loss.item(),kl.item(),p_loss.item(),rec.item(), tc_loss.item(), gamma)
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it > 40000:
            model.eval()
            ae.eval()
            gen_feat, gen_label = synthesize_feature_test(model, ae, dataset, opt)
            with torch.no_grad():
                train_feature = ae.encoder(dataset.train_feature.to(opt.gpu))[:,:opt.S_dim].cpu()
                test_unseen_feature = ae.encoder(dataset.test_unseen_feature.to(opt.gpu))[:,:opt.S_dim].cpu()
                test_seen_feature = ae.encoder(dataset.test_seen_feature.to(opt.gpu))[:,:opt.S_dim].cpu()

            train_X = torch.cat((train_feature, gen_feat), 0)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)
            if opt.zsl:

                """ZSL"""
                cls = classifier.CLASSIFIER(opt, gen_feat, gen_label, dataset, test_seen_feature, test_unseen_feature,
                                            dataset.ntrain_class + dataset.ntest_class, True, 0.004, 0.5, 20,
                                            opt.nSample, False)
                result_zsl_soft.update(it, cls.acc)
                log_print("ZSL Softmax:", log_dir)
                log_print("Acc {:.2f}%  Best_acc [{:.2f}% | Iter-{}]".format(
                    cls.acc, result_zsl_soft.best_acc, result_zsl_soft.best_iter), log_dir)

            else:
                """ GZSL"""
                cls = classifier.CLASSIFIER(opt, train_X, train_Y, dataset, test_seen_feature, test_unseen_feature,
                                    dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, 40, opt.nSample, True)

                result_gzsl_soft.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

                log_print("GZSL Softmax:", log_dir)
                log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                    cls.acc_unseen, cls.acc_seen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                    result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

                if result_gzsl_soft.save_model:
                    files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    save_model(it, model, opt.manualSeed, log_text,
                               out_dir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl_soft.best_acc,
                                                                                                 result_gzsl_soft.best_acc_S_T,
                                                                                                 result_gzsl_soft.best_acc_U_T))
                ###############################################################################################################

                # retrieval code
                cls_centrild = np.zeros((dataset.ntest_class, opt.S_dim))
                for i in range(dataset.ntest_class):
                    cls_centrild[i] = torch.mean(gen_feat[gen_label == i, ], dim=0)

                dist = cosine_similarity(cls_centrild, test_unseen_feature)

                precision_100 = torch.zeros(dataset.ntest_class)
                precision_50 = torch.zeros(dataset.ntest_class)
                precision_25 = torch.zeros(dataset.ntest_class)

                dist = torch.from_numpy(-dist)
                for i in range(dataset.ntest_class):
                    is_class = dataset.test_unseen_label == i
                    # print(is_class.sum())
                    cls_num = int(is_class.sum())

                    # 100%
                    _, idx = torch.topk(dist[i, :], cls_num, largest=False)
                    precision_100[i] = (is_class[idx]).sum().float() / cls_num

                    # 50%
                    cls_num_50 = int(cls_num / 2)
                    _, idx = torch.topk(dist[i, :], cls_num_50, largest=False)
                    precision_50[i] = (is_class[idx]).sum().float() / cls_num_50

                    # 25%
                    cls_num_25 = int(cls_num / 4)
                    _, idx = torch.topk(dist[i, :], cls_num_25, largest=False)
                    precision_25[i] = (is_class[idx]).sum().float() / cls_num_25
                print("retrieval results 100%%: %.3f 50%%: %.3f 25%%: %.3f" % (precision_100.mean().item(),
                                                                               precision_50.mean().item(),
                                                                               precision_25.mean().item()))
                ###############################################################################################################
                model.train()
                ae.train()
            if it % opt.save_interval == 0 and it:
                save_model(it, model, opt.manualSeed, log_text,
                           out_dir + '/Iter_{:d}.tar'.format(it))
                print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))


if __name__ == "__main__":
    train()