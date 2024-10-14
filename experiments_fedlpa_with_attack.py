import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
from collections import OrderedDict

import datetime
# from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from kfac import KFACOptimizer
# from torch.distributions import Laplace
# laplace_distribution = Laplace(0,0.333)


import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image
from model import *
from collections import OrderedDict

class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs  # img paths
        self.labs = labs  # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset used for training')
    parser.add_argument(
        '--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo',
                        help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,
                        help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50,
                        help='number of maximum communication roun')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False,
                        default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False,
                        default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5,
                        help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False,
                        default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False,
                        default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run the program')
    parser.add_argument('--log_file_name', type=str,
                        default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str,
                        default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1,
                        help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0,
                        help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level',
                        help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0,
                        help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1,
                        help='Sample ratio for each communication round')
    parser.add_argument('--coor', type=float, default=0.0001,
                        help='Parameter controlling the wight of FIM')
    parser.add_argument('--eta', type=float, default=1.0)
    # parameters for KFACOptimizer
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--stat_decay', default=0.95, type=float)
    parser.add_argument('--damping', default=1e-3, type=float)
    parser.add_argument('--kl_clip', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=3e-3, type=float)
    parser.add_argument('--TCov', default=1, type=int)
    parser.add_argument('--TScal', default=10, type=int)
    parser.add_argument('--TInv', default=100, type=int)
    args = parser.parse_args()
    return args


def newton_solve(fed_avg_freqs, Ak, Bk, Z, device):
    return newton_solve_auto_gradient(fed_avg_freqs, Ak, Bk, Z, device)


def newton_solve_auto_gradient(fed_avg_freqs, Ak, Bk, Z, device):
    # HatM = torch.zeros_like(Z).to(device)
    # nn.init.xavier_normal(HatM)
    #print("here_end")
    InverseSumA = torch.inverse(sum([fed_avg_freqs[idx] * Ak[idx] for idx in range(len(Ak))]))
    InverseSumB = torch.inverse(sum([fed_avg_freqs[idx] * Bk[idx] for idx in range(len(Bk))]))
    HatM = InverseSumB @ Z @ InverseSumA
    objective = torch.nn.MSELoss()  # torch.nn.L1Loss()
    ulr = 1.0  # update learning rate
    # gradient_cap = 10

    for i in range(10000):
        objective.zero_grad()
        HatM.requires_grad_()
        MuZ = sum([fed_avg_freqs[idx] * Bk[idx] @ HatM @ Ak[idx]
                   for idx in range(len(Ak))])
        # if i == 0: print("start - newton solve error ", (MuZ - Z).abs().mean())
        loss = objective(MuZ, Z)
        loss.backward(retain_graph=True)
        with torch.no_grad():
            gradient = ulr * HatM.grad.data.clone().detach()
            # gradient[gradient > gradient_cap] = gradient_cap
            # gradient[gradient < -gradient_cap] = - gradient_cap
            HatM = HatM - gradient
    # print("end - newton solve error ", (MuZ - Z).abs().mean())
    return HatM

def newton_solve_partial_gradient(fed_avg_freqs, Ak, Bk, Z, device):
    HatM = torch.zeros_like(Z).to(device)
    J = torch.ones_like(Z).to(device)
    GradientF = sum([fed_avg_freqs[idx] * Bk[idx] @ J @ Ak[idx]
                     for idx in range(len(Ak))])
    for i in range(10000):
        fHatM = sum([fed_avg_freqs[idx] * Bk[idx] @ HatM @ Ak[idx]
                     for idx in range(len(Ak))]) - Z
        HatM = HatM - fHatM / GradientF
    # print(" newton solve error ", fHatM.abs().mean())
    return HatM


def newton_solve_expectation_approximate(fed_avg_freqs, Ak, Bk, Z, device):
    InverseSumA = torch.inverse(sum([fed_avg_freqs[idx] * Ak[idx] for idx in range(len(Ak))]))
    InverseSumB = torch.inverse(sum([fed_avg_freqs[idx] * Bk[idx] for idx in range(len(Bk))]))

    HatM = torch.zeros_like(Z).to(device)

    for i in range(1):
        fHatM = sum([fed_avg_freqs[idx] * Bk[idx] @ HatM @ Ak[idx]
                     for idx in range(len(Ak))]) - Z
        error = fHatM.abs().mean()
        # if error<1e-7:
        #     break
        HatM = HatM - InverseSumB @ fHatM @ InverseSumA
    print(" newton solve error ", error)
    return HatM


def run(net,gt_data,gt_label):
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'oursexperiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'ours_'+args.partition+'_'+str(args.beta)+'_our_sexperiment_log-%s' % (
            datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)
    logger.info(args)
    logger.info("#" * 100)



    if args.alg == 'blockofflinenewton':
        logger.info("Initializing nets")
        nets = {net_i: None for net_i in range(1)}
        nets[0] = net


        global_model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
        #global_model = LeNet(channel=1, hideen=588, num_classes=10)
        param_shapes = {}

        global_fim = OrderedDict()
        for name, module in global_model.named_modules():
            if module.__class__.__name__ in {'Linear', 'Conv2d'}:
                global_fim[name] = {}

        for round in range(args.comm_round):

            global_para = global_model.state_dict()
            for key in global_model.state_dict():
                param_shapes[key] = global_para[key].shape

            fed_avg_freqs = [1.0]
            global_temp_para = {}
            net_para = nets[0].state_dict()
            fnet={}
            kfac_optimizer = KFACOptimizer(net,
                                           lr=1e-2,
                                           momentum=args.momentum,
                                           stat_decay=args.stat_decay,
                                           damping=args.damping,
                                           kl_clip=args.kl_clip,
                                           weight_decay=args.weight_decay,
                                           TCov=args.TCov,
                                           TInv=args.TInv)
            kfac_optimizer.zero_grad()
            out_now = net(gt_data)
            criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(out_now, gt_label)
            kfac_optimizer.acc_stats = True
            loss.backward()

            prior_lambda = 1e-3
            for module in kfac_optimizer.modules:
                name = kfac_optimizer.module2name[module]
                if name not in fnet:
                    fnet[name] = [None, None]
                fnet[name][0] = (
                                        len(gt_label) / 1) * copy.deepcopy(
                    kfac_optimizer.m_aa[module])
                fnet[name][0] += prior_lambda * torch.eye(fnet[name][0].shape[0]).to(device)
                fnet[name][1] = (
                                        len(gt_label) / 1) * copy.deepcopy(
                    kfac_optimizer.m_gg[module])
                fnet[name][1] += prior_lambda * torch.eye(fnet[name][1].shape[0]).to(device)

            for name in fnet:

                weight_para = net_para[name + '.weight']
                if len(weight_para.shape) == 4:
                    weight_para = weight_para.reshape(
                        param_shapes[name + '.weight'][0],
                        param_shapes[name + '.weight'][1] * param_shapes[name + '.weight'][2] *
                        param_shapes[name + '.weight'][3])


                if name + '.bias' in net_para.keys():
                    new_param = torch.cat([weight_para, net_para[name + '.bias'].unsqueeze(1)], dim=1)
                else:
                    new_param = copy.deepcopy(weight_para)
                # B_k@M_k@A_k
                new_param = fnet[name][1] @ new_param @ fnet[name][0]
                global_temp_para[name] = new_param


            newtons_param = copy.deepcopy(nets[0].state_dict())
            for name in global_fim.keys():
                _newtons_param = newton_solve(fed_avg_freqs,
                                              Ak=[fnet[name][0]],
                                              Bk=[fnet[name][1]],
                                              Z=global_temp_para[name],
                                              device=device)

                if name + '.bias' in nets[0].state_dict().keys():

                    newtons_param[name + '.weight'] = _newtons_param[:,
                                                      :-1].reshape(param_shapes[name + '.weight'])
                    newtons_param[name + '.bias'] = _newtons_param[:, -1]
                else:
                    newtons_param[name + '.weight'] = _newtons_param.reshape(
                        param_shapes[name + '.weight'])


            global_model.load_state_dict(newtons_param)
            global_model.to(device='cuda:0')
            return global_model


def main():
    loss_DLG = [None]
    loss_iDLG = [None]
    mse_DLG = [None]
    mse_iDLG = [None]
    all_loss_DLG=[]
    all_loss_iDLG=[]
    all_mse_DLG=[]
    all_mse_iDLG=[]
    dataset = 'FMNIST'
    root_path = '.'
    data_path = os.path.join(root_path, './data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/iDLG_%s' % dataset).replace('\\', '/')
    lr = 1.0
    num_dummy = 1
    Iteration = 200
    num_exp = 500

    use_cuda = torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if dataset == 'FMNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.FashionMNIST(data_path, download=True)




    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        net =  SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
        #net = LeNet(channel=1, hideen=588, num_classes=10)
        net.apply(weights_init)

        print('running %d|%d experiment' % (idx_net, num_exp))
        net = net.to(device)
        idx_shuffle = np.random.permutation(len(dst))

        for method in ['iDLG','DLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            # compute original gradient
            if method =='iDLG':#只应该更新一次
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-2, weight_decay=1e-5)
                out = net(gt_data)
                loss = criterion(out, gt_label)
                loss.backward()
                optimizer.step()


                net=run(net,gt_data,gt_label)
            out = net(gt_data)
            print(out)
            if out.isnan().any():
                print("yes")
                break
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
            elif method == 'iDLG':
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                    (1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            print('lr =', lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    # if iters==0:
                    #     print(pred)
                    if method == 'DLG':
                        dummy_loss = - torch.mean(
                            torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                        #print(torch.softmax(pred,-1))
                        # dummy_loss = criterion(pred, gt_label)
                        # if iters == 0:
                        #     print(dummy_label)
                        #     print("========")
                    elif method == 'iDLG':
                        if pred.isnan().all():
                            pred = torch.ones_like(pred)
                        dummy_loss = criterion(pred, label_pred)
                        # if iters == 0:
                        #     print(label_pred)
                        #     print("========")

                    #change  allow_unused=True
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                if iters % 20 == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))


                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx])
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'DLG':
                            plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'iDLG':
                            plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()

                    if current_loss < 0.000001:  # converge
                        print("convege")
                        break

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses

        print('imidx_list:', imidx_list)
        print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
        print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
        # print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)

        all_loss_DLG.append(loss_DLG[-1])
        all_loss_iDLG.append(loss_iDLG[-1])
        all_mse_DLG.append(mse_DLG[-1])
        all_mse_iDLG.append(mse_iDLG[-1])

        print(all_loss_DLG)
        print(all_loss_iDLG)
        print(all_mse_DLG)
        print(all_mse_iDLG)

        print('----------------------\n\n')





if __name__ == '__main__':
    main()
            
#python3 -W ignore experiments_riem_with_attack.py  --model=simple-cnn --dataset=fmnist --alg=blockofflinenewton --lr=0.01  --batch-size=64  --epochs=1  --n_parties=1  --rho=0.9  --comm_round=1  --partition=noniid-labeldir --beta=0.1 --device='cuda:0'  --datadir='./data/'  --logdir='./logs/'  --noise=0  --init_seed=0  --coor=0.99

