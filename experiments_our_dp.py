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
from utils import * #utils_dp
from vggmodel import *
from resnetcifar import *
from kfac import KFACOptimizer
from torch.distributions import Laplace
laplace_distribution = Laplace(0,0.125)

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


def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16,8]
                    elif args.dataset == 'fmnist':
                        input_size = 784
                        output_size = 10
                        hidden_sizes = [254, 64]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("cifar100"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=100)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ('emnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=37)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net_riem(train_net_func, nets, selected, f_nets, mu_global, f_global, args, net_dataidx_map,
                         coordinator, test_dl=None, device="cpu"):
    avg_acc = 0.0
    #print("here_in")
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" %
                    (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(
                args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(
                args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(
            args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        #print("here_here")
        trainacc, testacc = train_net_func(net_id, net, f_nets[net_id], mu_global, f_global, len(
            dataidxs), train_dl_local, test_dl, n_epoch, args.lr, coordinator, args, device=device)
        #print("here_here_done")
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_blockoffline(nets, selected, f_nets, mu_global, f_global, args, net_dataidx_map, test_dl=None,
                                 device="cpu"):
    return local_train_net_riem(train_net_blockoffline, nets, selected, f_nets, mu_global, f_global, args,
                                net_dataidx_map, args.coor, test_dl, device)


def train_net_blockoffline(net_id, net_local, f_local, mu_global, f_global, len_of_data, train_dataloader,
                           test_dataloader, epochs, lr, coordinator, args, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net_local, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(
        net_local, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, net_local.parameters()), lr=lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net_local.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net_local.parameters(
        )), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()
    def posterior_loss(f_global, f_local, mu_global, net):
        result = 0.0
        mu_local = net.state_dict()
        for params_name in f_local:
            if params_name in f_global.keys() and len(f_global[params_name]) > 0:
                weight_local = mu_local[params_name + '.weight']
                weight_global = mu_global[params_name + '.weight']
                if len(weight_local.shape) == 4:
                    weight_local = weight_local.reshape(
                        param_shapes[params_name + '.weight'][0],
                        param_shapes[params_name + '.weight'][1] * param_shapes[params_name + '.weight'][2] *
                        param_shapes[params_name + '.weight'][3])
                    weight_global = weight_global.reshape(
                        param_shapes[params_name + '.weight'][0],
                        param_shapes[params_name + '.weight'][1] * param_shapes[params_name + '.weight'][2] *
                        param_shapes[params_name + '.weight'][3])
                if params_name + '.bias' in mu_local.keys():  # [out_c,in_c*k_h*k_w] [out_c]
                    # have bias for this layer [2,10] [2]
                    diff = torch.cat(
                        [weight_local, mu_local[params_name + '.bias'].unsqueeze(1).to(device)], dim=1) \
                           - torch.cat(
                        [weight_global.to(device), mu_global[params_name + '.bias'].unsqueeze(1).to(device)], dim=1)
                else:
                    diff = weight_local - weight_global
                result += (diff * ((f_global[params_name][1] - f_local[params_name][1]) @ diff @ (
                            f_global[params_name][0] - f_local[params_name][0]))).sum()
        return 0.5 * result

    for epoch in range(epochs):
        #print("here I am in just slow")
        epoch_loss_collector = []
        epoch_posterior_loss = 0.0
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                #print(batch_idx)
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                target = target.long()
                out = net_local(x)
                loss = criterion(out, target)
                ploss = args.eta * posterior_loss(f_global, f_local, mu_global, net_local)
                loss = loss + ploss if ploss > 0 else loss
                loss.backward()
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_posterior_loss += ploss.item() if ploss > 0 else 0
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f Posterior Loss: %f' % (
        epoch, epoch_loss, epoch_posterior_loss / len(epoch_loss_collector)))
    del optimizer
    kfac_optimizer = KFACOptimizer(net_local,
                                   lr=lr,
                                   momentum=args.momentum,
                                   stat_decay=args.stat_decay,
                                   damping=args.damping,
                                   kl_clip=args.kl_clip,
                                   weight_decay=args.weight_decay,
                                   TCov=args.TCov,
                                   TInv=args.TInv)

    for tmp in train_dataloader:
        for batch_idx, (x, target) in enumerate(tmp):
            x, target = x.to(device), target.to(device)
            kfac_optimizer.zero_grad()
            # x.requires_grad = True
            # target.requires_grad = False
            target = target.long()
            out = net_local(x)
            loss = criterion(out, target)

            kfac_optimizer.acc_stats = True
            loss.backward()
            for module in kfac_optimizer.modules:
                name = kfac_optimizer.module2name[module]
                if batch_idx == 0:
                    f_local[name][0] = (
                                               len(target) / len_of_data) * copy.deepcopy(kfac_optimizer.m_aa[module])
                    f_local[name][1] = (
                                               len(target) / len_of_data) * copy.deepcopy(kfac_optimizer.m_gg[module])
                else:
                    f_local[name][0] += (len(target) / len_of_data) * \
                                        copy.deepcopy(kfac_optimizer.m_aa[module])
                    f_local[name][1] += (len(target) / len_of_data) * \
                                        copy.deepcopy(kfac_optimizer.m_gg[module])

    for hook in kfac_optimizer.hooks:
        hook.remove()
    del kfac_optimizer
    for name in f_local:
        # adding I into the Fisher information and minimizing the approximation error from this add
        pi = 1.0
        coor = sqrt(1 - coordinator)
        # s1 = torch.norm(f_local[name][0], p=2)
        # s2 = torch.norm(f_local[name][1], p=2)
        # pi = torch.sqrt(s1/s2)
        f_local[name][0] = f_local[name][0] + pi * coor * \
                           torch.eye(f_local[name][0].shape[0]).to(device)
        f_local[name][1] = f_local[name][1] + 1 / pi * coor * \
                           torch.eye(f_local[name][1].shape[0]).to(device)

    train_acc = compute_accuracy(net_local, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(
        net_local, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    logger.info(' ** Training complete **')
    return train_acc, test_acc


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map


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


if __name__ == '__main__':
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

    seed = args.init_seed
    logger.info(args)
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)

    print("len train_dl_global:", len(train_ds_global))

    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
                    args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id,
                    args.n_parties - 1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
                    args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(
            dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(
            dataset=test_all_in_ds, batch_size=32, shuffle=False)

    if args.alg == 'our':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(
            args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(
            args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_para = global_model.state_dict()

        param_shapes = {}
        fim_shapes = {}

        global_fim = OrderedDict()
        for name, module in global_model.named_modules():
            if module.__class__.__name__ in {'Linear', 'Conv2d'}:
                global_fim[name] = {}

        f_nets = [copy.deepcopy(global_fim) for i in range(args.n_parties)]
        #print("here1")
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()

            for key in global_model.state_dict():
                param_shapes[key] = global_para[key].shape

            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_blockoffline(
                nets, selected, f_nets, global_para, global_fim, args, net_dataidx_map, test_dl=test_dl_global,
                device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)
            #print("here2")
            # update global model
            total_data_points = sum([len(net_dataidx_map[r])
                                     for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) /
                             total_data_points for r in selected]


            # avg_param
            avg_param = OrderedDict()
            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                for key in net_para:
                    if idx == 0:
                        temp1=laplace_distribution.sample(net_para[key].shape).cuda()
                        avg_param[key] = fed_avg_freqs[idx] * (net_para[key]+temp1)#+ 2 * (torch.rand_like(net_para[key]) - torch.ones_like(net_para[key]))* 0.025 * ( torch.max(net_para[key]) - torch.min(net_para[key])))
                    else:
                        temp1 = laplace_distribution.sample(net_para[key].shape).cuda()
                        avg_param[key] +=fed_avg_freqs[idx] * (net_para[key]+ temp1)#+ 2 * (torch.rand_like(net_para[key]) - torch.ones_like(net_para[key])) * 0.025 * ( torch.max(net_para[key]) - torch.min(net_para[key])))
            #print("here3")
            # newtons_param and biased_param
            newtons_param = copy.deepcopy(avg_param)
            biased_param = copy.deepcopy(avg_param)
            global_temp_para = {}
            #print("here4")
            ####___
            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()
                f_local = f_nets[selected[idx]]
                for name in f_local:  # module_name -->> 0:A, 1:G
                    weight_para = net_para[name + '.weight']
                    if len(weight_para.shape) == 4:
                        weight_para = weight_para.reshape(
                            param_shapes[name + '.weight'][0],
                            param_shapes[name + '.weight'][1] * param_shapes[name + '.weight'][2] *
                            param_shapes[name + '.weight'][3])
                    if name + '.bias' in newtons_param.keys():  # [out_c,in_c*k_h*k_w] [out_c]
                        # have bias for this layer [2,10] [2]
                        new_param = torch.cat(
                            [weight_para, net_para[name + '.bias'].unsqueeze(1)], dim=1)
                    else:
                        new_param = copy.deepcopy(weight_para)
                    # B_k@M_k@A_k
                    new_param = f_local[name][1] @ new_param @ f_local[name][0]

                    if idx == 0:
                        global_fim[name][0] = fed_avg_freqs[idx] * \
                                              f_local[name][0]
                        global_fim[name][1] = fed_avg_freqs[idx] * \
                                              f_local[name][1]
                        global_temp_para[name] = fed_avg_freqs[idx] * new_param
                    else:
                        global_fim[name][0] += fed_avg_freqs[idx] * \
                                               f_local[name][0]
                        global_fim[name][1] += fed_avg_freqs[idx] * \
                                               f_local[name][1]
                        global_temp_para[name] += fed_avg_freqs[idx] * new_param
            #print("here5")
            for name in global_fim.keys():
                # computing the global expectations
                # mlp 0.4* 0.7* 0.7* backpack-->kfac
                # cnn 0.3* 0.1 0.5*
                # biased global expectations by Expectation Approximation
                _biased_param = torch.inverse(
                    global_fim[name][1]) @ global_temp_para[name] @ torch.inverse(global_fim[name][0])
                # newtons global expectations by Newton Approximation
                # print(name)
                # for idx in range(len(selected)):
                #     print(idx)
                #     print(torch.max(f_nets[selected[idx]][name][0]))
                #     print(torch.min(f_nets[selected[idx]][name][0]))
                #     print(torch.max(f_nets[selected[idx]][name][1]))
                #     print(torch.min(f_nets[selected[idx]][name][1]))
                #     print(torch.max(global_temp_para[name]))
                #     print(torch.min(global_temp_para[name]))
                _newtons_param = newton_solve(fed_avg_freqs,
                                              Ak=[f_nets[selected[idx]][name][0]+ laplace_distribution.sample(f_nets[selected[idx]][name][0].shape).cuda()#+(2*torch.rand_like(f_nets[selected[idx]][name][0])-torch.ones_like(f_nets[selected[idx]][name][0]))*0.025*(torch.max(f_nets[selected[idx]][name][0])-torch.min(f_nets[selected[idx]][name][0]))
                                                  for idx in range(len(selected))],
                                              Bk=[f_nets[selected[idx]][name][1]+ laplace_distribution.sample(f_nets[selected[idx]][name][1].shape).cuda()#+(2*torch.rand_like(f_nets[selected[idx]][name][1])-torch.ones_like(f_nets[selected[idx]][name][1]))*0.025*(torch.max(f_nets[selected[idx]][name][1])-torch.min(f_nets[selected[idx]][name][1]))
                                                  for idx in range(len(selected))],
                                              Z=global_temp_para[name]+laplace_distribution.sample(global_temp_para[name].shape).cuda(),#+(2*torch.rand_like(global_temp_para[name])-torch.ones_like(global_temp_para[name]))*0.025*(torch.max(global_temp_para[name])-torch.min(global_temp_para[name])),
                                              device=device)

                if name + '.bias' in biased_param.keys():
                    biased_param[name + '.weight'] = _biased_param[:,
                                                     :-1].reshape(param_shapes[name + '.weight'])
                    biased_param[name + '.bias'] = _biased_param[:, -1]
                    newtons_param[name + '.weight'] = _newtons_param[:,
                                                      :-1].reshape(param_shapes[name + '.weight'])
                    newtons_param[name + '.bias'] = _newtons_param[:, -1]
                else:
                    biased_param[name + '.weight'] = _biased_param.reshape(
                        param_shapes[name + '.weight'])
                    newtons_param[name + '.weight'] = _newtons_param.reshape(
                        param_shapes[name + '.weight'])

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))
            ####___
            global_model.load_state_dict(avg_param)
            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(
                global_model, test_dl_global, get_confusion_matrix=True)
            logger.info('>> fedavg Global Model Train accuracy: %f' %
                        train_acc)
            logger.info('>> fedavg Global Model Test accuracy: %f' % test_acc)
            print('>> fedavg Global Model Train accuracy: %f' % train_acc)
            print('>> fedavg Global Model Test accuracy: %f' % test_acc)

            global_model.load_state_dict(newtons_param)
            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(
                global_model, test_dl_global, get_confusion_matrix=True)
            logger.info('>> newtons Global Model Train accuracy: %f' %
                        train_acc)
            logger.info('>> newtons Global Model Test accuracy: %f' % test_acc)
            print('>> newtons Global Model Train accuracy: %f' % train_acc)
            print('>> newtons Global Model Test accuracy: %f' % test_acc)

            global_model.load_state_dict(biased_param)
            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(
                global_model, test_dl_global, get_confusion_matrix=True)
            logger.info('>> biased Global Model Train accuracy: %f' %
                        train_acc)
            logger.info('>> biased Global Model Test accuracy: %f' % test_acc)
            print('>> biased Global Model Train accuracy: %f' % train_acc)
            print('>> biased Global Model Test accuracy: %f' % test_acc)

            # using newtons' result for the next round
            global_model.load_state_dict(newtons_param)
            
            
#python3 -W ignore experiments_riem_selfKFAC.py  --model=simple-cnn --dataset=cifar10 --alg=blockofflinenewton --lr=0.01  --batch-size=64  --epochs=1  --n_parties=2  --rho=0.9  --comm_round=1  --partition=noniid-labeldir --beta=0.1 --device='cuda:0'  --datadir='./data/'  --logdir='./logs/'  --noise=0  --init_seed=0  --coor=0.99

