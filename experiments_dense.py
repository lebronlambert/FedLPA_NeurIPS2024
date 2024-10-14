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

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *





from tqdm import tqdm
import argparse
import copy
import os

import warnings
# import torchvision.models as models
import numpy as np


###########here
# from helpers.datasets import partition_data
# from torch.utils.data import DataLoader, Dataset
from helpers.utils import average_weights, KLDiv, setup_seed, test #get_dataset,
##########
from helpers.synthesizers import AdvSynthesizer

from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar100

import torch.nn.functional as F

from models.resnet import resnet18
# from models.vit import deit_tiny_patch16_224
# import wandb

warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)

class LocalUpdate(object):
    def __init__(self, args, dataloader):# ,#dataset, idxs):
        self.args = args
        self.train_loader= dataloader
        # #self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
        #                                batch_size=self.args.local_bs, shuffle=True, num_workers=4)

    def update_weights(self, model, client_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)
        # label_list = [0] * 100
        # for batch_idx, (images, labels) in enumerate(self.train_loader):
        #     for i in range(100):
        #         label_list[i] += torch.sum(labels == i).item()
        # print(label_list)
        local_acc_list = []
        for iter in tqdm(range(self.args.local_ep)):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                # ---------------------------------------
                output = model(images)
                loss = F.cross_entropy(output, labels)
                # ---------------------------------------
                loss.backward()
                optimizer.step()
            acc, test_loss = test(model, test_loader)
            # if client_id == 0:
            #     wandb.log({'local_epoch': iter})
            # wandb.log({'client_{}_accuracy'.format(client_id): acc})
            local_acc_list.append(acc)
        return model.state_dict(), np.array(local_acc_list)


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to non-IID. Set to 0 for non-IID.')

    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for adv loss')

    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='noniid-labeldir', type=str)
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--g_steps', default=20, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--other', default="", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    args = parser.parse_args()
    return args



class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


def kd_train(synthesizer, model, criterion, optimizer):
    student, teacher = model
    student.train()
    teacher.eval()
    description = "loss={:.4f} acc={:.2f}%"
    total_loss = 0.0
    correct = 0.0
    with tqdm(synthesizer.get_data()) as epochs:
        for idx, (images) in enumerate(epochs):
            optimizer.zero_grad()
            images = images.cuda()
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(synthesizer.data_loader.dataset) * 100

            epochs.set_description(description.format(avg_loss, acc))


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def get_model(args):
    if args.model == "mnist_cnn":
        global_model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).cuda()
        #global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        global_model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).cuda()
        #global_model = CNNMnist().cuda()
    elif args.model == "cnn":
        global_model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).cuda()
        #global_model = CNNCifar().cuda()
    elif args.model == "svhn_cnn":
        global_model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).cuda()
        #global_model = CNNCifar().cuda()
    elif args.model == "cifar100_cnn":
        global_model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=100).cuda()
        #global_model = CNNCifar100().cuda()
    elif args.model == "res":
        # global_model = resnet18()
        global_model = resnet18(num_classes=10).cuda() #10
        
    elif args.model == "vgg-9":
	    if args.dataset in ("mnist", 'femnist'):
		net = ModerateCNNMNIST()
	    elif args.dataset in ("cifar10", "cinic10", "svhn"):
		# print("in moderate cnn")
		net = ModerateCNN()
	    elif args.dataset == 'celeba':
		net = ModerateCNN(output_dim=2)

    # elif args.model == "vit":
    #     global_model = deit_tiny_patch16_224(num_classes=1000,
    #                                          drop_rate=0.,
    #                                          drop_path_rate=0.1)
    #     global_model.head = torch.nn.Linear(global_model.head.in_features, 10)
    #     global_model = global_model.cuda()
    #     global_model = torch.nn.DataParallel(global_model)
    return global_model




if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = args_parser()
    mkdirs(args.logdir)
    argument_path='dense_experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)


    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_path='dense_'+args.partition+'_'+str(args.beta)+'_dense_experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")) +'.log'

    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    seed = args.seed
    logger.info(args)
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args.datadir, args.logdir, args.partition, args.num_users, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)
    print("len train_dl_global:", len(train_ds_global))
    data_size = len(test_ds_global)
    train_all_in_list = []
    test_all_in_list = []



    logger.info("Initializing nets")


    #============
    # train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
    #     args.dataset, args.partition, beta=args.beta, num_users=args.num_users)
    #
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
    #                                           shuffle=False, num_workers=4)
    test_loader=test_dl_global
    # BUILD MODEL

    global_model = get_model(args)
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    local_weights = []
    global_model.train()
    acc_list = []
    users = []
    if args.type == "pretrain":
        # ===============================================
        for idx in range(args.num_users):
            print("client {}".format(idx))
            users.append("client_{}".format(idx))
            net_id=idx
            dataidxs = net_dataidx_map[net_id]
            noise_level = 0 / (args.num_users - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
            #local_model = LocalUpdate(args=args, dataset=train_dataset,
            #                          idxs=user_groups[idx])

            local_model = LocalUpdate(args=args,dataloader=train_dl_local)
            w, local_acc = local_model.update_weights(copy.deepcopy(global_model), idx)

            acc_list.append(local_acc)
            local_weights.append(copy.deepcopy(w))

        # wandb

        # for i in range(args.local_ep):
        #     logger.info({"client_{}_acc".format(users[0]):acc_list[0][i],
        #         "client_{}_acc".format(users[1]):acc_list[1][i],
        #         "client_{}_acc".format(users[2]):acc_list[2][i],
        #         "client_{}_acc".format(users[3]):acc_list[3][i],
        #         "client_{}_acc".format(users[4]):acc_list[4][i],
        #     })


        # np.save("client_{}_acc.npy".format(args.num_users), acc_list)
        # logger.info({"client_accuracy" : wandb.plot.line_series(
        #     xs=[ i for i in range(args.local_ep) ],
        #     ys=[ acc_list[i] for i in range(args.num_users) ],
        #     keys=users,
        #     title="Client Accuacy")})
        # torch.save(local_weights, '{}_{}.pkl'.format(name, iid))
        print("###~~~")
        torch.save(local_weights, '{}_{}clients_{}.pkl'.format(args.dataset, args.num_users, args.beta))
        print("###~~~")
        # update global weights by FedAvg
        # global_weights = average_weights(local_weights)
        # global_model.load_state_dict(global_weights)
        # logger.info("avg acc:")
        # test_acc, test_loss = test(global_model, test_loader)
        # model_list = []
        # for i in range(len(local_weights)):
        #     net = copy.deepcopy(global_model)
        #     net.load_state_dict(local_weights[i])
        #     model_list.append(net)
        # ensemble_model = Ensemble(model_list)
        # logger.info("ensemble acc:")
        # test(ensemble_model, test_loader)


        #global_weights = average_weights(local_weights)
        #global_model.load_state_dict(global_weights)
        #logger.info("avg acc:")
        #test_acc, test_loss = test(global_model, test_loader)
        #logger.info(test_acc)
        #model_list = []
        #for i in range(len(local_weights)):
        #    net = copy.deepcopy(global_model)
        #    net.load_state_dict(local_weights[i])
        #    model_list.append(net)
        #ensemble_model = Ensemble(model_list)
        #logger.info("ensemble acc:")
        #acc, test_loss=test(ensemble_model, test_loader)
        #logger.info(acc)


        # ===============================================
    else:
        # ===============================================
        local_weights = torch.load('{}_{}clients_{}.pkl'.format(args.dataset, args.num_users, args.beta))
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        logger.info("avg acc:")
        test_acc, test_loss = test(global_model, test_loader)
        model_list = []
        for i in range(len(local_weights)):
            net = copy.deepcopy(global_model)
            net.load_state_dict(local_weights[i])
            model_list.append(net)
        ensemble_model = Ensemble(model_list)
        logger.info("ensemble acc:")
        test(ensemble_model, test_loader)
        # ===============================================
        global_model = get_model(args)
        # ===============================================

        # data generator
        nz = args.nz
        nc = 3 if "cifar" in args.dataset or args.dataset == "svhn" else 1
        img_size = 32 if "cifar" in args.dataset or args.dataset == "svhn" else 28
        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
        args.cur_ep = 0
        img_size2 = (3, 32, 32) if "cifar" in args.dataset or args.dataset == "svhn" else (1, 28, 28)
        num_class = 100 if args.dataset == "cifar100" else 10
        synthesizer = AdvSynthesizer(ensemble_model, model_list, global_model, generator,
                                     nz=nz, num_classes=num_class, img_size=img_size2,
                                     iterations=args.g_steps, lr_g=args.lr_g,
                                     synthesis_batch_size=args.synthesis_batch_size,
                                     sample_batch_size=args.batch_size,
                                     adv=args.adv, bn=args.bn, oh=args.oh,
                                     save_dir=args.save_dir, dataset=args.dataset)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        criterion = KLDiv(T=args.T)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.9)
        global_model.train()
        distill_acc = []
        for epoch in tqdm(range(args.epochs)):
            # 1. Data synthesis
            synthesizer.gen_data(args.cur_ep)  # g_steps
            args.cur_ep += 1
            kd_train(synthesizer, [global_model, ensemble_model], criterion, optimizer)  # # kd_steps
            acc, test_loss = test(global_model, test_loader)
            distill_acc.append(acc)
            is_best = acc > bst_acc
            bst_acc = max(acc, bst_acc)
            _best_ckpt = 'df_ckpt/{}.pth'.format(args.other)
            logger.info("best acc:{}".format(bst_acc))
            # save_checkpoint({
            #     'state_dict': global_model.state_dict(),
            #     'best_acc': float(bst_acc),
            # }, is_best, _best_ckpt)
            logger.info({'accuracy': acc})
            logger.info({'best accuracy': bst_acc})

        # logger.info({"global_accuracy" :
        #     xs=[ i for i in range(args.epochs) ],
        #     ys=distill_acc,
        #     keys="DENSE",
        #     title="Accuacy of DENSE"})
        # np.save("distill_acc_{}.npy".format(args.dataset), np.array(distill_acc))

        # ===============================================



    #==============



# python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=cnn  --dataset=cifar10 --beta=0.001  \
# --seed=0 --num_users=2 --local_ep=1 --epochs=200 --partition  noniid-labeldir

# python3 experiments_dense.py  --type=kd_train --epochs=2 --lr=0.005 --batch_size 64  \
#  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/cifar10  \
#  --other=cifar10 --model=cnn --dataset=cifar10 --adv=1 --beta=0.001 --seed=1 --num_users 2  --partition  noniid-labeldir
#
#
#
# python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=cifar100_cnn  --dataset=cifar100  \
# --beta=0.001 --seed=0 --num_users=2 --local_ep=1 --epochs=200  --partition  noniid-labeldir

# python3 experiments_dense.py  --type=kd_train  --epochs=2 --lr=0.005 --batch_size 64   \
# --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/cifar100 \
#  --other=cifar100 --model=cifar100_cnn --dataset=cifar100 --adv=1 --beta=0.001 --seed=1 --num_users 2  --partition  noniid-labeldir
#
#
#
# python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=svhn_cnn  --dataset=svhn  \
# --beta=0.001 --seed=0 --num_users=2 --local_ep=1 --epochs=200  --partition  noniid-labeldir

# python3 experiments_dense.py  --type=kd_train  --epochs=2 --lr=0.005 --batch_size 64  \
# --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/svhn  \
# --other=svhn --model=svhn_cnn --dataset=svhn --adv=1 --beta=0.001 --seed=1 --num_users 2  --partition  noniid-labeldir
#
#
# python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=mnist_cnn  --dataset=mnist  \
# --beta=0.001 --seed=0 --num_users=2 --local_ep=1 --epochs=200  --partition  noniid-labeldir

# python3 experiments_dense.py  --type=kd_train --epochs=2 --lr=0.005 --batch_size 64 \
#  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/mnist  \
#  --other=mnist --model=mnist_cnn --dataset=mnist --adv=1 --beta=0.001 --seed=1 --num_users 2  --partition  noniid-labeldir
#
#
# python3 experiments_dense.py --type=pretrain   --lr=0.01 --model=fmnist_cnn   \
# --dataset=fmnist --beta=0.001 --seed=0 --num_users=2 --local_ep=1 --epochs=200  --partition  noniid-labeldir

# python3 experiments_dense.py  --type=kd_train --epochs=2 --lr=0.005 --batch_size 64 \
#  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/fmnist \
#  --other=fmnist --model=fmnist_cnn --dataset=fmnist --adv=1 --beta=0.001 --seed=1 --num_users 2  --partition  noniid-labeldir
#
#
#  python3 experiments_dense.py --type=pretrain  --lr=0.01 --model=cnn  --dataset=cifar10 --beta=0.001  \
# --seed=0 --num_users=10 --local_ep=1 --epochs=200 --partition   noniid-#label1

# python3 experiments_dense.py  --type=kd_train --epochs=2 --lr=0.005 --batch_size 64  \
#  --synthesis_batch_size=256 --g_steps 30 --lr_g 1e-3 --bn 1.0 --oh 1.0 --T 20 --save_dir=run/cifar10  \
#  --other=cifar10 --model=cnn --dataset=cifar10 --adv=1 --beta=0.001 --seed=1 --num_users 10  --partition  noniid-#label1

