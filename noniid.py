import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F
import numpy
import copy
import random
import numpy as np
# from Subdata import  MNIST_truncated,CIFAR10_truncated


# import trainmodel as approach
# from do_ot import doot
import modelset as network
from utils_mae import noniid_aggr,createp #,valid,noniid_avgmodel_o,noniid_ensemble,partition,
import argparse
from utils import *
import logging
import datetime



def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # if args_optimizer == 'adam':
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    # elif args_optimizer == 'amsgrad':
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
    #                            amsgrad=True)
    # elif args_optimizer == 'sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        #train_acc = compute_accuracy(net, train_dataloader, device=device)
        #test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        #writer.add_scalar('Accuracy/train', train_acc, epoch)
        #writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.01,type=float,)
    parser.add_argument('--model_type', default='mlpnet', help='mlpnet for mnist and cnnnet for cifar')
    parser.add_argument('--data', default='mnist',help='mnist or cifar')
    parser.add_argument('--n_nets', default=5, type=int, help='model nums')
    parser.add_argument('--diff_init', default=True, help='True means different init')
    parser.add_argument('--maxt_times', default=50,type=int, help='iterate times')
    parser.add_argument('--C', default=0.5,type=float,)
    parser.add_argument('--norm', default=False)
    parser.add_argument('--test', default=True, help='If True, record all accuracy rates of each iteration. If just want to see accuracy of aggregation, set False.')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU id to use')
    parser.add_argument('--seed', default=1,type=int,)
    parser.add_argument('--lambdastep', default=1.6,type=float, help='step_size')
    parser.add_argument('--split',  default=True, help='whether to discard the untrained parameters of the last layer')
    parser.add_argument('--batch_size', default=64,type=int,)
    parser.add_argument('--learning_rate', default=0.01,type=float,)
    parser.add_argument('--num_epochs', default=10,type=int,)
    parser.add_argument('--alter', default=True, help='alternate or parallel')
    # parser.add_argument('--logdir', default='logfinal')
    parser.add_argument('--expe', default='agg1',help='name of experiment')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--other', default="", type=str, help='seed for initializing training.')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mkdirs(args.logdir)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


    log_file_name = 'Noniid_'+args.partition+'_'+str(args.alpha)+'_noniid_experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))

    log_path = log_file_name + '.log'
    temp_filename=os.path.join(args.logdir, log_path)
    logger = logging.getLogger()
    logging.basicConfig(
        filename=temp_filename,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        level=logging.DEBUG,
        filemode='w')

    logger.info(args)
    logger.info("#" * 100)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    models = []

    splits=[[] for i in range(args.n_nets)]

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
         args.data, args.datadir, args.logdir, args.partition, args.n_nets, beta=args.alpha)
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.data,
                                                                                      args.datadir,                                                                                      args.batch_size,
                                                                                      32)
    testdata=test_dl_global


    if args.diff_init:
        if args.model_type=='mlpnet':
            for mnum in range(args.n_nets):
                models.append(network.mnistnet().cuda(args.gpu_id))
        elif args.model_type=='cnnnet':
            for mnum in range(args.n_nets):
                models.append(network.cnnNet().cuda(args.gpu_id))
        elif args.model_type=='svhn_cnn':
            for mnum in range(args.n_nets):
                models.append( SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).cuda(args.gpu_id)) #
        elif args.model_type=='cifar100_cnn':
            for mnum in range(args.n_nets):
                models.append(SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=100).cuda(args.gpu_id))
        elif args.model_type=='cifar10_cnn':
            for mnum in range(args.n_nets):
                models.append( SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).cuda(args.gpu_id)) #.cuda(args.gpu_id)
        elif args.model_type=='mnist_cnn':
            for mnum in range(args.n_nets):
                models.append(SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).cuda(args.gpu_id))
        elif args.model_type=='fmnist_cnn':
            for mnum in range(args.n_nets):
                models.append(SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).cuda(args.gpu_id))
    else:
        if args.model_type=='mlpnet':
            models.append(network.mnistnet().cuda(args.gpu_id))
        elif args.model_type=='cnnnet':
            models.append(network.cnnNet().cuda(args.gpu_id))
        elif args.model_type=='svhn_cnn':
            models.append( SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).cuda(args.gpu_id)) #
        elif args.model_type=='cifar100_cnn':
            models.append(SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=100).cuda(args.gpu_id))
        elif args.model_type=='cifar10_cnn':
            models.append( SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).cuda(args.gpu_id))
        elif args.model_type=='mnist_cnn':
            models.append(SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).cuda(args.gpu_id))
        elif args.model_type=='fmnist_cnn':
            models.append(SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).cuda(args.gpu_id))
        for mnum in range(1,args.n_nets):
            models.append(copy.deepcopy(models[0]))

    acc = []
    logger.info('Training...')
    print(('Training...'))
    for i,model in enumerate(models):


        dataidxs = net_dataidx_map[i]
        noise_level = 0 / (args.n_nets - 1) * i
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.data, args.datadir, args.batch_size, 32,
                                                             dataidxs, noise_level)

        trainacc, testacc = train_net(i, model, train_dl_local, testdata, args.num_epochs, args.learning_rate, 'sgd',
                                      device=args.device)
        logger.info("net %d final test acc %f" % (i, testacc))

    logger.info('Training is OK')
    print('Training is OK')

    print('prepare P')
    logger.info('Prepare P')
    pps = []
    for i,model in enumerate(models):

        dataidxs = net_dataidx_map[i]
        noise_level = 0 / (args.n_nets - 1) * i
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.data, args.datadir, args.batch_size, 32,
                                                             dataidxs, noise_level)
        pps.append(createp(model.cuda(args.gpu_id),train_dl_local,args = args)) #.cuda(args.gpu_id),


    print('start aggregation ours !')
    logger.info('MAE')
    ours  = noniid_aggr(models,pps,testdata,splits=splits,args = args)
    logger.info("global accuracy: %f",ours)
    print('done!')


    # log = {
    #     'ours':{
    #         'acc':ours,
    #
    #     },
    #     'all_model_acc':np.array(acc),
    #     'partition':traindata_cls_counts
    #     }


    # logger.info("acc:%f",ours)
    # logger.info("all_model_acc:%f",np.array(acc))
    # logger.info(traindata_cls_counts)

# old ones
    # python3  noniid.py - -alpha 0.1 - -model_type cnnnet - -data  cifar - -n_nets
    # 2 - -diff_init
    # True - -norm
    # True - -maxt_times
    # 300 - -C
    # 0.5 - -test
    # True - -lambdastep
    # 0.05 - -logdir
    # logfinal - -expe
    # 2
    # cnn - -num_epochs = 1



#best
#python3  noniid.py --alpha 0.1 --model_type cifar10_cnn --data cifar10 --n_nets 2 --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05  --num_epochs=1  --partition noniid-labeldir   --device cuda:0
#python3  noniid.py --alpha 0.1 --model_type cifar100_cnn --data cifar100 --n_nets 2 --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05  --num_epochs=1  --partition noniid-labeldir   --device cuda:0
#python3  noniid.py --alpha 0.1 --model_type svhn_cnn       --data svhn --n_nets 2 --diff_init False --norm False --maxt_times 100 --C 0.5 --test True --lambdastep 0.05   --num_epochs=1  --partition noniid-labeldir   --device cuda:0
#python3  noniid.py --alpha 0.1 --model_type mnist_cnn --data mnist --n_nets 2 --diff_init False --norm False --maxt_times 50 --C 0.5 --test True --lambdastep 0.05   --num_epochs=1  --partition noniid-labeldir   --device cuda:0
#python3  noniid.py --alpha 0.1 --model_type fmnist_cnn --data fmnist --n_nets 2 --diff_init False --norm False --maxt_times 50 --C 0.5 --test True --lambdastep 0.05   --num_epochs=1  --partition noniid-labeldir   --device cuda:0

#cifar100 GG
# ??? n_nets number matters?
#python3  noniid.py --alpha 0.1 --model_type cifar10_cnn --data cifar10 --n_nets 10 --diff_init False --norm False --maxt_times 300 --C 0.5 --test True --lambdastep 0.05   --num_epochs=1  --partition noniid-#label2  --device cuda:0
