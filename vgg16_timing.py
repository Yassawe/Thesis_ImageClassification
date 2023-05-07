"""
all research code is always a mess, i didn't care about clean code or anything like that here
"""

import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import random
import numpy as np
import pandas as pd
import torch.cuda.profiler as profiler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def setrandom(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--lr', default = 1e-3, type=float)

    parser.add_argument('--name', default="VGG16", type=str)
    parser.add_argument('--experiment', default="baseline", type=str)


    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2023'


    os.environ['NCCL_ALGO'] = 'Ring'

    train_dataset = torchvision.datasets.CIFAR10(root='../datasets',
                                               train=True,
                                               transform=train_transform,
                                               download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='../datasets',
                                                train=False,
                                                transform=test_transform,
                                                download=True)


    mp.spawn(train, nprocs=args.gpus, args=(train_dataset, test_dataset, args,))

def train(gpu, train_dataset, test_dataset, args):
    #DISTRIBUTED
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', world_size=args.gpus, rank=gpu)


    #SETUPS
    setrandom(20214229)
    filename = "./"+args.experiment+"/"+args.name
    testdump = filename+"TEST_ACC.txt"
    traindump = filename+"TRAIN_ACC.txt"
    ext = ".csv"


    model = torchvision.models.vgg16(weights=None)


    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    #HYPERPARAMETERS

    batch_size = 32//args.gpus # global batch size of 256
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    #DATASETS                           
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpus, rank=gpu)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=train_sampler)



    total_step = len(train_loader)


    model.train()

    idx = 0
    # target_iter=100

    # time_iters=[i for i in range(100,200)]
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True) 

    # runtimes = [] 

    total_step = len(train_loader)

    if gpu==0:
        profiler.start()

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):

            idx += 1
            images = images.cuda(gpu, non_blocking=True)

            labels = labels.cuda(gpu, non_blocking=True)


            # if idx in time_iters:
            #     start.record()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if gpu == 0:
                print('Epoch [{}/{}]. Step [{}/{}], Loss: {:.4f}'.format(epoch, args.epochs, i + 1, total_step, loss.item()))
        
        scheduler.step()

        #     if idx in time_iters:
        #         end.record()
        #         torch.cuda.synchronize()
        #         runtimes.append(start.elapsed_time(end))

        #     if idx>=time_iters[-1]:
        #         break

        # if idx>=time_iters[-1]:
        #     break

    if gpu==0:
        profiler.stop()
    

    #print("GPU {}. Model {}. Average iteration time = {}".format(gpu, args.name, sum(runtimes)/len(runtimes)))

def evaluation(model, gpu, epoch, dataloader, filename, evalname, args, scheduler):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

    if evalname=="Test set":
        scheduler.step(accuracy)

    model.train()
   

if __name__ == '__main__':
    main()
