"""
by progressive we mean using different degree of pruning on different iterations. here, it is achieved by checkpointing and loading. specify checkpoint folder for an experiment 
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

def save_checkpoint(ddp_model, optimizer, scheduler, epoch, folder, name):
    path = folder+name + ".pt"

    state = {
            'epoch':epoch,
            'model': ddp_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler':scheduler.state_dict()
            }
    
    torch.save(state, path)

def load_checkpoint(rank, model, optimizer, scheduler, path):

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint['model']) 
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint['epoch']

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--lr', default = 1e-3, type=float)

    parser.add_argument('--name', default="VGG16", type=str)
    parser.add_argument('--experiment', default="baseline", type=str)
    parser.add_argument('--chkpt_dump', default="./checkpoints/", type=str)

    parser.add_argument('--checkpoint', default=None, type=str)


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

    customGroup = dist.new_group(backend='nccl') #inherits params from default process group

    #SETUPS
    setrandom(20214229)
    filename = "./"+args.experiment+"/"+args.name
    testdump = filename+"TEST_ACC.txt"
    traindump = filename+"TRAIN_ACC.txt"
    checkpointdump = args.chkpt_dump
    ext = ".csv"

    accSamplePeriod = 5
    
    model = torchvision.models.vgg16(weights=None).cuda(gpu)

    #HYPERPARAMETERS
    batch_size = 512//args.gpus # global batch size of 512
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.05, epochs=201, steps_per_epoch=98)

    if args.checkpoint is not None:
        print("loading a model")
        load_checkpoint(gpu, model, optimizer, scheduler, args.checkpoint)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu, process_group=customGroup)

    #DATASETS                           
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpus, rank=gpu)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size = batch_size,
                                              shuffle=False,
                                              pin_memory=True)

    eval_set = torch.utils.data.Subset(train_dataset, [random.randint(0,len(train_dataset)-1) for i in range(len(test_dataset))])
    eval_loader = torch.utils.data.DataLoader(dataset=eval_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True)

    total_step = len(train_loader)

    if gpu==0:
        with open(filename+ext, "w+") as f:
            print("Loss", file=f)
        open(testdump, "w+").close()
        open(traindump, "w+").close()

    model.train()

    idx = 0

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):

            idx += 1
            images = images.cuda(gpu, non_blocking=True)

            labels = labels.cuda(gpu, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            if gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
                with open(filename+ext, "a+") as f:
                    print("{}".format(loss.item()), file=f)

        if gpu==0 and epoch%accSamplePeriod==0:
            evaluation(model, gpu, epoch+1, eval_loader, traindump, "Train set", args, scheduler)
            evaluation(model, gpu, epoch+1, test_loader, testdump, "Test set", args, scheduler)
        
        if gpu==0 and epoch==args.epochs-1:
            save_checkpoint(model, optimizer, scheduler, epoch+1, checkpointdump, args.name)
        


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

    # if evalname=="Test set":
    #     scheduler.step(accuracy)

    model.train()
    with open(filename, "a+") as f:
        print("{}%".format(accuracy), file=f)

if __name__ == '__main__':
    main()
