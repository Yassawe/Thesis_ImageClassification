import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.distributed as dist
import random
import numpy as np
import pandas as pd


train_transform = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(p=.40),
    T.RandomRotation(30),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def setrandom(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')

    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--lr', default = 1e-3, type=float)
    

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8008'

    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                               train=True,
                                               transform=train_transform,
                                               download=True)
    
    mp.spawn(train, nprocs=args.gpus, args=(train_dataset, args,))



def train(gpu, train_dataset, args):

    setrandom(20214229)
 
    dist.init_process_group(backend='nccl', world_size=args.gpus, rank=gpu)
    
    model = torchvision.models.resnet152(pretrained=False)

    torch.cuda.set_device(gpu)

    model.cuda(gpu)

    batch_size = 128
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
                                               
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.gpus,
                                                                    rank=gpu)
                                                                    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    total_step = len(train_loader)

    filename = "./grads/GPU" + str(gpu) + "_"

    target_iters = [10,100,1000] #here global iterations, regardless of epoch num and etc. i.e. iter = epoch*numsteps+step
    
    model.train()

    idx = 0

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):

            idx+=1

            images = images.cuda(gpu, non_blocking=True)

            labels = labels.cuda(gpu, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            if gpu==0:
                if idx in target_iters:
                    print("extracting grads at iteration {}".format(idx))
                    g = torch.Tensor().cuda(gpu)
                    for params in model.parameters():
                        t = params.grad
                        t = torch.flatten(t)
                        g = torch.cat((g,t))
                    
                    g_np = g.cpu().numpy()
                    np.savetxt(filename+str(idx), g_np)
                    print("done extracting grads")
            
            optimizer.step()
            
            if gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
        scheduler.step()



if __name__ == '__main__':
    main()