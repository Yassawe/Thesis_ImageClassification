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
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])


def setrandom(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(ddp_model, optimizer, epoch, folder, name):
    path = folder+name + ".pt"

    state = {
            'epoch':epoch,
            'model': ddp_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
    
    torch.save(state, path)

def load_checkpoint(rank, model, optimizer, path):

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint['model']) 
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']

def main():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    

    parser.add_argument('--lr', default = 1e-3, type=float)

    parser.add_argument('--name', default="baseline", type=str)
    parser.add_argument('--experiment', default="baseline", type=str)
    parser.add_argument('--recordCheckpoints', default=0, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)

    


    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2023'
    os.environ['NCCL_ALGO'] = 'Ring'
    #os.environ['NCCL_DEBUG'] = "INFO"

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
    ext = ".csv"

    testdump = filename+"TEST_ACC.txt"
    traindump = filename+"TRAIN_ACC.txt"
    checkpointdump = "./checkpoints/"

    accSamplePeriod = 5
    checkpointPeriod = 50


    #MODEL AND HYPERPARAMETERS
    model = torchvision.models.resnet50(weights=None).cuda(gpu)

    batch_size = 512//args.gpus
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    total_epochs = args.epochs
    lastCheckpointEpoch=0

    if args.checkpoint_path is not None:
        lastCheckpointEpoch = load_checkpoint(gpu, model, optimizer, args.checkpoint_path)

    total_epochs-=lastCheckpointEpoch
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    
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
                                                    pin_memory=True
                                                    )

    total_step = len(train_loader)

    if gpu==0:
        with open(filename+ext, "w+") as f:
            print("Loss", file=f)
        open(testdump, "w+").close()
        open(traindump, "w+").close()

    idx = 0
    model.train()

    for epoch in range(total_epochs):
        for i, (images, labels) in enumerate(train_loader):
            idx+=1
            
            images = images.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            # if idx == target_iter:
            #     profiler.start()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            # if idx == target_iter:
            #     dist.barrier()
            #     profiler.stop()
            #     break
            
            if gpu == 0:
                print('Epoch [{}/{}]. Step [{}/{}], Loss: {:.4f}'.format(epoch+lastCheckpointEpoch, args.epochs, i + 1, total_step, loss.item()))
                with open(filename+ext, "a+") as f:
                    print("{}".format(loss.item()), file=f)
        
        

        scheduler.step()   
        
        if gpu==0 and (epoch)%accSamplePeriod==0:
            evaluation(model, gpu, epoch+lastCheckpointEpoch+1, eval_loader, traindump, "TRAIN SET", args)
            evaluation(model, gpu, epoch+lastCheckpointEpoch+1, test_loader, testdump, "TEST SET", args)

        if gpu==0 and (epoch)%checkpointPeriod==0 and args.recordCheckpoints==1:
            save_checkpoint(model, optimizer, epoch+lastCheckpointEpoch+1, checkpointdump, args.name+"_"+str(epoch+lastCheckpointEpoch))


def evaluation(model, gpu, epoch, dataloader, file, evalname, args):
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
    model.train()
    with open(file, "a+") as f:
        print("{}%".format(accuracy), file=f)  

if __name__ == '__main__':
    main()
