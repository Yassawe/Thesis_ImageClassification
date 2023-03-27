import os, torch, argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

from torch.distributions import normal


def saveData(data, rank, state = "beforeAllReduce"):
    path = "./test/"
    filename = "GPU" + str(rank) + "_" + state
    np.savetxt(path+filename+".txt", data.numpy())

def generateTensor(M, mean, std):
    return torch.randn(M) * std + mean

def runProcess(rank, args):
    torch.cuda.set_device(rank)

    data = generateTensor(args.M, args.mean, args.std)

    # DATA BEFORE ALLREDUCE
    saveData(data, rank, "beforeAllReduce")

    ###
    data.cuda()
    dist.all_reduce(data)
    ###

    # DATA AFTER ALLREDUCE
    data.cpu()
    saveData(data, rank, "afterAllReduce")


def init_process(rank, function, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8008" 
    dist.init_process_group("nccl", rank=rank, world_size=args.gpus)
    function(rank, args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpus", default=4, type=int)
    parser.add_argument("--M", default=10000, type=int)
    parser.add_argument("--mean", default = 0.00000028992896587, type=float)
    parser.add_argument("--std", default = 0.000511206 , type=float)

    args = parser.parse_args()

    mp.spawn(init_process, nprocs = args.gpus, args=(runProcess, args)) 
    