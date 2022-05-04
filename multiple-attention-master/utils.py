import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class dist_average:
    def __init__(self, local_rank,dist=True):
        self.world_size = dist.get_world_size() if dist else 0
        self.rank = dist.get_rank() if dist else 0
        self.local_rank = local_rank
        self.acc = torch.zeros(1).to(local_rank)
        self.count = 0
        self.dist = dist

    def step(self, input_):
        self.count += 1
        if type(input_) != torch.Tensor:
            input_ = torch.tensor(input_).to(self.local_rank, dtype=torch.float)
        else:
            input_ = input_.detach()
        self.acc += input_

    def get(self):
        if self.dist:
            dist.all_reduce(self.acc, op=dist.ReduceOp.SUM)
        self.acc /= self.world_size
        return self.acc.item() / self.count


def ACC(x, y):
    with torch.no_grad():
        x = torch.nn.functional.softmax(x)
        a = torch.max(x, dim=1)[1]
        precision = torch.sum((a == 1) & (a == y)).float() / torch.sum(a == 1).float()
        recall = torch.sum((a == 1) & (a == y)).float() / torch.sum(y == 1).float()
        acc = torch.sum(a == y).float() / x.shape[0]
        print(acc,precision,recall)
    # print(y,a,acc)
    return acc
    # return acc, precision, recall


def cont_grad(x, rate=1):
    return rate * x + (1 - rate) * x.detach()
