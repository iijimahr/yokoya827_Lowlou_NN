import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PINN_NN(nn.Module):
    
    def __init__(self, x0, y0, z0, width = 128, depth = 5, lb = 0, ub = 63):
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.lb = lb
        self.ub = ub

        #ネットの作成
        in_dim = 3  #x, y, z
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 3))  # Bx, By, Bz
        self.net = nn.Sequential(*layers)

    def forward(self, x, y, z):

        lb = self.lb

        x = 2.0*(x - lb)/(self.x0 - lb) - 1.0
        y = 2.0*(y - lb)/(self.y0 - lb) - 1.0
        z = 2.0*(z - lb)/(self.z0 - lb) - 1.0

        inp = torch.cat([x, y, z], dim=1)

        return self.net(inp)

