import os
import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
import dgl

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

class MNISTNet(LightningModule):
    def __init__(self):
        super().__init__()
        
        self.l1 = torch.nn.Linear(28 * 28, 128)
        self.l2 = torch.nn.Linear(128, 256)
        self.l3 = torch.nn.Linear(256, 10)
        
    def forward(self, x):
        batch_size, channels, width, height = x.size()
        
        x = x.view(batch_size, -1) # flatten to vector
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.log_softmax(x, dim=1)
        
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
        
mnist_train = MNIST(os.getcwd(), train=True, download=True)
mnist_train = DataLoader(mnist_train, batch_size=64)
        
net = MNISTNet()
trainer = pl.Trainer()
trainer.fit(net, mnist_train)