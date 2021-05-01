import pandas as pd
import torch
from torchinfo import summary
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from pathlib import Path
import time
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelPruning
import torch.nn.functional as F

"""
    MobileNet-v1 model written in PyTorch
    CIFAR10 test dataset accuracy: 77.15%
"""
class Block(pl.LightningModule):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetv1(pl.LightningModule):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNetv1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        self.num_classes = num_classes
        self.mask_dict=None
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters())
        return optim

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        logits = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

class CIFAR_DataModule(pl.LightningDataModule):

    def prepare_data(self, *args, **kwargs):
        dsets.CIFAR10(root='data', train=True, download=True)
        dsets.CIFAR10(root='data', train=False, download=True)

    def train_dataloader(self):
        train_ds = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]), download=True)
        train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=128, shuffle=True)
        return train_dl

    def val_dataloader(self):
        test_ds = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]))
        test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=128, shuffle=False)
        return test_dl

    def test_dataloader(self):
        test_ds = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]))
        test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=128, shuffle=False)
        return test_dl

if __name__ == '__main__':
    # ########### Load Trained Model ############
    LOAD_CUSTOM_MODEL = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not LOAD_CUSTOM_MODEL:
        model = MobileNetv1().to(device)
        state_dict = torch.load('mbnv1_pt.pt', map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        model_name = 'model_frac_0.65.pt'
        model = torch.load(f'm3/structural_pruned_retrain_100/{model_name}', map_location=device)
        model.to(device)

    ##################################
    #  Iterative Structural Pruning  #
    ##################################
    RUN_ITER_STRUCT_PRUN = True

    STRUCT_PRUN_SAVE_DIR = "m3/structural_pruned_noisy0"
    if RUN_ITER_STRUCT_PRUN:
        Path(STRUCT_PRUN_SAVE_DIR).mkdir(exist_ok=True, parents=True)

        max_prune_fraction = 0.7
        prune_each_step = 0.01
        epochs_after_each_prune = 100
        currently_pruned_frac = 0.65

        prune_frac_hist = []
        test_acc_hist = []
        train_acc_hist = []
        num_epochs_trained_hist = []


        def compute_prune_frac(epoch):
            currently_pruned_frac = prune_each_step * epoch
            prune_frac = 1 - ((1 - currently_pruned_frac) - prune_each_step) / (1 - currently_pruned_frac)
            if prune_frac > max_prune_fraction:
                return 0
            else:
                return prune_frac


        prune_list = [(model.conv1, 'weight')]
        for i in range(0, 13):
            prune_list.append((model.layers[i].conv2, 'weight'))

        trainer = Trainer(callbacks=[ModelPruning("ln_structured", amount=compute_prune_frac, parameters_to_prune=prune_list, pruning_dim=0, pruning_norm=1, use_global_unstructured=False)], max_epochs=2)
        model = MobileNetv1()
        trainer.fit(model, CIFAR_DataModule())

        torch.save(model, 'model.save')



