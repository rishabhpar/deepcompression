import pandas as pd
import torch
from torchinfo import summary
from mobilenet_rm_filt_pt import MobileNetv1, remove_channel
import torch.nn.utils.prune as prune
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from pathlib import Path
import time
import glob

train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

# Dataset Loader (Input Pipeline)
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

STRUCT_PRUN_SAVE_DIR = "m3/structural_pruned_retrain_100"
assert Path(STRUCT_PRUN_SAVE_DIR).exists()

train_acc_hist = []
test_acc_hist = []
frac_hist = []


for model_path in glob.glob(f"{STRUCT_PRUN_SAVE_DIR}/*.pt"):

    frac = float(model_path.replace(STRUCT_PRUN_SAVE_DIR, '').split('_')[2].replace('.pt', ''))
    print(f"Testing model with fraction: {frac}")

    # ########### Load Trained Model ############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # model = MobileNetv1().to(device)
    # state_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(state_dict)
    model = torch.load(model_path, map_location=device)
    model.to(device)

    # ########## Train Accuracy #############
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Put the images and labels on the GPU
            images = images.to(device)
            labels = labels.to(device)

            # Perform the actual inference
            outputs = model(images)
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
    train_acc = (100. * train_correct / train_total)
    print(f'Train Accuracy %: {train_acc}')

    ##### Test Accuracy ######
    # Testing phase loop
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Put the images and labels on the GPU
            images = images.to(device)
            labels = labels.to(device)

            # Perform the actual inference
            outputs = model(images)
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_acc = (100. * test_correct / test_total)
    print(f'Test Accuracy %: {test_acc}')
    print("---------------------------------")
    frac_hist.append(frac)
    test_acc_hist.append(test_acc)
    train_acc_hist.append(train_acc)

prune_info_df = pd.DataFrame(columns=('Prune Fraction', 'Test Accuracy', 'Train Accuracy'), data=list(zip(frac_hist, test_acc_hist, train_acc_hist)))
prune_info_df.to_csv(f'{STRUCT_PRUN_SAVE_DIR}/prune_info_log.csv')