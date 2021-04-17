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


def channel_fraction_pruning(model, fraction):
    model.conv1 = prune.ln_structured(model.conv1, name="weight", amount=fraction, n=1, dim=0)

    for i in range (0,13):
        model.layers[i].conv2 = prune.ln_structured(model.layers[i].conv2, name="weight", amount=fraction, n=1, dim=0)

def test(frac, e):
    os.system(f'python tester.py --fraction {frac} --epoch {e}')
    

def train(model, num_epochs, device, batch_size=128, random_seed=1, compute_test_acc=False):
    torch.manual_seed(random_seed)

    # CIFAR10 Dataset (Images and Labels)
    train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]), download=True)

    test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]))

    # Dataset Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Define your loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
    optimizer = torch.optim.Adam(model.parameters())

    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []

    retData = {}

    training_time = 0

    for epoch in range(num_epochs):
        # Training phase loop
        start_time = time.time()
        train_correct = 0
        train_total = 0
        train_loss = 0
        # Sets the model in training mode.
        model = model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Put the images and labels on the GPU
            images = images.to(device)
            labels = labels.to(device)

            # Sets the gradients to zero
            optimizer.zero_grad()
            # The actual inference
            outputs = model(images)
            # Compute the loss between the predictions (outputs) and the ground-truth labels
            loss = criterion(outputs, labels)
            # Do backpropagation to update the parameters of your model
            loss.backward()
            # Performs a single optimization step (parameter update)
            optimizer.step()
            train_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            # Print every 100 steps the following information
            if (batch_idx + 1) % 100 == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                                 len(train_dataset) // batch_size,
                                                                                 train_loss / (batch_idx + 1),
                                                                                 100. * train_correct / train_total))

        train_loss_hist.append(train_loss / len(train_loader))
        train_acc_hist.append(100. * train_correct / train_total)
        training_time += time.time() - start_time

        if compute_test_acc:
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
                    # Compute the loss
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    # The outputs are one-hot labels, we need to find the actual predicted
                    # labels which have the highest output confidence
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            print(
                'Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1), 100. * test_correct / test_total))
            test_loss_hist.append(test_loss / len(test_loader))
            test_acc_hist.append(100. * test_correct / test_total)
            print(f'Test Accuracy %: {test_acc_hist[-1]}')

        print(f'Train Accuracy %: {train_acc_hist[-1]}')
        print(f'Training Time: {training_time}(s)')

    print(str(train_acc_hist))
    print(str(test_acc_hist))
    retData['train_acc_hist'] = train_acc_hist
    retData['test_acc_hist'] = test_acc_hist
    return retData


if __name__ == '__main__':

    ONNX_SAVE_DIR = "onnx_models"
    PT_SAVE_DIR = "pt_models"
    Path(ONNX_SAVE_DIR).mkdir(exist_ok=True, parents=True)
    Path(PT_SAVE_DIR).mkdir(exist_ok=True, parents=True)

    # ########## Load Trained Model ##########
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}")
    batch_size = 1

    fractions_to_prune = [0.05, 0.25, 0.5, 0.75, 0.9]
    epochs_to_train = [0, 3, 5]

    for frac in fractions_to_prune:
        print(f"Creating model with Pruning Fraction: {frac}")
        for epochs in epochs_to_train:

            print(f"Epoch: {epochs}")

            model = MobileNetv1().to(device)
            #state_dict = torch.load('mbnv1_pt.pt', map_location=torch.device('cpu'))
            state_dict = torch.load('mbnv1_pt.pt', map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)

            channel_fraction_pruning(model, frac)
            summary(model, input_size=(batch_size, 3, 32, 32), verbose=0)
            cleaned_model = remove_channel(model)

            # Retrain for epochs epochs
            if epochs > 0:
                cleaned_model.to(device)
                train_info = train(model=cleaned_model, num_epochs=epochs, device=device, batch_size=128, random_seed=1, compute_test_acc=False)
                print(f"M1: Training Info: {train_info}")

            # Export to ONNX
            torch.save(cleaned_model, f'{PT_SAVE_DIR}/model_epochs_{epochs}_frac_{frac}.pt')
            cleaned_model.cpu() # Send cleaned model to CPU for ONNX export
            torch.onnx.export(cleaned_model, torch.randn(1, 3, 32, 32), f'{ONNX_SAVE_DIR}/model_epochs_{epochs}_frac_{frac}.onnx', export_params=True, opset_version=10)
            #test(frac, e)
