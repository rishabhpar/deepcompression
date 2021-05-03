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


def channel_fraction_pruning(model, fraction):
    model.conv1 = prune.ln_structured(model.conv1, name="weight", amount=fraction, n=1, dim=0)

    for i in range(0, 13):
        model.layers[i].conv2 = prune.ln_structured(model.layers[i].conv2, name="weight", amount=fraction, n=1, dim=0)


def test(frac, e):
    os.system(f'python tester.py --fraction {frac} --epoch {e}')


def train(model, num_epochs, device, batch_size=128, random_seed=1, compute_test_acc=False, trans=transforms.Compose([])):
    torch.manual_seed(random_seed)

    # CIFAR10 Dataset (Images and Labels)
    train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
        trans,
        #transforms.ToTensor(),
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

    prev_test_acc = 0
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
            test_loss_hist.append(test_loss / len(test_loader))
            test_acc_hist.append(100. * test_correct / test_total)
            print(f'Test Accuracy %: {test_acc_hist[-1]}')
            if prev_test_acc >= test_acc_hist[-1]:
                print("Model is overfitting!, stopping training")
                break

        print(f'Train Accuracy %: {train_acc_hist[-1]}')
        print(f'Training Time: {training_time}(s)')

    retData['train_acc_hist'] = train_acc_hist
    retData['test_acc_hist'] = test_acc_hist
    return retData


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

################# Functions copied from PyTorch tutorial ################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end='')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5

    return top1, top5


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end='')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return


from torch.quantization import QuantStub, DeQuantStub


class QATWrapper(nn.Module):

    def __init__(self, model):
        super(QATWrapper, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.model = model

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


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
        model_name = 'model_frac_0.880.pt'
        model = torch.load(f'm3/structural_pruned_noisy1/{model_name}', map_location=device)
        model.to(device)

    ##################################
    #  Iterative Structural Pruning  #
    ##################################
    RUN_ITER_STRUCT_PRUN = True

    STRUCT_PRUN_SAVE_DIR = "m3/structural_pruned_noisy2"
    if RUN_ITER_STRUCT_PRUN:
        Path(STRUCT_PRUN_SAVE_DIR).mkdir(exist_ok=True, parents=True)

        max_prune_fraction = 0.95
        prune_each_step = 0.01
        epochs_after_each_prune = 100
        currently_pruned_frac = 0.88

        prune_frac_hist = []
        test_acc_hist = []
        train_acc_hist = []
        num_epochs_trained_hist = []

        train_transformations = transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                                                    transforms.RandomAffine(degrees=2, translate=(0.025, 0.025), scale=(0.98, 1.02), shear=1),
                                                    transforms.ToTensor(),
                                                  ])

        while currently_pruned_frac < max_prune_fraction:

            # Calculate Pruning Amount
            prune_frac = 1 - ((1 - currently_pruned_frac) - prune_each_step) / (1 - currently_pruned_frac)

            # Prune model
            channel_fraction_pruning(model, prune_frac)
            summary(model, input_size=(1, 3, 32, 32), verbose=0)
            model = remove_channel(model)
            currently_pruned_frac += prune_each_step
            print(f"--- Model Pruned Fraction: {currently_pruned_frac} ---")

            # Retrain model after pruning
            if epochs_after_each_prune > 0:
                model.to(device)
                train_info = train(model=model, num_epochs=epochs_after_each_prune, device=device, batch_size=128,
                                   random_seed=1, compute_test_acc=True, trans=train_transformations)
                num_epochs_trained = len(train_info['train_acc_hist'])
                train_acc, test_acc = train_info['train_acc_hist'][-1], train_info['test_acc_hist'][-1]
                test_acc_hist.append(test_acc)
                train_acc_hist.append(train_acc)
                num_epochs_trained_hist.append(num_epochs_trained)
            prune_frac_hist.append(currently_pruned_frac)

            # Save model after each prune + retrain iteration
            torch.save(model, f'{STRUCT_PRUN_SAVE_DIR}/model_frac_{currently_pruned_frac:0.3f}.pt')
            prune_info_df = pd.DataFrame(columns=('Prune Fraction', 'Test Accuracy', 'Train Accuracy', 'epochs_trained'),
                                         data=list(zip(prune_frac_hist, test_acc_hist, train_acc_hist, num_epochs_trained_hist)))
            prune_info_df.to_csv(f'{STRUCT_PRUN_SAVE_DIR}/prune_info_log.csv')

    ###################################
    #          Quantization           #
    ###################################
    RUN_QUANTIZATION = False

    ONNX_SAVE_DIR = "m3/qat_static_quantized"
    Path(ONNX_SAVE_DIR).mkdir(exist_ok=True, parents=True)

    if RUN_QUANTIZATION:
        assert Path(STRUCT_PRUN_SAVE_DIR).exists()

        import warnings

        warnings.filterwarnings(
            action='ignore',
            category=DeprecationWarning,
            module=r'.*'
        )
        warnings.filterwarnings(
            action='default',
            module=r'torch.quantization'
        )

        qat_model = QATWrapper(model)
        # qat_model.fuse_model()

        optimizer = torch.optim.Adam(qat_model.parameters())
        criterion = nn.CrossEntropyLoss()

        train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]), download=True)

        test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]))

        # Dataset Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)


        qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        torch.quantization.prepare_qat(qat_model, inplace=True)
        # print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n', qat_model.features[1].conv)

        num_train_batches = 20
        num_eval_batches = 200
        eval_batch_size = 128

        # QAT takes time and one needs to train over a few epochs.
        # Train and check accuracy after each epoch
        qat_acc_hist = []
        for nepoch in range(1):
            train_one_epoch(qat_model, criterion, optimizer, train_loader, torch.device('cpu'), num_train_batches)
            if nepoch > 3:
                # Freeze quantizer parameters
                qat_model.apply(torch.quantization.disable_observer)
            if nepoch > 2:
                # Freeze batch norm mean and variance estimates
                qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            # Check the accuracy after each epoch
            quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
            quantized_model.eval()
            top1, top5 = evaluate(quantized_model, criterion, test_loader, neval_batches=num_eval_batches)
            print('Epoch %d :Evaluation accuracy on %d images, %2.2f' % (
                nepoch, num_eval_batches * eval_batch_size, top1.avg))
            qat_acc_hist.append(top1.avg)

        quantized_model = torch.quantization.convert(qat_model.model.eval(), inplace=False)
        quantized_model.eval()
        torch.onnx.export(quantized_model.cpu(), torch.randn(1, 3, 32, 32),
                          f'{ONNX_SAVE_DIR}/{model_name}_qat_static.onnx', export_params=True, opset_version=10)
