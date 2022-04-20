import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from tqdm import tqdm

from common.datasets import TransformingTensorDataset


def curriculum_score(model, images, target):
    output = model(images)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    loss = criterion(output, target)
    return loss


def curriculum_pace(epoch_num, total_epochs, dataset_size, family="log"):
    N = dataset_size
    t = epoch_num
    T = total_epochs
    # Params from Fig. 19, Wu et al. 2021
    if family == "log":
        a = 0.4
        b = 0.2
        if t < a * T:
            g = b * N + (1 - b) * N * (1 + 0.1 * np.log(t / (a * T) + np.exp(-10)))
        else:
            g = N
    elif family == "exp":
        a = 0.01
        b = 0.4
        if t < a * T:
            g = b * N + (1 - b) * N * (np.exp(t / (a * T) * 10 - 1) / (np.exp(10) - 1))
        else:
            g = N
    elif family == "step":
        a = 0.1
        b = 0.8
        if t < a * T:
            g = b * N
        else:
            g = N
    elif family == "linear":
        a = 0.01
        b = 0.4
        if t < a * T:
            g = b * N + (t / (a * T)) * (1 - b) * N
        else:
            g = N

    return min(int(g), N)


def sort_dataset(X_tr, y_tr):
    pretrained_model = models.resnet18(pretrained=True)
    pretrained_model.eval()
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = TransformingTensorDataset(X_tr, y_tr, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024,
                                              shuffle=False, num_workers=1)
    scores = []
    with torch.no_grad():
        for x, y in tqdm(trainloader, desc="Run scoring"):
            _scores = curriculum_score(pretrained_model, x, y)
            scores.extend(_scores)
    scores = [float(s) for s in scores]
    X_tr_sorted = [x for _, x in sorted(zip(scores, X_tr), key=lambda pair: pair[0])]
    y_tr_sorted = [y for _, y in sorted(zip(scores, y_tr), key=lambda pair: pair[0])]
    return X_tr_sorted, y_tr_sorted

if __name__ == '__main__':
    pretrained_model = models.resnet18(pretrained=True)
    pretrained_model.eval()
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    total_epochs, dataset_size = 80, 50000
    for epoch_num in range(80):
        print(f"epoch {epoch_num}: {curriculum_pace(epoch_num, total_epochs, dataset_size, family='log')}")
