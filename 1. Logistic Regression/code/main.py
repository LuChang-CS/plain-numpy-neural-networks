import os

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

from data.utils import train_test_split
from data.dataloader import DataLoader
from data.dataset import Iris
from model.logistic import Logistic, logistic_sgd
from model.metrics import accuracy, precision, recall, f1_score


def preprocess(y):
    y_new = (y[:, 0] == 1).astype(np.float32)
    return y_new


class LogisticTorch(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.linear = nn.Linear(dim_in, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.linear(x)
        output = self.sigmoid(output)
        output = output.squeeze()
        return output


def torch_model_step(optimizer, model, input_, target, loss_fn):
    optimizer.zero_grad()
    y_hat = model(input_)
    loss = loss_fn(y_hat, target)
    loss.backward()
    optimizer.step()

    loss = loss.item()
    y_hat = y_hat.detach().numpy()
    target = target.numpy()
    return loss, y_hat, target


def numpy_model_step(optimizer, model, input_, target):
    y_hat = model(input_)
    loss = model.loss(y_hat, target)
    grad = model.backward(input_, y_hat, target)
    optimizer(model, grad, learning_rate)
    return loss, y_hat


def train_logistic(model, train_loader, epochs=100, learning_rate=0.01, use_torch=False):
    if use_torch:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCELoss()
    steps = len(train_loader)
    losses, accuracies, precisions, recalls, f1s = [], [], [], [], []
    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        total_loss = 0.0
        total_num = 0
        y_hat, y = [], []
        for step, (x_i, y_i) in enumerate(train_loader):
            print('\r    Step %d / %d' % (step + 1, steps), end='')
            if use_torch:
                loss_i, y_hat_i, y_i = torch_model_step(optimizer, model, x_i, y_i, loss_fn)
            else:
                loss_i, y_hat_i = numpy_model_step(logistic_sgd, logistic_model, x_i, y_i)
            total_loss += loss_i
            total_num += len(x_i)
            y_hat.append(y_hat_i)
            y.append(y_i)

            print('\r    Step %d / %d, loss: %.4f' % (step + 1, steps, total_loss / total_num), end='')

        pred = (np.vstack(y_hat) > 0.5).astype(np.int32)
        y = np.vstack(y)

        loss = total_loss / train_loader.size()
        acc, prec, rec, f1 = get_metrics_result(pred, y)
        losses.append(loss)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        print('\r    Step %d / %d, loss: %.4f -- accuracy: %.4f -- precision: %.4f -- recall: %.4f -- f1: %.4f'
                  % (steps, steps, loss, acc, prec, rec, f1))
    history = {
        'loss': losses,
        'accuracy': accuracies,
        'precision': precisions,
        'recall': recalls,
        'f1': f1s
    }
    return history


def evaluate_logistic(model, test_loader, use_torch=False):
    if use_torch:
        model.eval()
        loss_fn = nn.BCELoss()
    steps = len(test_loader)
    y_hat, y = [], []
    total_loss = 0.0
    for step, (x_i, y_i) in enumerate(test_loader):
        print('\r    Step %d / %d' % (step + 1, steps), end='')

        y_hat_i = model(x=x_i)
        if use_torch:
            loss_i = loss_fn(y_hat_i, y_i)
            loss_i = loss_i.item()
            y_hat_i = y_hat_i.detach().numpy()
            y_i = y_i.numpy()
        else:
            loss_i = model.loss(y_hat_i, y_i)

        total_loss += loss_i
        y_hat.append(y_hat_i)
        y.append(y_i)
    print('\r', end='')
    pred = (np.vstack(y_hat) > 0.5).astype(np.int32)
    y = np.vstack(y)

    loss = total_loss / test_loader.size()
    acc, prec, rec, f1 = get_metrics_result(pred, y)
    return loss, acc, prec, rec, f1


def get_metrics_result(y_hat, y):
    acc, prec, rec, f1 = accuracy(y_hat, y), precision(y_hat, y), recall(y_hat, y), f1_score(y_hat, y)
    return acc, prec, rec, f1


if __name__ == '__main__':
    seed = 6666
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = os.path.join('..', '..', 'common', 'dataset')
    iris_dataset = Iris(dataset_path)
    x, y = iris_dataset.x, iris_dataset.y
    y = preprocess(y)
    dim_in = x.shape[1]
    epochs = 25
    learning_rate = 0.01

    train_data, test_data, _ = train_test_split(x, y, shuffle=True, train_num=100, test_num=50, validation=None)
    train_loader = DataLoader(train_data, batch_size=8)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=8)

    print('Training with numpy logistic model ...')
    logistic_model = Logistic(dim_in)
    history = train_logistic(logistic_model, train_loader, epochs, learning_rate)
    loss, acc, prec, rec, f1 = evaluate_logistic(logistic_model, test_loader)
    print('Test set, loss: %.4f -- accuracy: %.4f -- precision: %.4f -- recall: %.4f -- f1: %.4f'
              % (loss, acc, prec, rec, f1))

    print('Training with pytorch logistic model ...')
    train_loader.use_torch()
    test_loader.use_torch()
    logistic_model_torch = LogisticTorch(dim_in)
    history_torch = train_logistic(logistic_model_torch, train_loader, epochs, learning_rate, use_torch=True)
    loss, acc, prec, rec, f1 = evaluate_logistic(logistic_model_torch, test_loader, use_torch=True)
    print('Test set, loss: %.4f -- accuracy: %.4f -- precision: %.4f -- recall: %.4f -- f1: %.4f'
              % (loss, acc, prec, rec, f1))

    idx = np.arange(1, epochs + 1)
    fig = plt.figure(figsize=(20, 4))
    # plot loss
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    for i, m in enumerate(metrics):
        ax = plt.subplot(1, len(metrics), i + 1)
        ax.plot(idx, history[m], marker='o', markersize=4)
        ax.plot(idx, history_torch[m], marker='v', markersize=4)
        ax.grid(b=True, color='grey', linewidth=0.5, alpha=0.6)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(m)
        # ax.legend()
    plt.figlegend(('Numpy logistic model', 'PyTorch logistic model'), ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plt.show()
