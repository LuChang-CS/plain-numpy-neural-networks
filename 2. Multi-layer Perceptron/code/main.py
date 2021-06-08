import os
import _pickle as pickle

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

from data.utils import train_test_split
from data.dataloader import DataLoader
from data.dataset import MNIST
from model.mlp import MLP, MLP_Adam
from model.metrics import accuracy


class MLPTorch(nn.Module):
    def __init__(self, dim_in, hidden_units, dim_out):
        super().__init__()
        dims = [dim_in] + hidden_units + [dim_out]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                         for i in range(len(dims) - 1)])
        self.relu = nn.ReLU()

    def forward(self, x):
        output = x
        for linear in self.linears[:-1]:
            output = self.relu(linear(output))
        output = self.linears[-1](output)
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
    grads = model.backward(y_hat, target)
    optimizer.step(grads)
    return loss, y_hat


def train_mlp(model, train_loader, valid_loader, epochs=100, learning_rate=0.01, use_torch=False):
    if use_torch:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
    else:
        optimizer = MLP_Adam(model, learning_rate)
    steps = len(train_loader)
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
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
                loss_i, y_hat_i = numpy_model_step(optimizer, model, x_i, y_i)
            total_loss += loss_i
            total_num += len(x_i)
            y_hat.append(y_hat_i)
            y.append(y_i)

            print('\r    Step %d / %d, loss: %.4f' % (step + 1, steps, total_loss / total_num), end='')

        pred = (np.argmax(np.vstack(y_hat), axis=-1)).astype(np.int32)
        y = np.concatenate(y)

        loss = total_loss / train_loader.size()
        acc = accuracy(pred, y)
        train_losses.append(loss)
        train_accuracies.append(acc)
        print('\r    Step %d / %d, loss: %.4f -- accuracy: %.4f' % (steps, steps, loss, acc))
        valid_loss, valid_acc = evaluate_mlp(model, valid_loader, use_torch)
        print('    Validation set, loss: %.4f -- accuracy: %.4f' % (valid_loss, valid_acc))
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'valid_loss': valid_losses,
        'valid_accuracy': valid_accuracies
    }
    return history


def evaluate_mlp(model, test_loader, use_torch=False):
    if use_torch:
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
    steps = len(test_loader)
    y_hat, y = [], []
    total_loss = 0.0
    for step, (x_i, y_i) in enumerate(test_loader):
        print('\r    Step %d / %d' % (step + 1, steps), end='')

        y_hat_i = model(x_i)
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
    pred = (np.argmax(np.vstack(y_hat), axis=-1)).astype(np.int32)
    y = np.concatenate(y)

    loss = total_loss / test_loader.size()
    acc = accuracy(pred, y)
    return loss, acc


if __name__ == '__main__':
    seed = 6666
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = os.path.join('..', '..', 'common', 'dataset')
    mnist_dataset = MNIST(dataset_path)
    (train_x, train_y), (test_x, test_y) = mnist_dataset.train_data, mnist_dataset.test_data
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    train_x = train_x.reshape((train_x.shape[0], -1)).astype(np.float32)
    test_x = test_x.reshape((test_x.shape[0], -1)).astype(np.float32)
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    train_data, valid_data, _ = train_test_split(train_x, train_y, shuffle=True, train_num=50000, test_num=10000, validation=None)

    dim_in = train_x.shape[1]
    hidden_units = [512, 256]
    dim_out = 10
    epochs = 25
    learning_rate = 0.001

    train_loader = DataLoader(train_data, batch_size=100)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=100)
    test_loader = DataLoader((train_x, train_y), shuffle=False, batch_size=100)

    print('Training with numpy MLP model ...')
    mlp_model = MLP(dim_in, hidden_units, dim_out)
    history = train_mlp(mlp_model, train_loader, valid_loader, epochs, learning_rate)
    loss, acc = evaluate_mlp(mlp_model, test_loader)
    print('Test set, loss: %.4f -- accuracy: %.4f' % (loss, acc))

    print('Training with pytorch logistic model ...')
    train_loader.use_torch()
    valid_loader.use_torch()
    test_loader.use_torch()
    mlp_model_torch = MLPTorch(dim_in, hidden_units, dim_out)
    history_torch = train_mlp(mlp_model_torch, train_loader, valid_loader, epochs, learning_rate, use_torch=True)
    loss, acc = evaluate_mlp(mlp_model_torch, test_loader, use_torch=True)
    print('Test set, loss: %.4f -- accuracy: %.4f' % (loss, acc))

    idx = np.arange(1, epochs + 1)
    fig = plt.figure(figsize=(15, 4))
    # plot loss
    metrics = ['train_loss', 'train_accuracy', 'valid_loss', 'valid_accuracy']
    for i, m in enumerate(metrics):
        ax = plt.subplot(1, len(metrics), i + 1)
        ax.plot(idx, history[m], marker='o', markersize=4)
        ax.plot(idx, history_torch[m], marker='v', markersize=4)
        ax.grid(b=True, color='grey', linewidth=0.5, alpha=0.6)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(m)
        # ax.legend()
    plt.figlegend(('Numpy MLP model', 'PyTorch MLP model'), ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plt.savefig('performance.png')
