import argparse
import os
from abc import ABC

import numpy as np
import torch
from torch.nn import Conv2d, MaxPool2d, ReLU, Linear, Module
from torch.nn.functional import cross_entropy
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


# model
class CNN(Module, ABC):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = Conv2d(in_channels=1,
                             out_channels=8,
                             stride=1,
                             kernel_size=3,
                             padding=1)  # 1x32x32 -> 8x32x32
        self.pool_1 = MaxPool2d(stride=2,
                                kernel_size=2,
                                padding=0)  # 8x32x32 -> 8x16x16
        self.relu_1 = ReLU(inplace=False)
        self.conv_2 = Conv2d(in_channels=8,
                             out_channels=16,
                             stride=1,
                             kernel_size=3,
                             padding=1)  # 8x16x16 -> 16x16x16
        self.pool_2 = MaxPool2d(stride=2,
                                kernel_size=2,
                                padding=0)  # 16x16x16-> 16x8x8
        self.relu_2 = ReLU(inplace=False)
        self.fc = Linear(in_features=(16 * 8 * 8),
                         out_features=10)  # 16x8x8 -> 10

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.relu_2(x)
        x = x.view(-1, (16 * 8 * 8))  # [Batch, Channel, Height, Width] -> [Batch, (Channel * Height * Width)]
        x = self.fc(x)
        return x


# train
def train(model, device, loader, optimizer):
    batches = len(loader)
    samples = len(loader.dataset)
    model.train()
    processed_samples = 0
    for batch_index, batch in enumerate(iterable=loader):
        batch_no = batch_index + 1
        batch_data, batch_target = batch
        batch_size = len(batch_data)
        processed_samples += batch_size
        batch_data, batch_target = batch_data.to(device=device), batch_target.to(device=device)
        optimizer.zero_grad()
        if batch_data.dim() == 3:
            batch_data = batch_data.unsqueeze(dim=1)  # [batch, height, width] -> [batch, channel, height, width]
        batch_output = model(batch_data)
        batch_loss = cross_entropy(input=batch_output,
                                   target=batch_target)
        batch_loss.backward()
        batch_loss = batch_loss.item()
        optimizer.step()
        print('Batch: [{batch_no}/{batches}], '
              'Sample: [{processed_samples}/{samples}], '
              'Batch Loss: {batch_loss:.6f}'.format(batch_no=batch_no,
                                                    batches=batches,
                                                    processed_samples=processed_samples,
                                                    samples=samples,
                                                    batch_loss=batch_loss))


# test
def test(model, device, loader):
    batches = len(loader)
    samples = len(loader.dataset)
    model.eval()
    batch_losses = []
    batch_corrects = []
    with torch.no_grad():
        processed_samples = 0
        for batch_index, batch in enumerate(iterable=loader):
            batch_no = batch_index + 1
            batch_data, batch_target = batch
            batch_size = len(batch_data)
            processed_samples += batch_size
            batch_data, batch_target = batch_data.to(device=device), batch_target.to(device=device)
            if batch_data.dim() == 3:
                batch_data = batch_data.unsqueeze(dim=1)  # [batch, height, width] -> [batch, channel, height, width]
            batch_output = model(batch_data)
            batch_loss = cross_entropy(input=batch_output,
                                       target=batch_target).item()
            batch_losses.append(batch_loss)
            batch_prediction = batch_output.max(dim=1, keepdim=True)[1]
            batch_correct = batch_prediction.eq(batch_target.view_as(other=batch_prediction)).sum().item()
            batch_corrects.append(batch_correct)
            batch_accuracy = 100. * batch_correct / batch_size
            print('Batch: [{batch_no}/{batches}], '
                  'Sample: [{processed_samples}/{samples}], '
                  'Batch Loss: {batch_loss:.6f}, '
                  'Batch Correct: [{batch_correct}/{batch_size}], '
                  'Batch Accuracy: {batch_accuracy:.0f}%'.format(batch_no=batch_no,
                                                                 batches=batches,
                                                                 processed_samples=processed_samples,
                                                                 samples=samples,
                                                                 batch_size=batch_size,
                                                                 batch_correct=batch_correct,
                                                                 batch_accuracy=batch_accuracy,
                                                                 batch_loss=batch_loss))

    epoch_loss = sum(batch_losses) / samples
    epoch_correct = sum(batch_corrects)
    epoch_accuracy = 100.0 * (sum(batch_corrects) / samples)
    print('Epoch Loss: {epoch_loss:.4f}, '
          'Epoch Correct: [{epoch_correct}/{samples}], '
          'Epoch Accuracy: {epoch_accuracy:.0f}%'.format(epoch_loss=epoch_loss,
                                                         epoch_correct=epoch_correct,
                                                         epoch_accuracy=epoch_accuracy,
                                                         samples=samples))


# main
def main():
    # paths
    paths = dict()
    paths['project'] = '.'
    paths['datasets'] = os.path.join(paths['project'], 'datasets')
    paths['epochs'] = os.path.join(paths['project'], 'epochs')

    # make directories if not exist
    for key in paths.keys():
        value = paths[key]
        if not os.path.exists(value):
            os.makedirs(value)

    # settings
    argument_parser = argparse.ArgumentParser()

    # argument parser
    argument_parser.add_argument('--train-batch-size', type=int, default=2000, help='batch size for training')
    argument_parser.add_argument('--test-batch-size', type=int, default=2000, help='batch size for testing')
    argument_parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    argument_parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    argument_parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate')
    argument_parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    argument_parser.add_argument('--cuda', type=bool, default=False, help='enable CUDA training')
    argument_parser.add_argument('--seed', type=int, default=1, help='random seed')

    # arguments
    arguments = argument_parser.parse_args()

    # cuda
    use_cuda = arguments.cuda and torch.cuda.is_available()

    # device
    device = torch.device('cuda' if use_cuda else 'cpu')

    # seed
    torch.manual_seed(seed=arguments.seed)
    np.random.seed(arguments.seed)

    # dataset
    train_dataset = MNIST(root=paths['datasets'],
                          train=True,
                          download=True,
                          transform=Compose([Resize((32, 32)),
                                             ToTensor(),
                                             Normalize((0.1307,), (0.3081,))]))

    test_dataset = MNIST(root=paths['datasets'],
                         train=False,
                         transform=Compose([Resize((32, 32)),
                                            ToTensor(),
                                            Normalize((0.1307,), (0.3081,))]))

    # loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=arguments.train_batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=arguments.test_batch_size,
                             shuffle=False)

    # model
    model = CNN().to(device=device)

    # optimizer
    optimizer = None
    if arguments.optimizer == 'adam':
        optimizer = Adam(params=model.parameters(),
                         lr=arguments.learning_rate)

    elif arguments.optimizer == 'sgd':
        optimizer = SGD(params=model.parameters(),
                        lr=arguments.learning_rate,
                        momentum=arguments.momentum)

    print('{} CNN {}'.format('=' * 10, '=' * 10))

    # epochs
    for epoch_number in range(1, arguments.epochs + 1):
        # train
        print('Train: Epoch: [{epoch_number}/{epochs}]'.format(epoch_number=epoch_number,
                                                               epochs=arguments.epochs))
        train(model=model,
              device=device,
              loader=train_loader,
              optimizer=optimizer)

        # test
        print('Test: Epoch: [{epoch_number}/{epochs}]'.format(epoch_number=epoch_number,
                                                              epochs=arguments.epochs))
        test(model=model,
             device=device,
             loader=test_loader)

        # save epoch file
        epoch_file = 'model_epoch_{epoch_number}.pkl'.format(epoch_number=epoch_number)
        epoch_file_path = os.path.join(paths['epochs'], epoch_file)
        torch.save(obj=model.state_dict(),
                   f=epoch_file_path)
        print('"{epoch_file}" file is saved as "{epoch_file_path}".'.format(epoch_file=epoch_file,
                                                                            epoch_file_path=epoch_file_path))


if __name__ == '__main__':
    main()
