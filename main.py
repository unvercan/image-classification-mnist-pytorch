# imports
import argparse
import numpy as np
import os
import torch
from abc import ABC
from torch.nn import Conv2d, MaxPool2d, Dropout, ReLU, Linear, BatchNorm2d, Module, Softmax
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


# model
class CNN(Module, ABC):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = Conv2d(in_channels=1,
                             out_channels=4,
                             stride=1,
                             kernel_size=5,
                             padding=0)  # 1x32x32 -> 4x28x28
        self.bn_1 = BatchNorm2d(4)
        self.relu_1 = ReLU(inplace=False)
        self.pool_1 = MaxPool2d(stride=2,
                                kernel_size=2,
                                padding=0)  # 4x28x28 -> 4x14x14
        self.drop_1 = Dropout(0.1)
        self.conv_2 = Conv2d(in_channels=4,
                             out_channels=8,
                             stride=1,
                             kernel_size=5,
                             padding=0)  # 4x14x14 -> 8x10x10
        self.bn_2 = BatchNorm2d(8)
        self.relu_2 = ReLU(inplace=False)
        self.pool_2 = MaxPool2d(stride=2,
                                kernel_size=2,
                                padding=0)  # 8x10x10 -> 8x5x5
        self.drop_2 = Dropout(0.1)
        self.fc = Linear(in_features=(8 * 5 * 5),
                         out_features=10)  # 8x5x5 -> 10
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)
        x = x.view(-1, (8 * 5 * 5))  # [Batch, Channel, Height, Width] -> [Batch, (Channel * Height * Width)]
        x = self.fc(x)
        x = self.softmax(x)
        return x


# train
def train(model, device, loader, optimizer):
    batches = len(loader)
    samples = len(loader.dataset)
    model.train()
    batch_losses = []
    batch_corrects = []
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
        batch_losses.append(batch_loss)
        batch_prediction = batch_output.max(dim=1, keepdim=True)[1]
        batch_correct = batch_prediction.eq(batch_target.view_as(other=batch_prediction)).sum().item()
        batch_corrects.append(batch_correct)
        optimizer.step()
        print('Batch: [{batch_no}/{batches}], '
              'Processed=[{processed_samples}/{samples}], '
              'Loss={batch_loss:.6f}'.format(batch_no=batch_no,
                                             batches=batches,
                                             processed_samples=processed_samples,
                                             samples=samples,
                                             batch_loss=batch_loss))
    epoch_loss = sum(batch_losses) / samples
    epoch_correct = sum(batch_corrects)
    epoch_accuracy = 100.0 * (sum(batch_corrects) / samples)
    print('Epoch:',
          'Loss={epoch_loss:.4f}, '
          'Correct=[{epoch_correct}/{samples}], '
          'Accuracy={epoch_accuracy:.0f}%'.format(epoch_loss=epoch_loss,
                                                  epoch_correct=epoch_correct,
                                                  epoch_accuracy=epoch_accuracy,
                                                  samples=samples))


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
                  'Processed=[{processed_samples}/{samples}], '
                  'Loss={batch_loss:.6f}, '
                  'Correct=[{batch_correct}/{batch_size}], '
                  'Accuracy={batch_accuracy:.0f}%'.format(batch_no=batch_no,
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
        print('Epoch:',
              'Loss={epoch_loss:.4f}, '
              'Correct=[{epoch_correct}/{samples}], '
              'Accuracy={epoch_accuracy:.0f}%'.format(epoch_loss=epoch_loss,
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
    argument_parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    argument_parser.add_argument('--cuda', type=bool, default=False, help='enable CUDA training')
    argument_parser.add_argument('--seed', type=int, default=1, help='random seed')

    # arguments
    arguments = argument_parser.parse_args(args=[])

    # cuda
    use_cuda = arguments.cuda and torch.cuda.is_available()

    # device
    device = torch.device('cuda' if use_cuda else 'cpu')

    # seed
    torch.manual_seed(seed=arguments.seed)
    np.random.seed(arguments.seed)

    # mean
    mean = 0.1307

    # standard deviation
    std = 0.3081

    # dataset
    train_dataset = MNIST(root=paths['datasets'],
                          train=True,
                          download=True,
                          transform=Compose([Resize((32, 32)),
                                             ToTensor(),
                                             Normalize(mean=mean, std=std)]))

    test_dataset = MNIST(root=paths['datasets'],
                         train=False,
                         transform=Compose([Resize((32, 32)),
                                            ToTensor(),
                                            Normalize(mean=mean, std=std)]))

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
    optimizer = Adam(params=model.parameters(),
                     lr=arguments.learning_rate)

    # info
    print('{} CNN on MNIST {}'.format('=' * 45, '=' * 45))

    # epochs
    for epoch_number in range(1, arguments.epochs + 1):
        # train info
        print('Train: Epoch: [{epoch_number}/{epochs}]'.format(epoch_number=epoch_number,
                                                               epochs=arguments.epochs))

        # train
        train(model=model,
              device=device,
              loader=train_loader,
              optimizer=optimizer)

        # break
        print('{}'.format('*' * 100))

        # test info
        print('Test: Epoch: [{epoch_number}/{epochs}]'.format(epoch_number=epoch_number,
                                                              epochs=arguments.epochs))

        # test
        test(model=model,
             device=device,
             loader=test_loader)

        # break
        print('{}'.format('*' * 100))

        # save epoch file
        epoch_file = 'model_epoch_{epoch_number}.pkl'.format(epoch_number=epoch_number)
        epoch_file_path = os.path.join(paths['epochs'], epoch_file)
        torch.save(obj=model.state_dict(),
                   f=epoch_file_path)

        # save info
        print('"{epoch_file}" file is saved as "{epoch_file_path}".'.format(epoch_file=epoch_file,
                                                                            epoch_file_path=epoch_file_path))

        # break
        print('{}'.format('*' * 100))


if __name__ == '__main__':
    main()
