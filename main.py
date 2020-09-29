# imports
from abc import ABC
from argparse import ArgumentParser
from collections import OrderedDict
from os import path, makedirs
import numpy as np
import torch
from torch.nn import Conv2d, MaxPool2d, Dropout, ReLU, Linear, BatchNorm2d, Module, Softmax, Flatten, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


# train epoch
def train_epoch(model, device, loader, optimizer, make_prediction=False, show_batch_info=True, show_epoch_info=True):
    number_of_batches = len(loader)
    model.train()
    epoch_info = dict()
    epoch_info['samples'] = len(loader.dataset)
    batch_infos = []
    processed_samples = 0
    for batch_index, batch in enumerate(iterable=loader):
        batch_data, batch_target = batch
        batch_info = train_batch(batch_data=batch_data, batch_target=batch_target, model=model,
                                 device=device, optimizer=optimizer, make_prediction=make_prediction)
        batch_infos.append(batch_info)
        processed_samples += batch_info['size']
        if show_batch_info:
            print('Batch: [{batch_no}/{number_of_batches}], '
                  'Processed=[{processed_samples}/{samples}], '
                  'Loss={batch_loss:.6f}'
                  .format(batch_no=(batch_index + 1), number_of_batches=number_of_batches,
                          processed_samples=processed_samples, samples=epoch_info['samples'],
                          batch_loss=batch_info['loss']))
    epoch_info['batches'] = batch_infos
    epoch_info['loss'] = sum(batch_info['loss'] for batch_info in batch_infos) / epoch_info['samples']
    if make_prediction:
        epoch_info['correct'] = sum(batch_info['correct'] for batch_info in batch_infos)
        epoch_info['accuracy'] = 100.0 * (epoch_info['correct'] / epoch_info['samples'])
    if show_epoch_info:
        print('Epoch: Samples={epoch_samples}, Loss={epoch_loss:.4f}'
              .format(epoch_loss=epoch_info['loss'], epoch_samples=epoch_info['samples']))
    return epoch_info


# train batch
def train_batch(batch_data, batch_target, model, device, optimizer, make_prediction=False):
    batch_info = dict()
    batch_info['size'] = len(batch_data)
    batch_data, batch_target = batch_data.to(device=device), batch_target.to(device=device)
    optimizer.zero_grad()
    if batch_data.dim() == 3:
        batch_data = batch_data.unsqueeze(dim=1)  # [batch, height, width] -> [batch, channel, height, width]
    batch_output = model(batch_data)
    batch_loss = cross_entropy(input=batch_output, target=batch_target)
    batch_loss.backward()
    batch_info['loss'] = batch_loss.item()
    if make_prediction:
        batch_prediction = batch_output.max(dim=1, keepdim=True)[1]
        batch_info['correct'] = batch_prediction.eq(batch_target.view_as(other=batch_prediction)).sum().item()
        batch_info['accuracy'] = 100. * batch_info['correct'] / batch_info['size']
    optimizer.step()
    return batch_info


# test epoch
def test_epoch(model, device, loader):
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
            batch_loss = cross_entropy(input=batch_output, target=batch_target).item()
            batch_losses.append(batch_loss)
            batch_prediction = batch_output.max(dim=1, keepdim=True)[1]
            batch_correct = batch_prediction.eq(batch_target.view_as(other=batch_prediction)).sum().item()
            batch_corrects.append(batch_correct)
            batch_accuracy = 100. * batch_correct / batch_size
            print('Batch: [{batch_no}/{batches}], '
                  'Processed=[{processed_samples}/{samples}], '
                  'Loss={batch_loss:.6f}, '
                  'Correct=[{batch_correct}/{batch_size}], '
                  'Accuracy={batch_accuracy:.0f}%'.format(batch_no=batch_no, batches=batches,
                                                          processed_samples=processed_samples, samples=samples,
                                                          batch_size=batch_size, batch_correct=batch_correct,
                                                          batch_accuracy=batch_accuracy, batch_loss=batch_loss))

        epoch_loss = sum(batch_losses) / samples
        epoch_correct = sum(batch_corrects)
        epoch_accuracy = 100.0 * (sum(batch_corrects) / samples)
        print('Epoch:',
              'Loss={epoch_loss:.4f}, '
              'Correct=[{epoch_correct}/{samples}], '
              'Accuracy={epoch_accuracy:.0f}%'.format(epoch_loss=epoch_loss, epoch_correct=epoch_correct,
                                                      epoch_accuracy=epoch_accuracy, samples=samples))


# model
class CNN(Module, ABC):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature_extraction = Sequential(
            OrderedDict([
                ('conv_1', Conv2d(in_channels=1, out_channels=4, stride=1, kernel_size=5, padding=0)),  # 4x28x28
                ('bn_1', BatchNorm2d(num_features=4)),
                ('relu_1', ReLU(inplace=False)),
                ('pool_1', MaxPool2d(stride=2, kernel_size=2, padding=0)),  # 4x14x14
                ('drop_1', Dropout(p=0.1)),
                ('conv_2', Conv2d(in_channels=4, out_channels=8, stride=1, kernel_size=5, padding=0)),  # 8x10x10
                ('bn_2', BatchNorm2d(num_features=8)),
                ('relu_2', ReLU(inplace=False)),
                ('pool_2', MaxPool2d(stride=2, kernel_size=2, padding=0)),  # 8x5x5
                ('drop_2', Dropout(p=0.1)),
                ('flatten', Flatten(start_dim=1))
            ])
        )
        self.classifier = Sequential(
            OrderedDict([
                ('fc', Linear(in_features=(8 * 5 * 5), out_features=10)),  # 10
                ('softmax', Softmax(dim=1))
            ])
        )

    def forward(self, x):
        features = self.feature_extraction(x)
        classes = self.classifier(features)
        return classes


# main
def main():
    # paths
    paths = dict()
    paths['project'] = '.'
    paths['dataset'] = path.join(paths['project'], 'dataset')
    paths['weight'] = path.join(paths['project'], 'weight')
    paths['train'] = path.join(paths['project'], 'train')

    # make directories if not exist
    for key in paths.keys():
        value = paths[key]
        if not path.exists(value):
            makedirs(value)

    # settings
    argument_parser = ArgumentParser()

    # argument parser
    argument_parser.add_argument('--train-batch-size', type=int, default=2000, help='batch size for training')
    argument_parser.add_argument('--test-batch-size', type=int, default=2000, help='batch size for testing')
    argument_parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    argument_parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    argument_parser.add_argument('--cuda', type=bool, default=True, help='enable CUDA training')
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

    # dataset mean
    mean = 0.1307

    # dataset standard deviation
    std = 0.3081

    # dataset
    train_dataset = MNIST(root=paths['dataset'], train=True, download=True,
                          transform=Compose([Resize((32, 32)), ToTensor(), Normalize(mean=mean, std=std)]))

    test_dataset = MNIST(root=paths['dataset'], train=False,
                         transform=Compose([Resize((32, 32)), ToTensor(), Normalize(mean=mean, std=std)]))

    # loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=arguments.train_batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=arguments.test_batch_size, shuffle=False)

    # model
    model = CNN().to(device=device)

    # optimizer
    optimizer = Adam(params=model.parameters(), lr=arguments.learning_rate)

    # info
    print('{} CNN on MNIST {}'.format('=' * 45, '=' * 45))
    print('Feature extraction: {feature_extraction_layers}'.format(feature_extraction_layers=model.feature_extraction))
    print('Classifier: {classifier_layers}'.format(classifier_layers=model.classifier))

    # break line
    print('{}'.format('*' * 100))

    # epochs
    for epoch_number in range(1, arguments.epochs + 1):
        # train info
        print('Train: Epoch: [{epoch_number}/{epochs}]'.format(epoch_number=epoch_number, epochs=arguments.epochs))

        # train epoch
        train_info = train_epoch(model=model, device=device, loader=train_loader, optimizer=optimizer)

        # break line
        print('{}'.format('*' * 100))

        # save weight
        weight_file = 'mnist_cnn_weight_epoch_{epoch_number}.pkl'.format(epoch_number=epoch_number)
        weight_file_path = path.join(paths['weight'], weight_file)
        torch.save(obj=model.state_dict(), f=weight_file_path)
        print('"{weight_file}" file is saved as "{weight_file_path}".'
              .format(weight_file=weight_file, weight_file_path=weight_file_path))

        # break line
        print('{}'.format('*' * 100))

        # save train info
        train_info_file = 'mnist_cnn_train_epoch_{epoch_number}.npy'.format(epoch_number=epoch_number)
        train_info_file_path = path.join(paths['train'], train_info_file)
        np.save(file=train_info_file_path, arr=np.array(train_info))
        print('"{train_info_file}" file is saved as "{train_info_file_path}".'
              .format(train_info_file=train_info_file, train_info_file_path=train_info_file_path))

        # break line
        print('{}'.format('*' * 100))

        # test info
        print('Test: Epoch: [{epoch_number}/{epochs}]'.format(epoch_number=epoch_number, epochs=arguments.epochs))

        # test epoch
        test_epoch(model=model, device=device, loader=test_loader)

        # break line
        print('{}'.format('*' * 100))


if __name__ == '__main__':
    main()
