from __future__ import print_function
import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchnet import meter

# Import models
import pretrainedmodels
import models.torchvision_models as models

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

parser = argparse.ArgumentParser(description='DCNNs transfer training')
parser.add_argument('--src_dataset', dest='src_dataset', default='flowers17', type=str,
                    help='dataset (options: flowers17, flowers102, plant, plankton, qut)')
parser.add_argument('--dst_dataset', dest='dst_dataset', default='flowers17', type=str,
                    help='the destination domain dataset')
parser.add_argument('--train_image_size', dest='train_image_size', default=224, type=int,
                    help='image size for training (default: 224)')
parser.add_argument('--test_image_size', dest='test_image_size', default=256, type=int,
                    help='image size for testing (default: 256)')
parser.add_argument('--test_crop_image_size', dest='test_crop_image_size', default=224, type=int,
                    help='image size for testing after cropping (default: 224)')
parser.add_argument('--model', dest='model', default='AlexNet', type=str,
                    help='model type (options: AlexNet')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--alpha', default=32, type=int,
                    help='number of new channel increase per depth (default: 12)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basic block (default: bottleneck)')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual start epoch number (useful on restarts)')
parser.add_argument('--b', '--batchsize', dest='batchsize', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='to use basicblock (default: bottleneck)')
parser.add_argument('--basemodel', default='', type=str,
                    help='path to the pretrained basemodel')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--expname', default='PyramidNet', type=str,
                    help='name of experiment')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--gpu_ids', default=0, help='gpu ids: e.g. 0  0,1,2, 0,2.')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test the trained model')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pretrained imagenet model')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
parser.set_defaults(verbose=True)
parser.set_defaults(test=False)
parser.set_defaults(pretrained=False)

best_err1 = 100


def main():
    global args, best_err1
    args = parser.parse_args()

    # TensorBoard configure
    if args.tensorboard:
        configure('transfer_from_{}_to_{}_checkpoints/{}'.format(args.src_dataset, args.dst_dataset, args.expname))

    # CUDA
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)
    if torch.cuda.is_available():
        cudnn.benchmark = True  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        kwargs = {'num_workers': 2, 'pin_memory': True}
    else:
        kwargs = {'num_workers': 2}

    # Source dataset
    if args.src_dataset == 'flowers17':
        src_num_classes = 17
    elif args.src_dataset == 'flowers102':
        src_num_classes = 102
    elif args.src_dataset == 'plant-0' or args.src_dataset == 'plant-1' or args.src_dataset == 'plant-2'\
            or args.src_dataset == 'plant-3' or args.src_dataset == 'plant-4':
        src_num_classes = 12
    elif args.src_dataset == 'plankton':
        src_num_classes = 121
    elif args.src_dataset == 'qut-0' or args.src_dataset == 'qut-1':
        src_num_classes = 482
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(args.src_dataset))

    # Data loading code
    if args.dst_dataset == 'plankton':
        normalize = transforms.Normalize(mean=[0.9044, 0.9044, 0.9044],
                                         std=[0.1864, 0.1864, 0.1864])
    elif args.dst_dataset == 'flowers17':
        normalize = transforms.Normalize(mean=[0.4920, 0.4749, 0.3117],
                                         std=[0.2767, 0.2526, 0.2484])
    elif args.dst_dataset == 'flowers102':
        normalize = transforms.Normalize(mean=[0.5208, 0.4205, 0.3441],
                                         std=[0.2944, 0.2465, 0.2735])
    elif args.dst_dataset == 'plant-0' or 'plant-1' or 'plant-2' or 'plant-3' or 'plant-4':
        normalize = transforms.Normalize(mean=[0.3209, 0.2885, 0.1951],
                                         std=[0.0964, 0.1011, 0.1161])
    elif args.dst_dataset == 'qut-0' or 'qut-1':
        normalize = transforms.Normalize(mean=[0.4700, 0.4936, 0.4471],
                                         std=[0.2967, 0.2767, 0.2817])
    else:
        raise Exception('Unknown dataset: {}'.format(args.dst_dataset))

    # Transforms
    if args.augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.train_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.train_image_size),
            transforms.ToTensor(),
            normalize,
        ])
    val_transform = transforms.Compose([
        transforms.Resize(args.test_image_size),
        transforms.CenterCrop(args.test_crop_image_size),
        transforms.ToTensor(),
        normalize
    ])

    # Datasets
    if args.dst_dataset == 'plankton':
        train_dataset = datasets.ImageFolder('datasets/plankton-set/train/',
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder('datasets/plankton-set/test/',
                                           transform=val_transform)
        test_dataset = None
        num_classes = 121
    elif args.dst_dataset == 'flowers17':
        train_dataset = datasets.ImageFolder('datasets/flowers17/train/',
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder('datasets/flowers17/val/',
                                           transform=val_transform)
        test_dataset = datasets.ImageFolder('datasets/flowers17/test/',
                                            transform=val_transform)
        num_classes = 17
    elif args.dst_dataset == 'flowers102':
        train_dataset = datasets.ImageFolder('datasets/flowers102/train/',
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder('datasets/flowers102/val/',
                                           transform=val_transform)
        test_dataset = datasets.ImageFolder('datasets/flowers102/test/',
                                            transform=val_transform)
        num_classes = 102
    elif args.dst_dataset == 'plant-0' or args.dst_dataset == 'plant-1' or args.dst_dataset == 'plant-2' \
            or args.dst_dataset == 'plant-3' or args.dst_dataset == 'plant-4':
        fold_num = args.dst_dataset.split('-')[1]
        train_dataset = datasets.ImageFolder(
            'datasets/plant-seedlings/plant-{}/train/'.format(fold_num),
            transform=train_transform)
        val_dataset = datasets.ImageFolder(
            'datasets/plant-seedlings/plant-{}/test/'.format(fold_num),
            transform=val_transform)
        test_dataset = None
        num_classes = 12
    elif args.dst_dataset == 'qut-0' or args.dst_dataset == 'qut-1':
        fold_num = args.dst_dataset.split('-')[1]
        train_dataset = datasets.ImageFolder(
            'datasets/qut-fish/qut-{}/train/'.format(fold_num),
            transform=train_transform)
        val_dataset = datasets.ImageFolder(
            'datasets/qut-fish/qut-{}/test/'.format(fold_num),
            transform=val_transform)
        test_dataset = None
        num_classes = 482
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(args.dst_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, **kwargs)
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, **kwargs)
    else:
        test_loader = val_loader

    # Create model
    if args.model in pretrainedmodels.model_names:
        if args.pretrained:
            model = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')
            dim_features = model.last_linear.in_features
            model.last_linear = nn.Linear(dim_features, src_num_classes)
        else:
            model = models.__dict__[args.model](num_classes=1000)
            dim_features = model.last_linear.in_features
            model.last_linear = nn.Linear(dim_features, src_num_classes)
    else:
        raise NotImplementedError('Unsupport model: {}. Option: {}'
                        .format(args.model, pretrainedmodels.model_names))

    if args.basemodel:
        if os.path.isfile(args.basemodel):
            print("==> loading checkpoint '{}'".format(args.basemodel))
            checkpoint = torch.load(args.basemodel)
            model.load_state_dict(checkpoint['state_dict'])
            dim_features = model.last_linear.in_features
            model.last_linear = nn.Linear(dim_features, num_classes)
    else:
        raise Exception('Unsupport basemodel {} at {}'.format(args.model, args.basemodel))

    # Get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_err1 = checkpoint['best_err1']
            model.load_state_dict(checkpoint['state_dict'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    print(model)

    if torch.cuda.is_available():
        model.cuda()

    # Define loss function (criterion
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # Test model
    if args.test:
        err1, cm = test(test_loader, model, criterion, num_classes)
        # save `cm` array to disk
        directory = 'transfer_from_{}_to_{}_checkpoints/{}/'.format(args.src_dataset, args.dst_dataset, args.expname)
        np.save(directory + 'confusion_matrix.npy', cm)
        # plot confusion matrix and save the fig
        plot_and_save_confusion_matrix(cm, num_classes, path=directory + 'confusion_matrix.png', normalize=True)
        # compute average precision, recall and F1 score and wirte them into txt file
        average_precision, average_recall, average_f1_score = compute_precision_recall_f1_score(cm)

        # print results
        if args.pretrained:
            pretrained = '-pretrained'
        else:
            pretrained = ''

        print('Test result of {}{} on {}'.format(args.model, pretrained, args.dst_dataset))
        print('top-1 error: {:.4f}'.format(err1))
        print('top-1 accuracy: {:.4f}'.format(100 - err1))
        print('precision: {:.4f}'.format(average_precision))
        print('recall: {:.4f}'.format(average_recall))
        print('F1 score: {:.4f}'.format(average_f1_score))

        # write results to text file
        with open(directory + 'result.txt', 'w') as f:
            f.write('Test result of {}{} on {}\n'.format(args.model, pretrained, args.dst_dataset))
            f.write('top-1 error: {:.4f}\n'.format(err1))
            f.write('top-1 accuracy: {:.4f}\n'.format(100 - err1))
            f.write('precision: {:.4f}\n'.format(average_precision))
            f.write('recall: {:.4f}\n'.format(average_recall))
            f.write('F1 score: {:.4f}\n'.format(average_f1_score))

        return

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # Evaluate on validation set
        err1 = validate(val_loader, model, criterion, epoch)

        # Remember best err1 and save checkpoint
        is_best = (err1 <= best_err1)
        best_err1 = min(err1, best_err1)
        print("Current best accuracy (error):", best_err1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
        }, is_best)

    print("Best accuracy (error):", best_err1)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Train for one epoch on the training set
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs_var = to_var(inputs)
        targets_var = to_var(targets)

        # Compute output
        outputs = model(inputs_var)
        # For Inception-v3, the output may be a tuple
        if type(outputs) is tuple:
            outputs = outputs[0]
        loss = criterion(outputs, targets_var)

        # Measure accuracy and record loss
        err1, err5 = accuracy(outputs.data.cpu(), targets, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(err1[0], inputs.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top 1-err {top1.val:.3f} ({top1.avg:.3f})'
                  .format(epoch + 1, i, len(train_loader), batch_time=batch_time,
                          loss=losses, top1=top1))

    # Log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_error', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    """
    Perform validation on the validation set
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        inputs_var = to_var(inputs)
        targets_var = to_var(targets)

        # Compute output
        outputs = model(inputs_var)
        loss = criterion(outputs, targets_var)

        # Measure accuracy and record loss
        err1, err5 = accuracy(outputs.data.cpu(), targets, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(err1[0], inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top 1-err {top1.val:.3f} ({top1.avg:.3f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, top1=top1))

    print('Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}'
          .format(epoch + 1, args.epochs, top1=top1))
    # Log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)

    return top1.avg


def test(test_loader, model, criterion, num_classes):
    """
    Perform validation on the validation set
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    confusion_matrix = meter.ConfusionMeter(num_classes)

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(test_loader):
        inputs_var = to_var(inputs)
        targets_var = to_var(targets)

        # Compute output
        outputs = model(inputs_var)
        loss = criterion(outputs, targets_var)
        confusion_matrix.add(outputs.data.squeeze().view(-1, outputs.shape[1]),
                             targets.type(torch.LongTensor))

        # Measure accuracy and record loss
        err1, err5 = accuracy(outputs.data.cpu(), targets, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(err1[0], inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top 1-err {top1.val:.3f} ({top1.avg:.3f})'
                  .format(i, len(test_loader), batch_time=batch_time,
                          loss=losses, top1=top1))

    cm_value = confusion_matrix.value()
    print('Test top-1 error: {top1.avg:.3f}'.format(top1=top1))
    # Log to TensorBoard
    if args.tensorboard:
        log_value('test_loss', losses.avg)
        log_value('test_acc', top1.avg)

    return top1.avg, cm_value


def to_var(x):
    """
    Convert tensor x to autograd Variable
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save checkpoint to disk
    """
    directory = 'transfer_from_{}_to_{}_checkpoints/{}/'.format(args.src_dataset, args.dst_dataset, args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        'transfer_from_{}_to_{}_checkpoints/{}/'.format(args.src_dataset, args.dst_dataset, args.expname)
                        + 'model_best.pth.tar')


class AverageMeter(object):
    """
    Compute and store the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def adjust_learning_rate(optimizer, epoch):
    """
    Adjust the learning rate
    """
    lr = args.lr * (0.1 ** (epoch // 100))

    # lr = args.lr

    # Log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """
    Computes the error@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


def plot_and_save_confusion_matrix(cm, number_of_class,
                                   normalize=False,
                                   title='Confusion matrix',
                                   path='confusion_matrix.png',
                                   cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float')
        non_zero_indices = (cm.sum(axis=1) > 0)
        cm[non_zero_indices] = cm[non_zero_indices] / cm[non_zero_indices].sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(0, number_of_class, 10)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)

    # thresh = cm.max() / 2.

    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, '',
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual labels')
    plt.xlabel('Predicted labels')
    plt.tight_layout()
    plt.savefig(path)


def compute_precision_recall_f1_score(confusion_matrix):
    """Compute the average precision, recall and F1-score"""
    sum_precision = 0
    sum_recall = 0
    sum_f1_score = 0
    for i in range(confusion_matrix.shape[0]):
        true_pos = confusion_matrix[i, i]
        true_and_false_pos = np.sum(confusion_matrix[i, :])
        true_pos_and_false_neg = np.sum(confusion_matrix[:, [i]])

        if true_pos == 0:
            if true_and_false_pos == 0 and true_pos_and_false_neg == 0:
                precision = 1
                recall = 1
                f1_score = 1
            else:
                if true_pos_and_false_neg == 0:
                    precision = 1
                    recall = 0
                elif true_and_false_pos == 0:
                    precision = 0
                    recall = 1
                else:
                    precision = 0
                    recall = 0

                f1_score = 0
        else:
            precision = float(true_pos) / true_and_false_pos
            recall = float(true_pos) / true_pos_and_false_neg
            f1_score = float(2 * precision * recall) / (precision + recall)

        sum_precision += precision
        sum_recall += recall
        sum_f1_score += f1_score

    average_precision = float(sum_precision) / confusion_matrix.shape[0]
    average_recall = float(sum_recall) / confusion_matrix.shape[0]
    average_f1_score = float(sum_f1_score) / confusion_matrix.shape[0]

    return average_precision, average_recall, average_f1_score


if __name__ == '__main__':
    main()
