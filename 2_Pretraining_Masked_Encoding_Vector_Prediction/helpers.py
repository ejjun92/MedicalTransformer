import numpy as np
from keras.datasets import mnist
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from glob import glob
import pickle
import random
import numpy as np
# import nibabel as nib
import os
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn import metrics
import nibabel as nib

from torch import nn

# Get current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def make_one_hot(y):
    return F.one_hot(y)

# Define the function to print and write log
def writelog(file, line):
    file.write(line + '\n')
    print(line)

# Load MNIST data using keras
def load_mnist(flatten=True, validation=True):
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize to 0-1
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Flatten out the data
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # Validation set
    if validation:
        return (x_train, y_train), (x_test, y_test)
    else:
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))

        return (x, y)


def get_metric_multi(y, y_score):
    y_pred = np.argmax(y_score, axis=1)

    acc = accuracy_score(y_pred, y) * 100
    a, b, c, \
    d, e, f, \
    g, h, i = confusion_matrix(y, y_pred, labels=[0, 1, 2]).ravel()
    tn, fp, fn, tp = (a + c + g + i), (b + h), (d + f), (e)
    # total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = 0, 0, 0
        auprc = 0
    spec = np.nan_to_num(tn / (tn + fp))
    # acc = ((tn + tp) / total) * 100
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    auc = MAUC(np.concatenate((y.reshape(-1,1), y_score), axis=1), num_classes=3)

    return auc, auprc, acc, balacc, sen, spec, prec, recall

def get_metric_binary(y, y_score):
    y_pred = np.argmax(y_score, axis=1)

    acc = accuracy_score(y_pred, y) * 100
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    # total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = precision_recall_curve(y, y_score[:,1:])
        auprc = np.nan_to_num(metrics.auc(r, p))
    spec = np.nan_to_num(tn / (tn + fp))
    # acc = ((tn + tp) / total) * 100
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    try:
        auc = roc_auc_score(y, y_score[:,1:])
    except ValueError:
        auc = 0

# Define sample loader
# def sample_loader(phase, images, labels, indexes, args, mean=None, std=None, shuffle=True, drop_last=False):
def sample_loader(phase, images, labels, indexes, args, shuffle=True, drop_last=False):

    # Random Seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    phase_dict = {
        'train': 0,
        'valid': 1,
    }

    if args.pre_dataset == 'ixi_camcan_abide':
        index = indexes[phase_dict[phase]]
        images_tensor = torch.from_numpy(images)  # (566, 193, 229, 193) -> [566, 96, 114, 96]

        if args.is_pool == 1:
            m = nn.AvgPool3d(2, stride=2)
            images_tensor = m(images_tensor)
        else:
            pass

        records = [{'data': images_tensor[i]} for i in index]
    elif args.pre_dataset == 'ADNI':
        index = indexes[phase_dict[phase]][args.fold]
        index = np.random.choice(len(index), int(len(index) * args.sample_ratio), replace=False)

        images_tensor = torch.from_numpy(images)  # (566, 193, 229, 193) -> [566, 96, 114, 96]

        if args.is_pool == 1:
            m = nn.AvgPool3d(2, stride=2)
            images_tensor = m(images_tensor)
        else:
            pass

        records = [{'data': images_tensor[i], 'label': torch.from_numpy(np.array(labels[i]))} for i in index]
    else:
        pass
    # Normalization
    # records = [{'data': (torch.from_numpy(images[i]) - torch.from_numpy(images[i]).min()) /
    #                     (torch.from_numpy(images[i]).max() - torch.from_numpy(images[i]).min()),
    #             'label': torch.from_numpy(np.array(labels[i]))} for i in index]
    # Take normalization
    # if phase == 'train':
    #     # Records data
    #     records = [{'data': torch.from_numpy(images[i])} for i in index]
    #     data = torch.cat([records[i]['data'].unsqueeze(0) for i in range(len(index))], dim=0)
    #     mean = torch.mean(data, dim=(1,2,3))
    #     std = torch.std(data, dim=(1,2,3))
    # records = [{'data': (torch.from_numpy(images[i]) - mean) / std, 'label': torch.from_numpy(np.array(labels[i]))} for i in index]

    # records['data'] -= mean
    # records['data'] /= std

    # # Get Dimensionality
    # N = index.shape[0]
    # data = images[index]
    # label = labels[index]
    #
    # records = []
    # for i in range(N):
    #     # Define records
    #     rec = {}
    #     rec['data'] = data[i]
    #     rec['label'] = label[i]
    #
    #     records.append(rec)

    # Define loader
    loader = DataLoader(records,
                        batch_size=args.batch_size,
                        num_workers=1,
                        shuffle=shuffle,
                        pin_memory=True,
                        drop_last=drop_last)
    records = None
    # return loader, mean, std
    return loader


def slice_loader(phase, images, labels, indexes, args, shuffle=True, drop_last=False):

    # Random Seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    phase_dict = {
        'train': 0,
        'valid': 1,
        'test': 2,
    }

    index = indexes[phase_dict[phase]][args.fold]
    # idx = index.tolist()

    b, num_sag, num_cor, num_axial = images.shape
    # images_ = images[index]

    # images_ = [torch.from_numpy(images[i]) for i in index]
    images_ = [torch.from_numpy(images[i][:, :, j]) for j in range(num_axial) for i in index]
    # labels_ = labels[index]


    # records = [{'data': torch.from_numpy(images[i]), 'label': torch.from_numpy(np.array(labels[i]))} for i in index]
    # Normalization
    # records = [{'data': (torch.from_numpy(images[i]) - torch.from_numpy(images[i]).min()) /
    #                     (torch.from_numpy(images[i]).max() - torch.from_numpy(images[i]).min()),
    #             'label': torch.from_numpy(np.array(labels[i]))} for i in index]
    # Take normalization
    # if phase == 'train':
    #     # Records data
    #     records = [{'data': torch.from_numpy(images[i])} for i in index]
    #     data = torch.cat([records[i]['data'].unsqueeze(0) for i in range(len(index))], dim=0)
    #     mean = torch.mean(data, dim=(1,2,3))
    #     std = torch.std(data, dim=(1,2,3))
    # records = [{'data': (torch.from_numpy(images[i]) - mean) / std, 'label': torch.from_numpy(np.array(labels[i]))} for i in index]

    # records['data'] -= mean
    # records['data'] /= std

    # # Get Dimensionality
    # N = index.shape[0]
    # data = images[index]
    # label = labels[index]
    #
    # records = []
    # for i in range(N):
    #     # Define records
    #     rec = {}
    #     rec['data'] = data[i]
    #     rec['label'] = label[i]
    #
    #     records.append(rec)

    # Define loader
    loader = DataLoader(images_,
                        batch_size=args.batch_size,
                        num_workers=1,
                        shuffle=shuffle,
                        pin_memory=True,
                        drop_last=drop_last)
    images_ = None
    # return loader, mean, std
    return loader


def visualize_reconstructions(dir, epoch, X, X_recon):
    rcParams['figure.figsize'] = 12, 8

    s1, s2, s3 = X.shape
    cs1 = np.ceil(s1 / 2)
    cs2 = np.ceil(s2 / 2)
    cs3 = np.ceil(s3 / 2)
    cs1, cs2, cs3

    xs = np.hstack([np.arange(0, cs1, 20), np.arange(cs1, s1, 20)]).astype('int32')
    ys = np.hstack([np.arange(0, cs2, 25), np.arange(cs2, s2, 25)]).astype('int32')
    zs = np.hstack([np.arange(0, cs3, 20), np.arange(cs3, s3, 20)]).astype('int32')
    indexes = [xs, ys, zs]

    fg, ax = plt.subplots(nrows=6, ncols=10)
    for i in range(3):
        for j, idx in enumerate(indexes[i]):
            img = None
            if i == 0:
                img = np.rot90(X[idx, :, :])
                img_recon = np.rot90(X_recon[idx, :, :])
            elif i == 1:
                img = np.rot90(X[:, idx, :])
                img_recon = np.rot90(X_recon[:, idx, :])
            else:
                img = np.rot90(X[:, :, idx])
                img_recon = np.rot90(X_recon[:, :, idx])

            ax[(i * 2), j].imshow(img, cmap='gray')
            ax[(i * 2) + 1, j].imshow(img_recon, cmap='gray')

            ax[(i * 2), j].axis('off')
            ax[(i * 2) + 1, j].axis('off')
    stre = str(epoch)
    plt.subplots_adjust(hspace=0.05, wspace=0.075)
    plt.savefig(dir + '/X_' + stre.rjust(4, '0') + '.png',
                bbox_inches='tight', dpi=300)
    fg.clear()
    plt.close(fg)

def visualize_reconstructions_pretrain(dir, epoch, X, X_recon):
    rcParams['figure.figsize'] = 12, 8
    s1, s2, s3 = X.shape
    cs1 = np.ceil(s1 / 2)
    cs2 = np.ceil(s2 / 2)
    # cs3 = np.ceil(s3 / 2)
    # cs1, cs2, cs3

    xs = np.hstack([np.arange(0, cs1, 20), np.arange(cs1, s1, 20)]).astype('int32')
    ys = np.hstack([np.arange(0, cs2, 25), np.arange(cs2, s2, 25)]).astype('int32')
    # zs = np.hstack([np.arange(0, cs3, 20), np.arange(cs3, s3, 2)]).astype('int32')
    zs = np.arange(0, 10)
    indexes = [xs, ys, zs]

    fg, ax = plt.subplots(nrows=6, ncols=10)
    for i in range(3):
        for j, idx in enumerate(indexes[i]):
            # print(i,j,idx)
            img = None
            if i == 0:
                img = np.rot90(X[idx, :, :])
                img_recon = np.rot90(X_recon[idx, :, :])
            elif i == 1:
                img = np.rot90(X[:, idx, :])
                img_recon = np.rot90(X_recon[:, idx, :])
            else:
                img = np.rot90(X[:, :, idx])
                img_recon = np.rot90(X_recon[:, :, idx])

            ax[(i * 2), j].imshow(img, cmap='gray')
            ax[(i * 2) + 1, j].imshow(img_recon, cmap='gray')

            ax[(i * 2), j].axis('off')
            ax[(i * 2) + 1, j].axis('off')
    stre = str(epoch)
    plt.subplots_adjust(hspace=0.05, wspace=0.075)
    plt.savefig(dir + '/X_' + stre.rjust(4, '0') + '.png',
                bbox_inches='tight', dpi=300)
    fg.clear()
    plt.close(fg)

def visualize_pretrain_reconstructions(dir, epoch, X, X_recon):
    rcParams['figure.figsize'] = 12, 8

    # s1, s2, s3 = X.shape
    b, s1, s2 = X.shape
    # cs1 = np.ceil(s1 / 2)
    # cs2 = np.ceil(s2 / 2)
    # cs3 = np.ceil(s3 / 2)
    # cs1, cs2, cs3

    # xs = np.hstack([np.arange(0, cs1, 20), np.arange(cs1, s1, 20)]).astype('int32')
    # ys = np.hstack([np.arange(0, cs2, 25), np.arange(cs2, s2, 25)]).astype('int32')
    # zs = np.hstack([np.arange(0, cs3, 20), np.arange(cs3, s3, 20)]).astype('int32')
    # indexes = [xs, ys, zs]

    fg, ax = plt.subplots(nrows=4, ncols=5)
    for i in range(2):
    # for i in range(1):
        for j in range(5):
            img = None
            img = np.rot90(X[(i * 2)+j, :, :])
            img_recon = np.rot90(X_recon[(i * 2)+j, :, :])

            ax[(i * 2), j].imshow(img, cmap='gray')
            ax[(i * 2) + 1, j].imshow(img_recon, cmap='gray')

            ax[(i * 2), j].axis('off')
            ax[(i * 2) + 1, j].axis('off')
    # for i in range(3):
    #     for j, idx in enumerate(indexes[i]):
    #         img = None
    #         if i == 0:
    #             img = np.rot90(X[idx, :, :])
    #             img_recon = np.rot90(X_recon[idx, :, :])
    #         elif i == 1:
    #             img = np.rot90(X[:, idx, :])
    #             img_recon = np.rot90(X_recon[:, idx, :])
    #         else:
    #             img = np.rot90(X[:, :, idx])
    #             img_recon = np.rot90(X_recon[:, :, idx])
    #
    #         ax[(i * 2), j].imshow(img, cmap='gray')
    #         ax[(i * 2) + 1, j].imshow(img_recon, cmap='gray')
    #
    #         ax[(i * 2), j].axis('off')
    #         ax[(i * 2) + 1, j].axis('off')
    stre = str(epoch)
    plt.subplots_adjust(hspace=0.05, wspace=0.075)
    plt.savefig(dir + '/X_' + stre.rjust(4, '0') + '.png',
                bbox_inches='tight', dpi=300)
    fg.clear()
    plt.close(fg)

import itertools
def a_value(probabilities, zero_label=0, one_label=1):
    """
    Approximates the AUC by the method described in Hand and Till 2001,
    equation 3.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        probabilities (list): A zipped list of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        zero_label (optional, int): The label to use as the class '0'.
            Must be an integer, see above for details.
        one_label (optional, int): The label to use as the class '1'.
            Must be an integer, see above for details.
    Returns:
        The A-value as a floating point.
    """
    # Obtain a list of the probabilities for the specified zero label class
    expanded_points = []
    for instance in probabilities:
        if instance[0] == zero_label or instance[0] == one_label:
            expanded_points.append((instance[0].item(), instance[zero_label+1].item()))
    sorted_ranks = sorted(expanded_points, key=lambda x: x[1])

    n0, n1, sum_ranks = 0, 0, 0
    # Iterate through ranks and increment counters for overall count and ranks of class 0
    for index, point in enumerate(sorted_ranks):
        if point[0] == zero_label:
            n0 += 1
            sum_ranks += index + 1  # Add 1 as ranks are one-based
        elif point[0] == one_label:
            n1 += 1
        else:
            pass  # Not interested in this class

    return (sum_ranks - (n0*(n0+1)/2.0)) / float(n0 * n1)  # Eqn 3

def MAUC(data, num_classes):
    """
    Calculates the MAUC over a set of multi-class probabilities and
    their labels. This is equation 7 in Hand and Till's 2001 paper.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        data (list): A zipped list (NOT A GENERATOR) of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        num_classes (int): The number of classes in the dataset.
    Returns:
        The MAUC as a floating point value.
    """
    # Find all pairwise comparisons of labels
    class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]

    # Have to take average of A value with both classes acting as label 0 as this
    # gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (a_value(data, zero_label=pairing[0], one_label=pairing[1]) + a_value(data, zero_label=pairing[1], one_label=pairing[0])) / 2.0

    return sum_avals * (2 / float(num_classes * (num_classes-1)))  # Eqn 7

def get_metric_multi(y, y_score):
    y_pred = np.argmax(y_score, axis=1)

    acc = accuracy_score(y_pred, y) * 100
    a, b, c, \
    d, e, f, \
    g, h, i = confusion_matrix(y, y_pred, labels=[0, 1, 2]).ravel()
    tn, fp, fn, tp = (a + c + g + i), (b + h), (d + f), (e)
    # total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = 0, 0, 0
        auprc = 0
    spec = np.nan_to_num(tn / (tn + fp))
    # acc = ((tn + tp) / total) * 100
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    auc = MAUC(np.concatenate((y.reshape(-1,1), y_score), axis=1), num_classes=3)

    return auc, auprc, acc, balacc, sen, spec, prec, recall

def get_metric_binary(y, y_score):
    y_pred = np.argmax(y_score, axis=1)

    acc = accuracy_score(y_pred, y) * 100
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    # total = tn + fp + fn + tp
    if tp == 0 and fn == 0:
        sen = 0.0
        recall = 0.0
        auprc = 0.0
    else:
        sen = tp / (tp + fn)
        recall = tp / (tp + fn)
        p, r, t = precision_recall_curve(y, y_score[:,1:])
        auprc = np.nan_to_num(metrics.auc(r, p))
    spec = np.nan_to_num(tn / (tn + fp))
    # acc = ((tn + tp) / total) * 100
    balacc = ((spec + sen) / 2) * 100
    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = np.nan_to_num(tp / (tp + fp))

    try:
        auc = roc_auc_score(y, y_score[:,1:])
    except ValueError:
        auc = 0

    return auc, auprc, acc, balacc, sen, spec, prec, recall

# def calculate_performance(y, y_score):
#     metric = {
#         'CN_MCI': None,
#         'CN_AD': None,
#         'MCI_AD': None,
#         'CN_MCI_AD': None,
#     }
#
#     # Get Indexing
#     index_cn_mci = np.where(y != 2)[0]
#     index_cn_ad = np.where(y != 1)[0]
#     index_mci_ad = np.where(y != 0)[0]
#
#     # CN vs MCI
#     y_gt = y.copy()[index_cn_mci]
#     metric['CN_MCI'] = get_metric_binary(y_gt,
#                                          y_score.copy()[index_cn_mci][:, (0, 1)])
#
#     # CN vs AD
#     y_gt = y.copy()[index_cn_ad]
#     y_gt[y_gt == 2] = 1
#     metric['CN_AD'] = get_metric_binary(y_gt,
#                                         y_score.copy()[index_cn_ad][:, (0, 2)])
#
#     # MCI vs AD
#     y_gt = y.copy()[index_mci_ad]
#     y_gt[y_gt == 1] = 0
#     y_gt[y_gt == 2] = 1
#     metric['MCI_AD'] = get_metric_binary(y_gt,
#                                          y_score.copy()[index_mci_ad][:, (1, 2)])
#
#     # mAUC = MAUC(zip(y, y_pred), 3)
#     metric['CN_MCI_AD'] = get_metric_multi(y, y_score)
#
#     return metric

def calculate_performance(y, y_score, args):
    metric = {}

    if args.class_scenario == 'cn_mc_ad':
        metric[args.class_scenario] = get_metric_multi(y, y_score)
    else:
        metric[args.class_scenario] = get_metric_binary(y, y_score)

    return metric

def topographic_error(d, map_size):
    """
    Calculate SOM topographic error (internal DESOM function)
    Topographic error is the ratio of data points for which the two best matching units are not neighbors on the map.
    """
    h, w = map_size

    def is_adjacent(k, l):
        return (abs(k//w-l//w) == 1 and abs(k % w - l % w) == 0) or (abs(k//w-l//w) == 0 and abs(k % w - l % w) == 1)
    btmus = np.argsort(d, axis=1)[:, :2]  # best two matching units
    return torch.from_numpy(np.array(1.-np.mean([is_adjacent(btmus[i, 0], btmus[i, 1]) for i in range(d.shape[0])]))).cuda()


def visualize2d(dir, epoch):
    ae.eval()
    som.eval()

    with torch.no_grad():
        # SOM Visualization
        prototypes = som.module.get_prototypes()
        dim, nprototypes = prototypes.shape
        decoded_prototypes = np.array([])
        for k in range(args.som_map_size[0] * args.som_map_size[1]):
            _, _, _, _, dec_proto = ae.module.decode(prototypes[:,k].view(1, dim, 1, 1, 1))
            if k == 0:
                decoded_prototypes = dec_proto.squeeze(1).to('cpu').detach().numpy()
            else:
                decoded_prototypes = np.vstack([decoded_prototypes, dec_proto.squeeze(1).to('cpu').detach().numpy()])

        s1, s2, s3 = decoded_prototypes[0].shape
        cs1 = np.ceil(s1 / 2).astype('int32')
        cs2 = np.ceil(s2 / 2).astype('int32')
        cs3 = np.ceil(s3 / 2).astype('int32')

        xs = np.hstack([np.arange(20, cs1, 20), np.arange(cs1, s1-20, 20)]).astype('int32')
        ys = np.hstack([np.arange(25, cs2, 25), np.arange(cs2, s2-25, 25)]).astype('int32')
        zs = np.hstack([np.arange(20, cs3, 20), np.arange(cs3, s3-20, 20)]).astype('int32')

        # Sagittal
        for i, xs_idx in enumerate(xs):
            id = 0
            fig, ax = plt.subplots(args.som_map_size[0], args.som_map_size[1], figsize=(10, 10))
            for j in range(args.som_map_size[0]):
                for k in range(args.som_map_size[1]):
                    ax[j, k].imshow(np.rot90(decoded_prototypes[id][xs_idx, :, :]), cmap='gray')
                    ax[j, k].axis('off')
                    id += 1
            plt.subplots_adjust(hspace=0.05, wspace=0.05)
            plt.savefig(dir + 'img/som/sagittal/som_sagittal_%d_%d.png' % (epoch, i), bbox_inches='tight')
            fig.clear()
            plt.close(fig)

        # Coronal
        for i, ys_idx in enumerate(ys):
            id = 0
            fig, ax = plt.subplots(args.som_map_size[0], args.som_map_size[1], figsize=(10, 10))
            for j in range(args.som_map_size[0]):
                for k in range(args.som_map_size[1]):
                    ax[j, k].imshow(np.rot90(decoded_prototypes[id][:, ys_idx, :]), cmap='gray')
                    ax[j, k].axis('off')
                    id += 1
            plt.subplots_adjust(hspace=0.05, wspace=0.05)
            plt.savefig(dir + 'img/som/coronal/som_coronal_%d_%d.png' % (epoch, i), bbox_inches='tight')
            fig.clear()
            plt.close(fig)

        # Axial
        for i, zs_idx in enumerate(zs):
            id = 0
            fig, ax = plt.subplots(args.som_map_size[0], args.som_map_size[1], figsize=(10, 10))
            for j in range(args.som_map_size[0]):
                for k in range(args.som_map_size[1]):
                    ax[j, k].imshow(np.rot90(decoded_prototypes[id][:, :, zs_idx]), cmap='gray')
                    ax[j, k].axis('off')
                    id += 1
            plt.subplots_adjust(hspace=0.05, wspace=0.05)
            plt.savefig(dir + 'img/som/axial/som_axial_%d_%d.png' % (epoch, i), bbox_inches='tight')
            fig.clear()
            plt.close(fig)

def visualize1d(dir, epoch):
    ae.eval()
    som.eval()

    with torch.no_grad():
        # SOM Visualization
        prototypes = som.module.get_prototypes()
        dim, nprototypes = prototypes.shape
        decoded_prototypes = np.array([])
        for k in range(args.som_map_size[0] * args.som_map_size[1]):
            _, _, _, _, dec_proto = ae.module.decode(prototypes[:,k].view(1, dim, 1, 1, 1))
            if k == 0:
                decoded_prototypes = dec_proto.squeeze(1).to('cpu').detach().numpy()
            else:
                decoded_prototypes = np.vstack([decoded_prototypes, dec_proto.squeeze(1).to('cpu').detach().numpy()])

        s1, s2, s3 = decoded_prototypes[0].shape
        cs1 = np.ceil(s1 / 2).astype('int32')
        cs2 = np.ceil(s2 / 2).astype('int32')
        cs3 = np.ceil(s3 / 2).astype('int32')

        xs = np.hstack([np.arange(20, cs1, 20), np.arange(cs1, s1-20, 20)]).astype('int32')
        ys = np.hstack([np.arange(25, cs2, 25), np.arange(cs2, s2-25, 25)]).astype('int32')
        zs = np.hstack([np.arange(20, cs3, 20), np.arange(cs3, s3-20, 20)]).astype('int32')

        # Sagittal
        for i, xs_idx in enumerate(xs):
            id = 0
            fig, ax = plt.subplots(args.som_map_size[0], args.som_map_size[1], figsize=(10, 10))
            for j in range(args.som_map_size[0]):
                for k in range(args.som_map_size[1]):
                    ax[j, k].imshow(np.rot90(decoded_prototypes[id][xs_idx, :, :]), cmap='gray')
                    ax[j, k].axis('off')
                    id += 1
            plt.subplots_adjust(hspace=0.05, wspace=0.05)
            plt.savefig(dir + 'img/som/sagittal/som_sagittal_%d_%d.png' % (epoch, i), bbox_inches='tight')
            fig.clear()
            plt.close(fig)

        # Coronal
        for i, ys_idx in enumerate(ys):
            id = 0
            fig, ax = plt.subplots(args.som_map_size[0], args.som_map_size[1], figsize=(10, 10))
            for j in range(args.som_map_size[0]):
                for k in range(args.som_map_size[1]):
                    ax[j, k].imshow(np.rot90(decoded_prototypes[id][:, ys_idx, :]), cmap='gray')
                    ax[j, k].axis('off')
                    id += 1
            plt.subplots_adjust(hspace=0.05, wspace=0.05)
            plt.savefig(dir + 'img/som/coronal/som_coronal_%d_%d.png' % (epoch, i), bbox_inches='tight')
            fig.clear()
            plt.close(fig)

        # Axial
        for i, zs_idx in enumerate(zs):
            id = 0
            fig, ax = plt.subplots(args.som_map_size[0], args.som_map_size[1], figsize=(10, 10))
            for j in range(args.som_map_size[0]):
                for k in range(args.som_map_size[1]):
                    ax[j, k].imshow(np.rot90(decoded_prototypes[id][:, :, zs_idx]), cmap='gray')
                    ax[j, k].axis('off')
                    id += 1
            plt.subplots_adjust(hspace=0.05, wspace=0.05)
            plt.savefig(dir + 'img/som/axial/som_axial_%d_%d.png' % (epoch, i), bbox_inches='tight')
            fig.clear()
            plt.close(fig)

class EarlyStopping():
    """
    Early Stopping to terminate training early under certain conditions
    """
    def __init__(self, delta=0, patience=5, verbose = True):
        self.delta = delta
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.best_mean_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = 0
        self.wait = 0
        self.stopped_epoch = 0
        super(EarlyStopping, self).__init__()

    def __call__(self, val_loss, val_acc):
        if self.early_stop == False:
            if val_loss != None:
                if self.best_loss is None:
                    self.best_loss = val_loss
                # better model has been found.
                if val_loss < self.best_loss + self.delta:
                    self.best_loss = val_loss
                    self.counter = 0
                # saved model is better.
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
            else:
                if self.best_acc is None:
                    self.best_acc= val_acc
                # better model has been found.
                if val_acc > self.best_acc + self.delta:
                    self.best_acc = val_acc
                    self.counter = 0
                # saved model is better.
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True

            if self.verbose == True:
                print(f'Early Stopping counter : {self.counter} out of {self.patience}')
        else:
            pass


def save_featureMap_tensor(tensor, dirToSave = './', name='test'):
    # tensor [w, h, d]
    tmp_dir = dirToSave +'/featuremap'
    if os.path.exists(tmp_dir) == False:
        os.makedirs(tmp_dir)
    tmp_array = tensor.data.cpu().numpy()
    f_img = nib.Nifti1Image(tmp_array, np.eye(4))
    nib.save(f_img, os.path.join(tmp_dir + '/'+ name+ '.nii.gz'))