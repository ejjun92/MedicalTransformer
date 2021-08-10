import torch.optim as optim
# from models.ResNet_Model import Multiview_MEP
from models.ResNet_Model import Plane_Feature_extraction
from models.ResNet import generate_model as generate_model_3d
from models.ResNet_2d import resnet2d as resnet2d
from opts import parse_opts
from losses import *
from helpers import *
import os
import random
import datetime
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
# from config import models_genesis_config
from tqdm import tqdm

# Define Arguments
args = parse_opts()

# GPU Configuration
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Dataset
images = np.load('/DataCommon/ejjun/dataset/IXI_camcan_ABIDE/output/data_norm_min_max.npy', mmap_mode="r", allow_pickle=True)  # (566, 193, 229, 193)
indexes = np.load('/DataCommon/ejjun/dataset/IXI_camcan_ABIDE/output/10idx.npy', allow_pickle=True)  # (509,), (57,)

# ixi_images = np.load('/DataCommon/ejjun/dataset/IXI/output/data_norm_min_max.npy', mmap_mode="r", allow_pickle=True)  # (566, 193, 229, 193)
# camcan_images = np.load('/DataCommon/ejjun/dataset/camcan/output/data_norm_min_max.npy', mmap_mode="r", allow_pickle=True)  # (653, 193, 229, 193)
# abide_images = np.load('/DataCommon/ejjun/dataset/ABIDE/output/data_norm_min_max.npy', mmap_mode="r", allow_pickle=True)  # (564, 193, 229, 193)
# images = np.concatenate((ixi_images, camcan_images, abide_images), axis=0)
# np.save('/DataCommon/ejjun/dataset/IXI_camcan_ABIDE/output/data_norm_min_max.npy', images)
labels = []
# ixi_indexes = np.load('/DataCommon/ejjun/dataset/IXI/output/10idx.npy', allow_pickle=True)  # (509,), (57,)
# camcan_indexes = np.load('/DataCommon/ejjun/dataset/camcan/output/10idx.npy', allow_pickle=True)  # (587,), (66,)
# abide_indexes = np.load('/DataCommon/ejjun/dataset/ABIDE/output/10idx.npy', allow_pickle=True)  # (507,), (57,)
#
# indexes = [np.concatenate((ixi_indexes[0], camcan_indexes[0]+np.array(len(ixi_images)), abide_indexes[0]+np.array(len(ixi_images))+np.array(len(camcan_images))), axis=0),
#            np.concatenate((ixi_indexes[1], camcan_indexes[1]+np.array(len(ixi_images)), abide_indexes[1]+np.array(len(ixi_images))+np.array(len(camcan_images))), axis=0)]
# np.save('/DataCommon/ejjun/dataset/IXI_camcan_ABIDE/output/10idx.npy', indexes)
# Logging purpose
date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
directory = './log/%s/%s/%s_batch_%d_lr_%f_lambda2_%f_ResNet_%d_inplanes_%d/' % (args.pre_dataset, args.approach, date_str, args.batch_size, args.lr, args.lambda2, args.depth, args.inplanes)

if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + 'img/')
    os.makedirs(directory + 'img/train/')
    os.makedirs(directory + 'img/valid/')
    os.makedirs(directory + 'img/test/')
    os.makedirs(directory + 'tflog/')
    os.makedirs(directory + 'model/')

# Text Logging
f = open(directory + 'setting.log', 'a')
writelog(f, '======================')
# writelog(f, 'Model: %s' % (args.model))
writelog(f, 'Lambda2: %.5f' % args.lambda2)
writelog(f, '----------------------')
writelog(f, 'Fold: %d' % args.fold)
writelog(f, 'Learning Rate: %.5f' % args.lr)
writelog(f, 'Batch Size: %d' % args.batch_size)
writelog(f, 'Epoch: %d' % args.epoch)
writelog(f, '======================')
f.close()

f = open(directory + 'log.log', 'a')
# Tensorboard Logging
# tfw_train = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/train_')
# tfw_valid = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/valid_')
# tfw_test = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/test_')
tfw_train = tf.compat.v1.summary.FileWriter(directory + 'tflog/train_')
tfw_valid = tf.compat.v1.summary.FileWriter(directory + 'tflog/valid_')

# Tensor Seed
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

# Define Loaders
writelog(f, 'Load training data')
train_loader = sample_loader('train', images, labels, indexes, args, drop_last=True)
writelog(f, 'Load validation data')
# valid_loader = sample_loader('valid', images, labels, indexes, args, shuffle=False)
valid_loader = sample_loader('valid', images, labels, indexes, args, shuffle=False, drop_last=True)
# writelog(f, 'Load test data')
# test_loader = sample_loader('test', images, labels, indexes, args, shuffle=False)
dataloaders = {'train': train_loader,
               'valid': valid_loader}
               # 'test': test_loader}

if args.approach == '25d':
    model = nn.DataParallel(Plane_Feature_extraction(args)).to(device)
elif args.approach == '2d':
    model = nn.DataParallel(resnet2d(args)).to(device)
elif args.approach == '3d':
    model = nn.DataParallel(generate_model_3d(model_depth=args.model_depth,
                                              inplanes=args.inplanes,
                                              n_classes=args.d_f,
                                              n_input_channels=1,
                                              shortcut_type=args.resnet_shortcut,
                                              conv1_t_size=args.conv1_t_size,
                                              conv1_t_stride=args.conv1_t_stride,
                                              no_max_pool=args.no_max_pool,
                                              widen_factor=args.resnet_widen_factor)).to(device)
else:
    pass

criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, last_epoch=-1)

# Define data type
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

# Best epoch checking
valid = {
    'epoch': 0,
    'auc': 0,
}


def train(dataloader, dir='.'):
    model.train()

    # Define training variables
    train_loss = 0
    n_batches = 0

    # Loop over the minibatch
    for i, xbatch in enumerate(tqdm(dataloader)):
        x = Variable(xbatch['data'].float(), requires_grad=True).cuda()  # [B, 193, 229, 193]

        optimizer.zero_grad()

        if args.approach == '25d':
            # anchor
            a_cor = x.clone().permute(0, 1, 3, 2)  # [B, 193, 193, 229]
            a_sag = x.clone().permute(0, 3, 2, 1)  # [B, 193, 229, 193]
            a_axial = x.clone()  # [B, 193, 229, 193]

            # positive examples
            p_cor = nonlinear_transformation(a_cor.cpu().detach().numpy())
            p_sag = nonlinear_transformation(a_sag.cpu().detach().numpy())
            p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())

            # negative examples
            n_cor = torch.cat([a_cor[1:], a_cor[0].unsqueeze(0)], dim=0)
            n_sag = torch.cat([a_sag[1:], a_sag[0].unsqueeze(0)], dim=0)
            n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)

            # anchor
            encoded_a_cor = model(a_cor, plane='cor')
            encoded_a_sag = model(a_sag, plane='sag')
            encoded_a_axial = model(a_axial, plane='axial')

            # positive examples
            encoded_p_cor = model(torch.from_numpy(p_cor).float().cuda(), plane='cor')
            encoded_p_sag = model(torch.from_numpy(p_sag).float().cuda(), plane='sag')
            encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda(), plane='axial')

            # negative examples
            encoded_n_cor = model(n_cor, plane='cor')
            encoded_n_sag = model(n_sag, plane='sag')
            encoded_n_axial = model(n_axial, plane='axial')

            loss_cor = criterion(encoded_a_cor, encoded_p_cor, encoded_n_cor)
            loss_sag = criterion(encoded_a_sag, encoded_p_sag, encoded_n_sag)
            loss_axial = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)
            loss = loss_cor + loss_sag + loss_axial

        elif args.approach == '2d':
            # anchor
            a_axial = x  # [2, 96, 114, 96]

            # positive examples
            p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())  # [2, 96, 114, 96]

            # negative examples
            n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)  # [2, 96, 114, 96]

            encoded_a_axial = model(a_axial)  # [2, 64]
            encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda())
            encoded_n_axial = model(n_axial)

            loss = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)

        elif args.approach == '3d':
            # anchor
            a_axial = x  # [2, 96, 114, 96]

            # positive examples
            p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())  # [2, 96, 114, 96]

            # negative examples
            n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)  # [2, 96, 114, 96]

            encoded_a_axial = model(a_axial)  # [2, 64]
            encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda())
            encoded_n_axial = model(n_axial)

            loss = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)
        else:
            pass

        loss.backward()
        optimizer.step()

        print('Training loss = (%.5f)' % loss)
        train_loss += (loss.item() * x.size(0))
        n_batches += 1

    # Take average
    # train_loss = train_loss / n_batches
    train_loss = train_loss / len(dataloader.dataset)
    writelog(f, 'Train Loss: %.8f' % train_loss)

    # Tensorboard Logging
    info = {
            'loss': train_loss,
           }
    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        tfw_train.add_summary(summary, epoch)


def evaluate(phase, dataloader, dir='.'):
    # Set mode as training
    model.eval()

    # Define training variables
    test_loss = 0
    n_batches = 0

    # No Grad
    with torch.no_grad():
        # Loop over the minibatch
        for i, xbatch in enumerate(tqdm(dataloader)):
            x = xbatch['data'].float().cuda()

            if args.approach == '25d':
                # anchor
                a_cor = x.clone().permute(0, 1, 3, 2)  # [B, 193, 193, 229]
                a_sag = x.clone().permute(0, 3, 2, 1)  # [B, 193, 229, 193]
                a_axial = x.clone()  # [B, 193, 229, 193]

                # positive examples
                p_cor = nonlinear_transformation(a_cor.cpu().detach().numpy())
                p_sag = nonlinear_transformation(a_sag.cpu().detach().numpy())
                p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())

                # negative examples
                n_cor = torch.cat([a_cor[1:], a_cor[0].unsqueeze(0)], dim=0)
                n_sag = torch.cat([a_sag[1:], a_sag[0].unsqueeze(0)], dim=0)
                n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)

                # anchor
                encoded_a_cor = model(a_cor, plane='cor')
                encoded_a_sag = model(a_sag, plane='sag')
                encoded_a_axial = model(a_axial, plane='axial')

                # positive examples
                encoded_p_cor = model(torch.from_numpy(p_cor).float().cuda(), plane='cor')
                encoded_p_sag = model(torch.from_numpy(p_sag).float().cuda(), plane='sag')
                encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda(), plane='axial')

                # negative examples
                encoded_n_cor = model(n_cor, plane='cor')
                encoded_n_sag = model(n_sag, plane='sag')
                encoded_n_axial = model(n_axial, plane='axial')

                loss_cor = criterion(encoded_a_cor, encoded_p_cor, encoded_n_cor)
                loss_sag = criterion(encoded_a_sag, encoded_p_sag, encoded_n_sag)
                loss_axial = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)
                loss = loss_cor + loss_sag + loss_axial

            elif args.approach == '2d':
                # anchor
                a_axial = x  # [2, 96, 114, 96]

                # positive examples
                p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())  # [2, 96, 114, 96]

                # negative examples
                n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)  # [2, 96, 114, 96]

                encoded_a_axial = model(a_axial)  # [2, 64]
                encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda())
                encoded_n_axial = model(n_axial)

                loss = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)

            elif args.approach == '3d':
                # anchor
                a_axial = x  # [2, 96, 114, 96]

                # positive examples
                p_axial = nonlinear_transformation(a_axial.cpu().detach().numpy())  # [2, 96, 114, 96]

                # negative examples
                n_axial = torch.cat([a_axial[1:], a_axial[0].unsqueeze(0)], dim=0)  # [2, 96, 114, 96]

                encoded_a_axial = model(a_axial)  # [2, 64]
                encoded_p_axial = model(torch.from_numpy(p_axial).float().cuda())
                encoded_n_axial = model(n_axial)

                loss = criterion(encoded_a_axial, encoded_p_axial, encoded_n_axial)
            else:
                pass

            print('(%s) loss = (%.5f)' % (phase, loss))
            test_loss += (loss.item() * x.size(0))
            n_batches += 1

    # Take average
    # test_loss = test_loss / n_batches
    test_loss = test_loss / len(dataloader.dataset)
    writelog(f, '%s Loss: %.8f' % (phase, test_loss))

    # Tensorboard Logging
    info = {'loss': test_loss}

    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        if phase == 'Validation':
            tfw_valid.add_summary(summary, epoch)
        # else:
        #     tfw_test.add_summary(summary, epoch)

    return test_loss

# Train Epoch
ES = EarlyStopping(delta=0, patience=30, verbose=True)
for epoch in range(args.epoch):
    writelog(f, '--- Epoch %d' % epoch)
    writelog(f, 'Training')
    train(dataloaders['train'], dir=directory)

    writelog(f, 'Validation')
    loss_val = evaluate('Validation', dataloaders['valid'], dir=directory)

    if epoch == 0:
        valid['loss'] = loss_val

    # Save Model
    if loss_val < valid['loss']:
        torch.save(model.state_dict(), directory + '/model/prepretrain_model.pt')
        # torch.save({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict()
        # }, os.path.join(directory + '/model', args.model + '_' + str(epoch) + '.pt'))
        print("Saving model ", os.path.join(directory + '/prepretrain_model.pt'))

        writelog(f, 'Best validation loss is found! Validation loss : %f' % loss_val)
        writelog(f, 'Models at Epoch %d are saved!' % epoch)
        valid['loss'] = loss_val
        valid['epoch'] = epoch
        ES(loss_val, None)

    # loss_test = evaluate('Test', dataloaders['test'], dir=directory)

    scheduler.step()
    if ES.early_stop == True:
        break

writelog(f, 'END OF TRAINING')
f.close()